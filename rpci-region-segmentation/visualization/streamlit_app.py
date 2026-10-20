"""Med_PMP: DICOM ZIP → nnU-Net → contours, volumes, PCI estimate."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_script_dir = str(Path(__file__).resolve().parent)
while _script_dir in sys.path:
    sys.path.remove(_script_dir)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from visualization.dicom_io import dicom_zip_to_nifti
from visualization.i18n import t

LANG = "en"
from visualization.pci_metrics import compute_pci_report
from visualization.rendering import render_slice_views
from visualization.nnunet_model_layout import install_from_local_folder
from visualization.viewer_utils import (
    DEFAULT_NNUNET_ROOT,
    ensure_nnunet_dirs,
    extract_slice,
    find_installed_datasets,
    install_pretrained_zip,
    load_volume,
    plane_depth,
    run_nnunet_predict,
    validate_model_zip,
)

FLUENT_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Segoe+UI:wght@400;600&display=swap');
html, body, [class*="css"] { font-family: "Segoe UI", -apple-system, BlinkMacSystemFont, sans-serif; }
.stApp {
  background: linear-gradient(145deg, #dbeef9 0%, #e8f4fc 35%, #f5faff 70%, #d4e8f7 100%);
}
.block-container {
  padding-top: 1.5rem;
  max-width: 1200px;
}
section[data-testid="stSidebar"],
[data-testid="collapsedControl"] {
  display: none;
}
section[data-testid="stMain"] > div {
  max-width: 100%;
}
.glass-panel {
  background: rgba(255, 255, 255, 0.62);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  border: 1px solid rgba(0, 120, 212, 0.2);
  border-radius: 12px;
  padding: 1rem 1.25rem;
  margin-bottom: 1rem;
  box-shadow: 0 4px 24px rgba(0, 120, 212, 0.08);
}
.metric-pill {
  background: linear-gradient(135deg, rgba(0,120,212,0.12), rgba(255,255,255,0.9));
  border-radius: 10px;
  padding: 0.75rem 1rem;
  border: 1px solid rgba(0,120,212,0.25);
}
h1, h2, h3 { color: #0078D4 !important; font-weight: 600 !important; }
.stButton > button {
  background: linear-gradient(180deg, #42a5f5 0%, #0078D4 100%);
  color: white;
  border: none;
  border-radius: 8px;
  font-weight: 600;
}
.stButton > button:hover { color: white; border: none; opacity: 0.92; }
</style>
"""


def inject_theme():
    st.markdown(FLUENT_CSS, unsafe_allow_html=True)


def page_settings():
    st.markdown(f'<div class="glass-panel">', unsafe_allow_html=True)
    st.subheader(t("tab_settings", LANG))
    paths = ensure_nnunet_dirs(DEFAULT_NNUNET_ROOT)
    st.code("\n".join(f"{k}={v}" for k, v in paths.items()), language="bash")
    zip_default = Path.home() / "Downloads" / "Dataset101_PM.zip"
    zip_path = Path(st.text_input(t("model_zip", LANG), value=str(zip_default))).expanduser()
    if zip_path.is_file():
        ok, msg = validate_model_zip(zip_path)
        st.success(msg) if ok else st.error(msg)
    if st.button(t("install_model", LANG)):
        with st.spinner("…"):
            ok, msg = install_pretrained_zip(zip_path, paths["nnUNet_results"])
        st.success(msg) if ok else st.error(msg)

    folder_default = Path.home() / "Downloads" / "Dataset101_PM"
    folder_path = Path(
        st.text_input(
            "Extracted model folder (fold_0, dataset.json)",
            value=str(folder_default) if folder_default.is_dir() else "",
        )
    ).expanduser()
    if st.button("Register local model folder"):
        with st.spinner("…"):
            ok, msg = install_from_local_folder(folder_path, paths["nnUNet_results"])
        st.success(msg) if ok else st.error(msg)

    datasets = find_installed_datasets(paths["nnUNet_results"])
    st.caption(t("installed_models", LANG))
    st.write(", ".join(datasets) if datasets else t("no_models", LANG))
    st.session_state["nnunet_paths"] = paths
    st.markdown("</div>", unsafe_allow_html=True)


def run_full_pipeline(zip_bytes: bytes, case_id: str) -> bool:
    paths = st.session_state.get("nnunet_paths") or ensure_nnunet_dirs(DEFAULT_NNUNET_ROOT)
    work = Path(tempfile.mkdtemp(prefix="rpci_case_"))
    ct_dir = work / "input"
    out_dir = work / "output"
    ct_dir.mkdir(parents=True, exist_ok=True)

    st.info(f"{t('step_dicom', LANG)}…")
    ct_nii, _sitk_ct = dicom_zip_to_nifti(zip_bytes, ct_dir / "ct.nii.gz", case_id=case_id)

    datasets = find_installed_datasets(paths["nnUNet_results"])
    if not datasets:
        st.error("nnU-Net model not installed. Install Dataset101_PM under Model setup.")
        return False

    st.info(f"{t('step_predict', LANG)}…")
    env = {k: str(v) for k, v in paths.items()}
    ok, log = run_nnunet_predict(
        input_dir=ct_dir,
        output_dir=out_dir,
        dataset_name=datasets[0],
        configuration="3d_lowres",
        folds=[0],
        env=env,
    )
    if not ok:
        st.error(log)
        return False

    seg_files = list(out_dir.glob("*.nii.gz"))
    if not seg_files:
        st.error("No segmentation output.")
        return False

    ct_vol, _ = load_volume(ct_nii)
    seg_vol, seg_img = load_volume(seg_files[0])
    pci_rows, total_pci = compute_pci_report(ct_vol, seg_vol.astype(int), seg_img, lang=LANG)

    st.session_state["ct_vol"] = ct_vol
    st.session_state["seg_vol"] = seg_vol
    st.session_state["pci_rows"] = pci_rows
    st.session_state["total_pci"] = total_pci
    st.session_state["case_id"] = case_id
    st.session_state["pipeline_ok"] = True
    return True


def page_workflow():
    st.markdown(f'<div class="glass-panel">', unsafe_allow_html=True)
    st.subheader(t("tab_workflow", LANG))
    st.caption(t("upload_hint", LANG))
    case_id = st.text_input(t("case_id", LANG), value="case001")
    upload = st.file_uploader(t("upload_dicom", LANG), type=["zip"])
    if upload and st.button(t("run_pipeline", LANG), type="primary"):
        with st.spinner(t("running", LANG)):
            ok = run_full_pipeline(upload.getvalue(), case_id.strip() or "case001")
        if ok:
            st.success(t("pipeline_ok", LANG))
    st.markdown("</div>", unsafe_allow_html=True)


def page_viewer():
    if not st.session_state.get("pipeline_ok"):
        st.markdown(f'<div class="glass-panel">', unsafe_allow_html=True)
        st.warning(t("no_result", LANG))
        example = _PROJECT_ROOT / "full_segmentation_analysis_3planes.png"
        if example.is_file():
            st.image(str(example), caption="Reference · Gort et al., rPCI segmentation")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    ct_vol = st.session_state["ct_vol"]
    seg_vol = st.session_state["seg_vol"]
    pci_rows = st.session_state["pci_rows"]
    total_pci = st.session_state["total_pci"]

    planes = {"axial": t("axial", LANG), "coronal": t("coronal", LANG), "sagittal": t("sagittal", LANG)}
    col_a, col_b, col_c = st.columns([2, 1, 1])
    with col_a:
        plane = st.radio(t("plane", LANG), list(planes.keys()), format_func=lambda k: planes[k], horizontal=True)
    with col_b:
        depth = plane_depth(ct_vol, plane)
        sl = st.slider(t("slice", LANG), 0, max(0, depth - 1), depth // 2)
    with col_c:
        alpha = st.slider(t("overlay_alpha", LANG), 0.0, 1.0, 0.38)
        show_contours = st.checkbox(t("show_contours", LANG), value=True)

    ct_sl = extract_slice(ct_vol, plane, sl)
    seg_sl = extract_slice(seg_vol.astype(int), plane, min(sl, plane_depth(seg_vol, plane) - 1))

    fig = render_slice_views(
        ct_sl,
        seg_sl,
        alpha=alpha,
        show_contours=show_contours,
        titles=(t("ct_only", LANG), t("overlay_view", LANG), t("contour_view", LANG)),
    )
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)

    st.markdown(f'<div class="glass-panel">', unsafe_allow_html=True)
    m1, m2 = st.columns(2)
    with m1:
        st.markdown(
            f'<div class="metric-pill"><h3 style="margin:0;color:#0078D4!important">{t("total_pci", LANG)}</h3>'
            f'<p style="font-size:2rem;margin:0;font-weight:700">{total_pci} / 39</p></div>',
            unsafe_allow_html=True,
        )
    with m2:
        vol_sum = sum(r["volume_ml"] for r in pci_rows if r["label"] > 0)
        st.markdown(
            f'<div class="metric-pill"><h3 style="margin:0;color:#0078D4!important">{t("volume_title", LANG)}</h3>'
            f'<p style="font-size:2rem;margin:0;font-weight:700">{vol_sum:.1f} mL</p>'
            f'<small>Σ rPCI compartments</small></div>',
            unsafe_allow_html=True,
        )

    st.subheader(t("pci_title", LANG))
    st.caption(t("pci_note", LANG))
    df = pd.DataFrame(pci_rows)
    df = df.rename(
        columns={
            "region": t("col_region", LANG),
            "volume_ml": t("col_volume", LANG),
            "pci_score": t("col_pci", LANG),
            "lesion_diameter_cm": t("col_lesion", LANG),
        }
    )
    st.dataframe(
        df[[t("col_region", LANG), t("col_volume", LANG), t("col_lesion", LANG), t("col_pci", LANG)]],
        use_container_width=True,
        hide_index=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="Med_PMP", page_icon="🩻", layout="wide")
    paths = ensure_nnunet_dirs(DEFAULT_NNUNET_ROOT)
    for key, val in paths.items():
        os.environ[key] = str(val)
    st.session_state.setdefault("nnunet_paths", paths)
    inject_theme()

    st.title(t("app_title", LANG))
    st.caption(t("app_subtitle", LANG))

    tab_w, tab_v, tab_s = st.tabs(
        [t("tab_workflow", LANG), t("tab_viewer", LANG), t("tab_settings", LANG)]
    )
    with tab_w:
        page_workflow()
    with tab_v:
        page_viewer()
    with tab_s:
        page_settings()


if __name__ == "__main__":
    main()
