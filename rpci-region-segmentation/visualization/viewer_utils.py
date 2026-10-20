"""Shared helpers for the rPCI Streamlit viewer."""

from __future__ import annotations

import os
import platform
import re
import subprocess
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np
import SimpleITK as sitk

from visualization.nnunet_model_layout import (
    dataset_id_for_predict,
    find_installed_datasets as _find_installed_datasets,
    normalize_dataset_root,
)
from visualization.plot_results import REGION_COLORS, REGION_LABELS

DEFAULT_NNUNET_ROOT = Path(__file__).resolve().parents[1] / "nnunet_data"


def ensure_nnunet_dirs(root: Path | None = None) -> dict[str, Path]:
    root = root or DEFAULT_NNUNET_ROOT
    paths = {
        "nnUNet_raw": root / "nnUNet_raw",
        "nnUNet_preprocessed": root / "nnUNet_preprocessed",
        "nnUNet_results": root / "nnUNet_results",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def nnunet_env_block(paths: dict[str, Path]) -> dict[str, str]:
    return {k: str(v) for k, v in paths.items()}


def validate_model_zip(zip_path: Path) -> tuple[bool, str]:
    if not zip_path.is_file():
        return False, f"文件不存在: {zip_path}"
    if zip_path.stat().st_size < 10_000:
        with open(zip_path, "rb") as f:
            head = f.read(64)
        if head.lstrip().startswith(b"<!") or head.lstrip().startswith(b"<html"):
            return False, (
                "该文件不是有效的模型压缩包（像是网页/HTML）。"
                "请从 GitHub Releases 用浏览器或 wget 重新下载完整的 Dataset101_PM.zip。"
            )
    if not zipfile.is_zipfile(zip_path):
        return False, "不是有效的 ZIP 文件，请重新下载 Dataset101_PM.zip。"
    return True, "ZIP 格式校验通过。"


def install_pretrained_zip(zip_path: Path, results_dir: Path) -> tuple[bool, str]:
    ok, msg = validate_model_zip(zip_path)
    if not ok:
        return False, msg

    try:
        proc = subprocess.run(
            [
                "nnUNetv2_install_pretrained_model_from_zip",
                str(zip_path),
            ],
            capture_output=True,
            text=True,
            env={**os.environ, "nnUNet_results": str(results_dir)},
            check=False,
        )
    except FileNotFoundError:
        return False, "未找到 nnUNetv2_install_pretrained_model_from_zip，请先: pip install nnunetv2"

    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        return False, f"模型安装失败:\n{detail}"

    for child in results_dir.iterdir():
        if child.is_dir() and child.name.startswith("Dataset"):
            normalize_dataset_root(child)
    return True, (proc.stdout or "模型已安装到 nnUNet_results。").strip()


def find_installed_datasets(results_dir: Path) -> list[str]:
    return _find_installed_datasets(results_dir, auto_normalize=True)


def list_nifti(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    files = sorted(folder.glob("*.nii.gz")) + sorted(folder.glob("*.nii"))
    return files


def segmentation_stem(path: Path) -> str:
    name = path.name
    for suffix in (".nii.gz", ".nii"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def match_ct_for_segmentation(seg_path: Path, ct_files: Iterable[Path]) -> Path | None:
    stem = segmentation_stem(seg_path)
    ct_list = list(ct_files)
    if not ct_list:
        return None

    candidates = [
        stem,
        f"{stem}_0000",
        stem.replace("Segmentations_", "Scan_"),
        re.sub(r"^segmentations_", "Scan_", stem, flags=re.I),
    ]
    by_stem = {segmentation_stem(p): p for p in ct_list}
    for key in candidates:
        if key in by_stem:
            return by_stem[key]
    for p in ct_list:
        if stem in segmentation_stem(p) or segmentation_stem(p) in stem:
            return p
    return ct_list[0] if len(ct_list) == 1 else None


def load_volume(path: Path) -> tuple[np.ndarray, sitk.Image]:
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)
    return arr, img


def extract_slice(
    volume: np.ndarray, plane: str, index: int
) -> np.ndarray:
    if plane == "axial":
        return volume[index, :, :]
    if plane == "coronal":
        return np.flipud(volume[:, index, :])
    if plane == "sagittal":
        return np.flipud(volume[:, :, index])
    raise ValueError(plane)


def plane_depth(volume: np.ndarray, plane: str) -> int:
    if plane == "axial":
        return volume.shape[0]
    if plane == "coronal":
        return volume.shape[1]
    if plane == "sagittal":
        return volume.shape[2]
    raise ValueError(plane)


def region_volume_stats(seg: np.ndarray) -> list[dict]:
    total = int(np.prod(seg.shape))
    rows = []
    for label in range(14):
        voxels = int((seg == label).sum())
        if label == 0 and voxels == 0:
            continue
        rows.append(
            {
                "label": label,
                "region": REGION_LABELS[label] if label < len(REGION_LABELS) else str(label),
                "voxels": voxels,
                "percent": 100.0 * voxels / total if total else 0.0,
            }
        )
    return rows


def run_nnunet_predict(
    *,
    input_dir: Path,
    output_dir: Path,
    dataset_name: str,
    configuration: str,
    folds: list[int],
    env: dict[str, str],
) -> tuple[bool, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_args = [str(f) for f in folds]
    device = "cuda" if os.environ.get("NNUNET_FORCE_CUDA") == "1" else "cpu"
    if device == "cuda" and platform.system() == "Darwin":
        device = "cpu"
    cmd = [
        "nnUNetv2_predict",
        "-i",
        str(input_dir),
        "-o",
        str(output_dir),
        "-d",
        dataset_id_for_predict(
            dataset_name,
            Path(env.get("nnUNet_results", "")) / dataset_name,
        ),
        "-c",
        configuration,
        "-device",
        device,
        "-f",
        *fold_args,
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env={**os.environ, **env},
            check=False,
        )
    except FileNotFoundError:
        return False, "未找到 nnUNetv2_predict，请先 pip install nnunetv2"

    log = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if proc.returncode != 0:
        return False, f"推理失败 (exit {proc.returncode}):\n{log.strip()}"
    return True, log.strip() or "推理完成。"


def dataset_id_from_name(name: str) -> str:
    m = re.match(r"Dataset(\d+)_", name)
    if m:
        return str(int(m.group(1)))
    return "101"
