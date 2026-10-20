"""UI strings for zh / en."""

from __future__ import annotations

from typing import Literal

Lang = Literal["en", "zh"]

_STRINGS: dict[str, dict[Lang, str]] = {
    "app_title": {
        "en": "Med_PMP",
        "zh": "Med_PMP",
    },
    "app_subtitle": {
        "en": "Peritoneal metastasis imaging · rPCI segmentation · volumes & PCI estimate",
        "zh": "Peritoneal metastasis imaging · rPCI segmentation · volumes & PCI estimate",
    },
    "lang_label": {"en": "Language", "zh": "语言"},
    "tab_workflow": {"en": "Upload & Analyze", "zh": "上传与分析"},
    "tab_viewer": {"en": "Viewer", "zh": "可视化"},
    "tab_settings": {"en": "Model setup", "zh": "模型配置"},
    "upload_dicom": {
        "en": "Upload CT DICOM series (ZIP)",
        "zh": "上传 CT DICOM 序列（ZIP）",
    },
    "upload_hint": {
        "en": "ZIP should contain one or more DICOM files (folders allowed). Portal-venous abdominal CT recommended.",
        "zh": "ZIP 内为 DICOM 文件（可含子文件夹），建议门静脉期腹部 CT。",
    },
    "case_id": {"en": "Case ID", "zh": "病例 ID"},
    "run_pipeline": {"en": "Convert DICOM & run segmentation", "zh": "转换 DICOM 并运行分割"},
    "running": {"en": "Processing…", "zh": "处理中…"},
    "step_dicom": {"en": "Reading DICOM", "zh": "读取 DICOM"},
    "step_nifti": {"en": "Writing NIfTI", "zh": "写入 NIfTI"},
    "step_predict": {"en": "nnU-Net inference", "zh": "nnU-Net 推理"},
    "pipeline_ok": {"en": "Segmentation complete.", "zh": "分割完成。"},
    "pipeline_fail": {"en": "Pipeline failed", "zh": "流程失败"},
    "no_result": {
        "en": "Upload a DICOM ZIP and run analysis to see contours and PCI.",
        "zh": "请上传 DICOM ZIP 并运行分析，以查看轮廓与 PCI。",
    },
    "plane": {"en": "Plane", "zh": "切面"},
    "axial": {"en": "Axial", "zh": "轴位"},
    "coronal": {"en": "Coronal", "zh": "冠状"},
    "sagittal": {"en": "Sagittal", "zh": "矢状"},
    "slice": {"en": "Slice", "zh": "层号"},
    "show_contours": {"en": "Show region contours", "zh": "显示区域轮廓"},
    "overlay_alpha": {"en": "Overlay opacity", "zh": "叠加透明度"},
    "pci_title": {"en": "PCI score (imaging estimate)", "zh": "PCI 评分（影像启发式估计）"},
    "pci_note": {
        "en": (
            "Anatomical regions come from the nnU-Net rPCI model. Lesion scores use CT density "
            "heuristics within each region (not a substitute for clinical PCI). Total PCI range 0–39."
        ),
        "zh": (
            "解剖分区来自 nnU-Net rPCI 模型。各区域病灶分基于 CT 密度启发式（不能替代临床 PCI）。"
            "总分范围 0–39。"
        ),
    },
    "total_pci": {"en": "Total PCI", "zh": "PCI 总分"},
    "volume_title": {"en": "Region volumes", "zh": "区域体积"},
    "col_region": {"en": "Region", "zh": "区域"},
    "col_volume": {"en": "Volume (mL)", "zh": "体积 (mL)"},
    "col_pci": {"en": "PCI score", "zh": "PCI 分"},
    "col_lesion": {"en": "Max lesion Ø (cm)", "zh": "最大病灶径 (cm)"},
    "model_zip": {"en": "Pretrained model ZIP", "zh": "预训练模型 ZIP"},
    "install_model": {"en": "Install Dataset101_PM", "zh": "安装 Dataset101_PM"},
    "installed_models": {"en": "Installed models", "zh": "已安装模型"},
    "no_models": {"en": "No nnU-Net models detected.", "zh": "未检测到 nnU-Net 模型。"},
    "ct_only": {"en": "CT", "zh": "CT"},
    "contour_view": {"en": "Contours", "zh": "轮廓"},
    "overlay_view": {"en": "Overlay", "zh": "叠加"},
}


def t(key: str, lang: Lang) -> str:
    block = _STRINGS.get(key)
    if not block:
        return key
    return block.get(lang, block["en"])
