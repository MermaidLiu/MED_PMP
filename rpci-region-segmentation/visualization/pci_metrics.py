"""Region volumes and imaging-based PCI heuristics."""

from __future__ import annotations

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

PCI_REGIONS_EN = [
    "0: Central",
    "1: Right upper",
    "2: Epigastrium",
    "3: Left upper",
    "4: Left flank",
    "5: Left lower",
    "6: Pelvis",
    "7: Right lower",
    "8: Right flank",
    "9: Upper jejunum",
    "10: Lower jejunum",
    "11: Upper ileum",
    "12: Lower ileum",
]

PCI_REGIONS_ZH = [
    "0: 中央",
    "1: 右上",
    "2: 上腹",
    "3: 左上",
    "4: 左腹",
    "5: 左下",
    "6: 盆腔",
    "7: 右下",
    "8: 右腹",
    "9: 空肠上段",
    "10: 空肠下段",
    "11: 回肠上段",
    "12: 回肠下段",
]


def _spacing_mm(image: sitk.Image) -> tuple[float, float, float]:
    sp = image.GetSpacing()
    return float(sp[2]), float(sp[1]), float(sp[0])  # z, y, x for array order


def pci_score_from_diameter_cm(diameter_cm: float) -> int:
    if diameter_cm <= 0:
        return 0
    if diameter_cm < 0.5:
        return 1
    if diameter_cm <= 5.0:
        return 2
    return 3


def _max_lesion_diameter_cm(
    region_mask: np.ndarray,
    ct: np.ndarray,
    spacing_zyx_mm: tuple[float, float, float],
    hu_threshold: float = 100.0,
) -> float:
    binary = region_mask & (ct >= hu_threshold)
    if not np.any(binary):
        return 0.0
    labeled, n = ndimage.label(binary)
    max_d = 0.0
    sz, sy, sx = spacing_zyx_mm
    for idx in range(1, n + 1):
        coords = np.argwhere(labeled == idx)
        if coords.size == 0:
            continue
        dz = (coords[:, 0].max() - coords[:, 0].min() + 1) * sz
        dy = (coords[:, 1].max() - coords[:, 1].min() + 1) * sy
        dx = (coords[:, 2].max() - coords[:, 2].min() + 1) * sx
        max_d = max(max_d, max(dz, dy, dx) / 10.0)
    return float(max_d)


def compute_pci_report(
    ct: np.ndarray,
    seg: np.ndarray,
    image: sitk.Image,
    lang: str = "en",
) -> tuple[list[dict], int]:
    """
    seg labels: 0 background, 1–13 → rPCI regions 0–12.
    Returns rows and total PCI sum.
    """
    names = PCI_REGIONS_ZH if lang == "zh" else PCI_REGIONS_EN
    spacing = _spacing_mm(image)
    voxel_ml = (spacing[0] * spacing[1] * spacing[2]) / 1000.0

    rows: list[dict] = []
    total = 0
    for rpci_idx in range(13):
        label = rpci_idx + 1
        mask = seg == label
        voxels = int(mask.sum())
        volume_ml = voxels * voxel_ml
        diam = _max_lesion_diameter_cm(mask, ct, spacing) if voxels else 0.0
        score = pci_score_from_diameter_cm(diam)
        total += score
        rows.append(
            {
                "rpci_index": rpci_idx,
                "label": label,
                "region": names[rpci_idx],
                "volume_ml": round(volume_ml, 2),
                "lesion_diameter_cm": round(diam, 2),
                "pci_score": score,
            }
        )
    return rows, total
