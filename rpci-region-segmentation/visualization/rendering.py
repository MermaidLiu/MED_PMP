"""Matplotlib rendering with optional region contours."""

from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import find_contours

from visualization.plot_results import REGION_COLORS, REGION_LABELS


def _window_ct(slice2d: np.ndarray, low: float = -150, high: float = 250) -> np.ndarray:
    return np.clip(slice2d, low, high)


def render_slice_views(
    ct_slice: np.ndarray,
    seg_slice: np.ndarray,
    *,
    alpha: float = 0.35,
    show_contours: bool = True,
    titles: tuple[str, str, str] | None = None,
) -> plt.Figure:
    titles = titles or ("CT", "Overlay", "Contours")
    cmap = mcolors.ListedColormap(REGION_COLORS[: len(REGION_LABELS)])
    norm = mcolors.BoundaryNorm(np.arange(len(REGION_LABELS) + 1) - 0.5, len(REGION_LABELS))
    display = _window_ct(ct_slice)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), facecolor="#F0F8FF")
    for ax in axes:
        ax.set_facecolor("#FAFCFF")

    axes[0].imshow(display, cmap="gray")
    axes[0].set_title(titles[0], color="#0078D4", fontsize=11, fontweight="600")
    axes[0].axis("off")

    axes[1].imshow(display, cmap="gray")
    axes[1].imshow(seg_slice, alpha=alpha, cmap=cmap, norm=norm)
    axes[1].set_title(titles[1], color="#0078D4", fontsize=11, fontweight="600")
    axes[1].axis("off")

    axes[2].imshow(display, cmap="gray")
    if show_contours:
        present = np.unique(seg_slice.astype(int))
        for label in present:
            if label <= 0 or label >= len(REGION_LABELS):
                continue
            mask = seg_slice == label
            color = REGION_COLORS[label]
            contours = find_contours(mask.astype(float), 0.5)
            for contour in contours:
                axes[2].plot(contour[:, 1], contour[:, 0], color=color, linewidth=1.2)
    axes[2].set_title(titles[2], color="#0078D4", fontsize=11, fontweight="600")
    axes[2].axis("off")

    present = np.unique(seg_slice.astype(int))
    patches = [
        mpatches.Patch(color=REGION_COLORS[i], label=REGION_LABELS[i])
        for i in present
        if 0 < i < len(REGION_LABELS)
    ]
    if patches:
        fig.legend(
            handles=patches,
            loc="lower center",
            ncol=min(7, len(patches)),
            fontsize=7,
            framealpha=0.85,
            facecolor="white",
        )
    plt.tight_layout()
    return fig
