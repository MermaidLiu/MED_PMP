"""Fix nnU-Net v2 folder layout under nnUNet_results."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

TRAINER_DIR = "nnUNetTrainer__nnUNetPlans__3d_lowres"


def read_dataset_name(dataset_root: Path) -> str | None:
    meta = dataset_root / "dataset.json"
    if not meta.is_file():
        meta = dataset_root / TRAINER_DIR / "dataset.json"
    if not meta.is_file():
        return None
    try:
        data = json.loads(meta.read_text(encoding="utf-8"))
        return data.get("name")
    except (json.JSONDecodeError, OSError):
        return None


def normalize_dataset_root(dataset_root: Path) -> tuple[bool, str]:
    """
    Ensure layout: DatasetXXX/<trainer>/fold_* , plans.json, dataset.json
    """
    if not dataset_root.is_dir():
        return False, f"Not a directory: {dataset_root}"

    trainer_path = dataset_root / TRAINER_DIR
    if trainer_path.is_dir() and any(trainer_path.glob("fold_*")):
        return True, "Already normalized."

    has_top_level_folds = any(dataset_root.glob("fold_*"))
    if not has_top_level_folds:
        return False, f"No fold_* folders in {dataset_root}"

    trainer_path.mkdir(parents=True, exist_ok=True)
    for name in ("fold_0", "fold_1", "fold_2", "fold_3", "fold_4"):
        src = dataset_root / name
        if src.is_dir():
            dst = trainer_path / name
            if dst.exists():
                shutil.rmtree(dst)
            shutil.move(str(src), str(dst))

    for fname in ("dataset.json", "plans.json", "dataset_fingerprint.json", "crossval_results_folds_0_1_2_3_4"):
        src = dataset_root / fname
        if src.exists():
            dst = trainer_path / fname
            if not dst.exists():
                shutil.move(str(src), str(dst))

    return True, f"Normalized → {TRAINER_DIR}/"


def discover_dataset_roots(results_dir: Path) -> list[Path]:
    if not results_dir.is_dir():
        return []
    roots: list[Path] = []
    for p in sorted(results_dir.iterdir()):
        if not p.is_dir() or p.name.startswith(".") or p.name == "__MACOSX":
            continue
        if not p.name.startswith("Dataset"):
            continue
        roots.append(p)
    return roots


def find_installed_datasets(results_dir: Path, *, auto_normalize: bool = True) -> list[str]:
    names: list[str] = []
    for root in discover_dataset_roots(results_dir):
        if auto_normalize:
            normalize_dataset_root(root)
        trainer = root / TRAINER_DIR
        if trainer.is_dir() and any(trainer.glob("fold_*")):
            names.append(root.name)
        elif any(root.glob("fold_*")):
            names.append(root.name)
    return sorted(set(names))


def dataset_id_for_predict(folder_name: str, dataset_root: Path) -> str:
    """Prefer numeric id from folder name; fall back to dataset.json name."""
    import re

    m = re.match(r"Dataset(\d+)_", folder_name)
    if m:
        return str(int(m.group(1)))
    meta_name = read_dataset_name(dataset_root)
    if meta_name:
        m2 = re.match(r"Dataset(\d+)_", meta_name)
        if m2:
            return str(int(m2.group(1)))
    return "101"


def install_from_local_folder(source: Path, results_dir: Path) -> tuple[bool, str]:
    if not source.is_dir():
        return False, f"Folder not found: {source}"
    if not (source / "dataset.json").is_file() and not any(source.glob("fold_*")):
        return False, "Not a valid nnU-Net model folder (missing dataset.json / fold_*)."

    folder_name = source.name if source.name.startswith("Dataset") else None
    if folder_name is None:
        meta_name = read_dataset_name(source)
        folder_name = meta_name or "Dataset101_PM"

    dest = results_dir / folder_name
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(source, dest, ignore=shutil.ignore_patterns(".DS_Store", "__MACOSX"))
    ok, msg = normalize_dataset_root(dest)
    if not ok:
        return False, msg
    return True, f"Installed to {dest} ({msg})"
