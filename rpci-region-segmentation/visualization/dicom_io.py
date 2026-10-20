"""DICOM ZIP → NIfTI conversion via SimpleITK."""

from __future__ import annotations

import io
import shutil
import tempfile
import zipfile
from pathlib import Path

import SimpleITK as sitk


def _collect_dicom_roots(extract_dir: Path) -> list[str]:
    reader = sitk.ImageSeriesReader()
    try:
        series_ids = reader.GetGDCMSeriesIDs(str(extract_dir))
    except RuntimeError:
        series_ids = []
    if series_ids:
        return list(series_ids)

    # Search nested folders
    all_ids: list[str] = []
    for sub in extract_dir.rglob("*"):
        if not sub.is_dir():
            continue
        try:
            ids = reader.GetGDCMSeriesIDs(str(sub))
            all_ids.extend(ids)
        except RuntimeError:
            continue
    return list(dict.fromkeys(all_ids))


def _series_file_count(extract_dir: Path, series_id: str) -> int:
    reader = sitk.ImageSeriesReader()
    for sub in [extract_dir, *extract_dir.rglob("*")]:
        if not sub.is_dir():
            continue
        try:
            files = reader.GetGDCMSeriesFileNames(str(sub), series_id)
        except RuntimeError:
            continue
        if files:
            return len(files)
    return 0


def _read_series(extract_dir: Path, series_id: str) -> sitk.Image:
    reader = sitk.ImageSeriesReader()
    for sub in [extract_dir, *extract_dir.rglob("*")]:
        if not sub.is_dir():
            continue
        try:
            names = reader.GetGDCMSeriesFileNames(str(sub), series_id)
        except RuntimeError:
            continue
        if names:
            reader.SetFileNames(names)
            return reader.Execute()
    raise RuntimeError(f"Could not read DICOM series {series_id}")


def dicom_zip_to_nifti(
    zip_bytes: bytes,
    output_path: Path,
    case_id: str = "case001",
) -> tuple[Path, sitk.Image]:
    """Extract ZIP, pick largest series, write nnU-Net-style NIfTI."""
    work = Path(tempfile.mkdtemp(prefix="rpci_dicom_"))
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            zf.extractall(work)

        series_ids = _collect_dicom_roots(work)
        if not series_ids:
            raise ValueError("No DICOM series found in ZIP.")

        best = max(series_ids, key=lambda sid: _series_file_count(work, sid))
        image = _read_series(work, best)
        image = sitk.DICOMOrient(image, "LPS")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        nnunet_name = output_path.parent / f"{case_id}_0000.nii.gz"
        sitk.WriteImage(image, str(nnunet_name), useCompression=True)
        return nnunet_name, image
    finally:
        shutil.rmtree(work, ignore_errors=True)
