# Med_PMP

Web workbench for **peritoneal metastasis (PM) imaging**: DICOM upload, nnU-Net rPCI segmentation, contour visualization, region volumes, and imaging-based PCI estimates.

Built on [rpci-region-segmentation](https://github.com/PieterGort/rpci-region-segmentation) (MIT).

## Quick start

```bash
cd rpci-region-segmentation
bash scripts/setup_local_venv.sh   # first time
./scripts/run_viewer.sh
```

Open the Streamlit UI (default `http://localhost:8501`). Install the released **Dataset101_PM** nnU-Net weights under **Model setup** (weights are not stored in git).

## Layout

- `rpci-region-segmentation/` — preprocessing, nnU-Net helpers, **Med_PMP** Streamlit app (`visualization/streamlit_app.py`)

## License

Upstream rPCI code is MIT; see `rpci-region-segmentation/LICENSE`.
