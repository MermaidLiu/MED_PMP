# nnU-Net Integration

This document describes how to use [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet) with the rPCI data format. The framework is installed via `pip install nnunetv2`; this repository does not vendor nnU-Net code (avoid naming a local folder `nnunetv2`, which would shadow the pip package).

## Setup

### 1. Install nnU-Net

```bash
pip install nnunetv2
```

### 2. Set Environment Variables

```bash
export nnUNet_raw="/path/to/nnUNet_raw"
export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
export nnUNet_results="/path/to/nnUNet_results"
```

### 3. Prepare Your Dataset

```bash
python preprocessing/convert_to_nnunet.py \
    --images_dir /path/to/processed_data \
    --segmentations_dir /path/to/processed_data \
    --output_dir "$nnUNet_raw" \
    --dataset_name Dataset001_rPCI
```

## Usage

See the main [README.md](../README.md) for training and inference commands.

The conversion utility lives at `preprocessing/convert_to_nnunet.py`.
