#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate
export NO_PROXY='*' no_proxy='*' HTTP_PROXY='' HTTPS_PROXY='' ALL_PROXY=''
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export nnUNet_raw="$ROOT/nnunet_data/nnUNet_raw"
export nnUNet_preprocessed="$ROOT/nnunet_data/nnUNet_preprocessed"
export nnUNet_results="$ROOT/nnunet_data/nnUNet_results"
mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"
pip install -q 'streamlit>=1.28.0' 'nnunetv2>=2.2'
exec streamlit run "$ROOT/visualization/streamlit_app.py" --server.headless true
