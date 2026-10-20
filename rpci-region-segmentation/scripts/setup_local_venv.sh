#!/usr/bin/env bash
# Local setup when system HTTP proxy (e.g. 127.0.0.1:7897) is configured but not running.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export NO_PROXY='*'
export no_proxy='*'
export HTTP_PROXY=''
export HTTPS_PROXY=''
export ALL_PROXY=''

python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

pip install --upgrade pip
pip install torch torchvision
pip install -r requirements.txt -c constraints-macos-py312.txt
pip install -e .

echo ""
echo "Done. Activate with: source $ROOT/.venv/bin/activate"
echo "Verify: rpci-preprocess --help"
