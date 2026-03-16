#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

source .venv/bin/activate

python main.py all
python main.py best-model
python main.py structural

echo "All artifacts were generated in output_artefacts/."
