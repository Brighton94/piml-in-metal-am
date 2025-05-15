#!/usr/bin/env bash
set -euo pipefail

# Install (no-op if already satisfied)
uv pip install -e .[dev]

uv venv exec python -m src.training.train_full \
  --keys tcr_phase1_build1 tcr_phase1_build2 \
  --epochs 40 --batch 8 --weight_decay 5e-3
