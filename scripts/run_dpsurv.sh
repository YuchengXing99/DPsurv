#!/bin/bash
# Run DPsurv nested cross-validation on a TCGA cancer dataset.
#
# Usage:
#   bash scripts/run_dpsurv.sh [CANCER]
#
# Arguments:
#   CANCER  — short dataset name: BLCA, BRCA, KIRC, LUAD, UCEC  (default: KIRC)
#
# Examples:
#   bash scripts/run_dpsurv.sh            # KIRC
#   bash scripts/run_dpsurv.sh LUAD      # LUAD
#   bash scripts/run_dpsurv.sh BRCA      # BRCA

DATASET=${1:-KIRC}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

python "${REPO_ROOT}/trainer/train_dpsurv.py" \
    --datasets "${DATASET}" \
    --device cuda
