#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

python "$PROJECT_ROOT/src/evaluation/online/auto_eval_webvoyager.py" \
    --provider gemini \
    --api_model gemini-2.5-pro \
    --process_dir "$PROJECT_ROOT/results_webvoyager" \
    --output_report_path "$PROJECT_ROOT/results_webvoyager/evaluation_summary.json" \
    --max_attached_imgs 15
