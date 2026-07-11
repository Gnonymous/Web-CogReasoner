#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

python "$PROJECT_ROOT/src/evaluation/online/auto_eval_mind2web_cross_task.py" \
    --provider gemini \
    --api_model gemini-2.5-pro \
    --process_dir "$PROJECT_ROOT/results_mind2web_task" \
    --output_report_path "$PROJECT_ROOT/results_mind2web_task/eval_summary_mind2web.json" \
    --mind2web_json "$PROJECT_ROOT/benchmark/mind2web_test_cross_task.jsonl" \
    --max_attached_imgs 15
