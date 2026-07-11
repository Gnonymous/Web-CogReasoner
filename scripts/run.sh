#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
mkdir -p "$PROJECT_ROOT/results_webvoyager"

python -u "$PROJECT_ROOT/src/online_exploration/run.py" \
    --test_file "$PROJECT_ROOT/benchmark/WebVoyager.jsonl" \
    --headless \
    --max_iter 15 \
    --max_attached_imgs 3 \
    --save_accessibility_tree \
    --output_dir "$PROJECT_ROOT/results_webvoyager" \
    --api_localhost "${MODEL_ENDPOINT:-http://localhost:8080/v1}" > "$PROJECT_ROOT/results_webvoyager.log"
