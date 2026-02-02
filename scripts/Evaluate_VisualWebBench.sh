#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== VisualWebBench Offline Evaluation ==="

python "$PROJECT_ROOT/src/evaluation/offline/VisualWebBench_Action_Ground.py"
python "$PROJECT_ROOT/src/evaluation/offline/VisualWebBench_Action_Prediction.py"
python "$PROJECT_ROOT/src/evaluation/offline/VisualWebBench_Element_Ground.py"
python "$PROJECT_ROOT/src/evaluation/offline/VisualWebBench_Element_Ocr.py"
python "$PROJECT_ROOT/src/evaluation/offline/VisualWebBench_Heading_OCR.py"
python "$PROJECT_ROOT/src/evaluation/offline/VisualWebBench_Webqa.py"

echo "All VisualWebBench offline tasks finished."
