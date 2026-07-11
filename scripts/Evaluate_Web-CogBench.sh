#!/usr/bin/env bash
#
# Web-CogBench Offline Evaluation Entry Point
#
# This script runs all offline evaluation tasks for the Web-CogBench benchmark.
# It supports running inference, evaluation (scoring), or both across multiple
# prediction and understanding sub-tasks.
#
# Usage:
#   ./scripts/Evaluate_Web-CogBench.sh [--mode inference|evaluation|all] [--gemini-api-key KEY]
#
# Options:
#   --mode            Execution mode: 'inference', 'evaluation', or 'all' (default: all).
#   --gemini-api-key  API key for Gemini evaluator (required for 'evaluation' and 'all' modes).
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
mkdir -p "$PROJECT_ROOT/results_Web-CogBench"

MODE="all"
GEMINI_API_KEY="${GEMINI_API_KEY:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --gemini-api-key)
      GEMINI_API_KEY="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [--mode inference|evaluation|all] [--gemini-api-key KEY]"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

if [[ "$MODE" != "inference" && "$MODE" != "evaluation" && "$MODE" != "all" ]]; then
  echo "[ERROR] --mode must be one of: inference, evaluation, all"
  exit 1
fi

if [[ ("$MODE" == "evaluation" || "$MODE" == "all") && -z "$GEMINI_API_KEY" ]]; then
  echo "[ERROR] GEMINI_API_KEY is required for mode '$MODE'"
  exit 1
fi

run_py() {
  python "$PROJECT_ROOT/$1"
}

run_py_mode() {
  GEMINI_API_KEY="$GEMINI_API_KEY" python "$PROJECT_ROOT/$1" --mode "$MODE"
}

echo "=== Web-CogBench Offline (mode=$MODE) ==="

run_py "src/evaluation/offline/Element_Attribute.py"
run_py_mode "src/evaluation/offline/Element_Understanding.py"
run_py "src/evaluation/offline/Next_Page_Prediction.py"
run_py "src/evaluation/offline/Popup_Close.py"
run_py "src/evaluation/offline/Single_Step_Exploration.py"
run_py "src/evaluation/offline/Source_Element_Prediction.py"
run_py_mode "src/evaluation/offline/User_Intent_Prediction.py"
run_py_mode "src/evaluation/offline/WebPage_Understanding.py"

echo "All Web-CogBench offline tasks finished."
