#!/bin/bash
# =============================================================================
# VisualWebBench Evaluation Script
# =============================================================================
# Evaluates the model on VisualWebBench (visual understanding benchmark)
# Usage: ./scripts/evaluate_visualwebbench.sh [OPTIONS]
# =============================================================================

set -e

# === Configuration ===
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default values
MODEL_ENDPOINT="${MODEL_ENDPOINT:-http://localhost:8080/v1}"
MODEL_NAME="${MODEL_NAME:-qwen2vl}"
BENCHMARK_DIR="${BENCHMARK_DIR:-$PROJECT_ROOT/benchmark/visualwebbench}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/results/visualwebbench}"
MAX_CONCURRENT="${MAX_CONCURRENT:-5}"

# === Parse Arguments ===
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-endpoint)
            MODEL_ENDPOINT="$2"
            shift 2
            ;;
        --model-name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --benchmark-dir)
            BENCHMARK_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# === Check Benchmark Directory ===
if [ ! -d "$BENCHMARK_DIR" ]; then
    echo "[ERROR] Benchmark directory not found: $BENCHMARK_DIR"
    echo "Please download VisualWebBench data first."
    exit 1
fi

# === Create Output Directory ===
mkdir -p "$OUTPUT_DIR"

echo "=================================================="
echo "VisualWebBench Evaluation"
echo "=================================================="
echo "Model Endpoint: $MODEL_ENDPOINT"
echo "Model Name: $MODEL_NAME"
echo "Benchmark Dir: $BENCHMARK_DIR"
echo "Output Dir: $OUTPUT_DIR"
echo "=================================================="

# === Run Evaluation Tasks ===
TASKS=("element_ocr" "heading_ocr" "element_ground" "action_ground" "action_prediction" "webqa")

for task in "${TASKS[@]}"; do
    echo ""
    echo ">>> Evaluating VisualWebBench task: $task"
    
    SCRIPT_NAME="visualwebbench_${task}.py"
    if [ -f "$PROJECT_ROOT/src/evaluation/$SCRIPT_NAME" ]; then
        python "$PROJECT_ROOT/src/evaluation/$SCRIPT_NAME" \
            --model-endpoint "$MODEL_ENDPOINT" \
            --model-name "$MODEL_NAME" \
            --benchmark-dir "$BENCHMARK_DIR" \
            --output-dir "$OUTPUT_DIR" \
            2>&1 | tee "$OUTPUT_DIR/${task}_eval.log"
    else
        echo "[SKIP] Script not found: $SCRIPT_NAME"
    fi
done

echo ""
echo "=================================================="
echo "VisualWebBench Evaluation Complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "=================================================="
