#!/bin/bash
# =============================================================================
# Web-CogBench Evaluation Script
# =============================================================================
# Evaluates the model on Web-CogBench (cognitive reasoning benchmark)
# Usage: ./scripts/evaluate_web_cogbench.sh [OPTIONS]
# =============================================================================

set -e

# === Configuration ===
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default values
MODEL_ENDPOINT="${MODEL_ENDPOINT:-http://localhost:8080/v1}"
MODEL_NAME="${MODEL_NAME:-qwen2vl}"
TEST_FILE="${TEST_FILE:-$PROJECT_ROOT/data/test/web_cogbench.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/results/web_cogbench}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"  # 0 means all samples
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
        --test-file)
            TEST_FILE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --max-samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# === Check Test File ===
if [ ! -f "$TEST_FILE" ]; then
    echo "[ERROR] Test file not found: $TEST_FILE"
    echo "Please ensure the Web-CogBench data is available."
    exit 1
fi

# === Create Output Directory ===
mkdir -p "$OUTPUT_DIR"

echo "=================================================="
echo "Web-CogBench Evaluation"
echo "=================================================="
echo "Model Endpoint: $MODEL_ENDPOINT"
echo "Model Name: $MODEL_NAME"
echo "Test File: $TEST_FILE"
echo "Output Dir: $OUTPUT_DIR"
echo "Max Samples: ${MAX_SAMPLES:-all}"
echo "=================================================="

# === Run Evaluation Tasks ===
TASKS=("action_prediction" "element_attribute" "next_page_prediction" 
       "user_intent_prediction" "element_understanding" "webpage_understanding")

for task in "${TASKS[@]}"; do
    echo ""
    echo ">>> Evaluating task: $task"
    
    python "$PROJECT_ROOT/src/evaluation/${task}.py" \
        --model-endpoint "$MODEL_ENDPOINT" \
        --model-name "$MODEL_NAME" \
        --test-file "$TEST_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --max-samples "$MAX_SAMPLES" \
        --max-concurrent "$MAX_CONCURRENT" \
        2>&1 | tee "$OUTPUT_DIR/${task}_eval.log"
done

echo ""
echo "=================================================="
echo "Evaluation Complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "=================================================="
