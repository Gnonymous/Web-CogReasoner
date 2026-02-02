#!/bin/bash
# Unified training entrypoint for CogReasoner stages.
# Usage: ./scripts/train.sh [stage1|stage2|stage3] [options]
# Example: ./scripts/train.sh stage2 -m /models/Qwen2.5-VL -d /data/train -o /ckpt/stage2 -c /configs/ds_z2.json


set -e

usage() {
    cat <<USAGE
Usage: ./scripts/train.sh [stage1|stage2|stage3] [options]

Options:
  -m, --model_path        Model path override
  -d, --dataset_dir       Dataset dir override
  -o, --output_dir        Output dir override
  -c, --deepspeed_config  DeepSpeed config override
  -e, --conda_env         Conda env to activate (default: Web-CogReasoner, use "none" to skip)
  -h, --help              Show this help

Example:
  ./scripts/train.sh stage2 -m /models/Qwen2.5-VL -d /data/train -o /ckpt/stage2 -c /configs/ds_z2.json
USAGE
}

die() {
    echo "[ERROR] $1"
    exit 1
}

if [[ $# -lt 1 ]]; then
    usage
    exit 1
fi

STAGE="$1"
shift

case "$STAGE" in
    stage1|stage2|stage3) ;;
    *) die "Unknown stage: $STAGE" ;;
esac

MODEL_PATH_OVERRIDE=""
DATASET_DIR_OVERRIDE=""
OUTPUT_DIR_OVERRIDE=""
DEEPSPEED_CONFIG_OVERRIDE=""
CONDA_ENV="Web-CogReasoner"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--model_path)
            MODEL_PATH_OVERRIDE="$2"
            shift 2
            ;;
        -d|--dataset_dir)
            DATASET_DIR_OVERRIDE="$2"
            shift 2
            ;;
        -o|--output_dir)
            OUTPUT_DIR_OVERRIDE="$2"
            shift 2
            ;;
        -c|--deepspeed_config)
            DEEPSPEED_CONFIG_OVERRIDE="$2"
            shift 2
            ;;
        -e|--conda_env)
            CONDA_ENV="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "Unknown argument: $1"
            ;;
    esac
done

# default setting for train
case "$STAGE" in
    stage1)
        MODEL_PATH="/code/Model/Qwen/Qwen2.5-VL-7B-Instruct"
        DATASET_DIR="/code/Web-CogReasoner/data/Factual"
        DATASETS="Element_Attribute,Sub_Element_Prediction,Page_Change_Prediction_1,Page_Change_Prediction_2,Next_Page_Prediction,Source_element_Prediction,VisualWebChat"
        CUTOFF_LEN="2048"
        LEARNING_RATE="2e-05"
        NUM_TRAIN_EPOCHS="2.0"
        PER_DEVICE_TRAIN_BATCH_SIZE="6"
        GRADIENT_ACCUMULATION_STEPS="8"
        SAVE_STEPS="500"
        SAVE_STRATEGY=""
        WARMUP_RATIO="0.1"
        EVAL_STEPS="500"
        PER_DEVICE_EVAL_BATCH_SIZE="6"
        RUN_NAME="Web-CogReasoner_Factual"
        OUTPUT_DIR="/code/Model/Checkpoints/Web-CogReasoner/Stage1"
        DEEPSPEED_CONFIG="/code/Web-CogReasoner/scripts/ds_z2_config.json"
        ;;
    stage2)
        MODEL_PATH="/code/Model/Checkpoints/Web-CogReasoner/Stage1-best"
        DATASET_DIR="/code/Web-CogReasoner/data/Conceptual"
        DATASETS="Element_Understanding,WebPage_Understanding,Stage_Factual"
        CUTOFF_LEN="4096"
        LEARNING_RATE="1e-05"
        NUM_TRAIN_EPOCHS="2.0"
        PER_DEVICE_TRAIN_BATCH_SIZE="4"
        GRADIENT_ACCUMULATION_STEPS="8"
        SAVE_STEPS="100"
        SAVE_STRATEGY=""
        WARMUP_RATIO="0.05"
        EVAL_STEPS="100"
        PER_DEVICE_EVAL_BATCH_SIZE="4"
        RUN_NAME="Web-CogReasoner_Conceptual"
        OUTPUT_DIR="/code/Model/Checkpoints/Web-CogReasoner/Stage2"
        DEEPSPEED_CONFIG="/code/Web-CogReasoner/scripts/ds_z2_config.json"
        ;;
    stage3)
        MODEL_PATH="/code/Model/Checkpoints/Web-CogReasoner/Stage2-best"
        DATASET_DIR="/code/Web-CogReasoner/data/Procedural"
        DATASETS="User_Intent_Prediction,Multi_Step_Web_Task,Single_Step_Web_Task,Stage_Factual_Conceptual"
        CUTOFF_LEN="8192"
        LEARNING_RATE="1e-05"
        NUM_TRAIN_EPOCHS="1.0"
        PER_DEVICE_TRAIN_BATCH_SIZE="1"
        GRADIENT_ACCUMULATION_STEPS="16"
        SAVE_STEPS=""
        SAVE_STRATEGY="epoch"
        WARMUP_RATIO="0.05"
        EVAL_STEPS="100"
        PER_DEVICE_EVAL_BATCH_SIZE="1"
        RUN_NAME="Web-CogReasoner_Procedural"
        OUTPUT_DIR="/code/Model/Checkpoints/Web-CogReasoner/Stage3"
        DEEPSPEED_CONFIG="/code/Web-CogReasoner/scripts/ds_z2_config.json"
        ;;
esac

if [[ -n "$MODEL_PATH_OVERRIDE" ]]; then MODEL_PATH="$MODEL_PATH_OVERRIDE"; fi
if [[ -n "$DATASET_DIR_OVERRIDE" ]]; then DATASET_DIR="$DATASET_DIR_OVERRIDE"; fi
if [[ -n "$OUTPUT_DIR_OVERRIDE" ]]; then OUTPUT_DIR="$OUTPUT_DIR_OVERRIDE"; fi
if [[ -n "$DEEPSPEED_CONFIG_OVERRIDE" ]]; then DEEPSPEED_CONFIG="$DEEPSPEED_CONFIG_OVERRIDE"; fi

echo "=================================================="
echo "CogReasoner Training"
echo "=================================================="
echo "Stage: $STAGE"
echo "Model Path: $MODEL_PATH"
echo "Dataset Dir: $DATASET_DIR"
echo "Output Dir: $OUTPUT_DIR"
echo "=================================================="

# Conda activation (matches run_stage scripts by default)
if [[ -n "$CONDA_ENV" && "$CONDA_ENV" != "none" ]]; then
    export PATH="/opt/conda/bin:$PATH"
    if [[ -f /opt/conda/etc/profile.d/conda.sh ]]; then
        # shellcheck disable=SC1091
        source /opt/conda/etc/profile.d/conda.sh
        conda activate "$CONDA_ENV"
        echo "Conda env: $CONDA_DEFAULT_ENV"
    else
        echo "[WARNING] Conda profile not found at /opt/conda/etc/profile.d/conda.sh"
    fi
fi

# Wandb login (keep command identical to stage scripts)
if [[ -n "${WANDB_API_KEY:-}" ]]; then
    echo "Logging into Weights & Biases..."
    wandb login "$WANDB_API_KEY"
else
    echo "[WARNING] WANDB_API_KEY not set. wandb login skipped."
fi

mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/train_log_$(date '+%Y%m%d_%H%M%S').txt"

TRAIN_CMD=(
    llamafactory-cli train
    --stage sft
    --do_train True
    --model_name_or_path "$MODEL_PATH"
    --preprocessing_num_workers 16
    --finetuning_type full
    --template qwen2_vl
    --flash_attn fa2
    --dataset_dir "$DATASET_DIR"
    --dataset "$DATASETS"
    --cutoff_len "$CUTOFF_LEN"
    --learning_rate "$LEARNING_RATE"
    --num_train_epochs "$NUM_TRAIN_EPOCHS"
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE"
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
    --lr_scheduler_type cosine
    --max_grad_norm 0.5
    --logging_steps 5
    --warmup_ratio "$WARMUP_RATIO"
    --packing False
    --report_to wandb
    --run_name "$RUN_NAME"
    --output_dir "$OUTPUT_DIR"
    --bf16 True
    --plot_loss True
    --trust_remote_code True
    --ddp_timeout 180000000
    --include_num_input_tokens_seen True
    --optim adamw_torch
    --val_size 0.02
    --eval_strategy steps
    --eval_steps "$EVAL_STEPS"
    --per_device_eval_batch_size "$PER_DEVICE_EVAL_BATCH_SIZE"
    --deepspeed "$DEEPSPEED_CONFIG"
)

if [[ -n "$SAVE_STEPS" ]]; then
    TRAIN_CMD+=(--save_steps "$SAVE_STEPS")
fi
if [[ -n "$SAVE_STRATEGY" ]]; then
    TRAIN_CMD+=(--save_strategy "$SAVE_STRATEGY")
fi

{
    echo "Training Configuration:"
    echo "------------------------"
    CMD_STR="${TRAIN_CMD[*]}"
    echo "$CMD_STR" | sed 's/ \+\(--\)/\n\1/g'
    echo "------------------------"
} | tee "$LOG_FILE"

echo "Starting llamafactory-cli training..." | tee -a "$LOG_FILE"
(
    "${TRAIN_CMD[@]}"
) 2>&1 | tee -a "$LOG_FILE"

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "[SUCCESS] Training completed successfully." | tee -a "$LOG_FILE"
else
    echo "[ERROR] Training failed. Check log: $LOG_FILE" | tee -a "$LOG_FILE"
    exit 2
fi
