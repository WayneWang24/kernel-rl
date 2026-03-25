#!/bin/bash
# ============================================================
# PolyU VM — SFT 训练 (3B)
#
# 硬件：2× A100 80GB
# 模型：Qwen/Qwen2.5-Coder-3B-Instruct → LoRA rank=64
# 数据：data/split/sft/ 或 data/sft_modelnew/ (KernelBook ModelNew 格式)
#
# 用法：
#   bash scripts/train/run_sft_3b_polyu.sh
#
# SFT 完成后：
#   1. 合并 LoRA: python scripts/eval/merge_lora.py \
#          --ckpt_dir checkpoints/sft_3b --output_dir checkpoints/sft_3b_merged
#   2. 启动 RL: bash scripts/train/run_grpo_polyu.sh
# ============================================================

set -euxo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
mkdir -p "${PROJECT_DIR}/logs"

export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export TORCHDYNAMO_DISABLE=1
export WANDB_MODE=disabled
unset PYTORCH_CUDA_ALLOC_CONF
unset ROCR_VISIBLE_DEVICES
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# ===== 数据路径 =====
if [ -f "${PROJECT_DIR}/data/split/sft/train.parquet" ]; then
    TRAIN_PATH="${PROJECT_DIR}/data/split/sft/train.parquet"
    VAL_PATH="${PROJECT_DIR}/data/split/sft/val.parquet"
    echo "Using split SFT data"
elif [ -f "${PROJECT_DIR}/data/sft_modelnew/train.parquet" ]; then
    TRAIN_PATH="${PROJECT_DIR}/data/sft_modelnew/train.parquet"
    VAL_PATH="${PROJECT_DIR}/data/sft_modelnew/val.parquet"
    echo "Using ModelNew SFT data"
else
    echo "ERROR: No SFT data found!"
    echo "Need data/split/sft/ or data/sft_modelnew/"
    exit 1
fi

# ===== 模型 =====
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-Coder-3B-Instruct}"
CHECKPOINT_DIR="${PROJECT_DIR}/checkpoints/sft_3b"

echo ""
echo "============================================"
echo "  SFT 3B Training (PolyU VM)"
echo "============================================"
echo "Model:  $MODEL_PATH"
echo "Train:  $TRAIN_PATH"
echo "Output: $CHECKPOINT_DIR"
echo "============================================"
echo ""

# 2 GPU, 3B 模型 micro_batch 可以开到 4
torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    -m verl.trainer.sft_trainer \
    data.train_files="$TRAIN_PATH" \
    data.val_files="$VAL_PATH" \
    data.train_batch_size=32 \
    data.micro_batch_size_per_gpu=4 \
    data.max_length=4096 \
    data.truncation=left \
    data.messages_key=messages \
    model.path="$MODEL_PATH" \
    model.enable_gradient_checkpointing=true \
    model.lora_rank=64 \
    model.lora_alpha=128 \
    model.target_modules=all-linear \
    model.use_liger=false \
    trainer.total_epochs=3 \
    trainer.default_local_dir="$CHECKPOINT_DIR" \
    trainer.save_freq=200 \
    trainer.test_freq=200 \
    trainer.project_name=kernel_sft \
    trainer.experiment_name=qwen25_coder_3b_sft_polyu \
    2>&1 | tee "${PROJECT_DIR}/logs/sft_3b_polyu.log"

echo "=== SFT 3B Complete ==="
echo "Next: merge LoRA and start RL"
echo "  1. Merge: python scripts/eval/merge_lora.py --ckpt_dir $CHECKPOINT_DIR --output_dir checkpoints/sft_3b_merged"
echo "  2. RL:    bash scripts/train/run_grpo_polyu.sh"
