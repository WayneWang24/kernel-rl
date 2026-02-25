#!/bin/bash
# ============================================================
# SFT 训练脚本
#
# 使用 verl 内置 SFT trainer 在 KernelBook 上微调
# 硬件：2× A100 80GB
# 模型：Qwen/Qwen2.5-Coder-7B-Instruct
#
# 用法：
#   bash scripts/train/run_sft.sh [额外 override 参数]
#
# 前提：
#   - 已执行 prepare_sft_data.py 生成 data/sft/*.parquet
#   - verl 已安装
# ============================================================

set -euxo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# 数据路径
TRAIN_PATH="${PROJECT_DIR}/data/sft/train.parquet"
VAL_PATH="${PROJECT_DIR}/data/sft/val.parquet"

# 模型
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-Coder-7B-Instruct}"

# 输出
CHECKPOINT_DIR="${PROJECT_DIR}/checkpoints/sft"

# 检查数据
if [ ! -f "$TRAIN_PATH" ]; then
    echo "ERROR: Training data not found at $TRAIN_PATH"
    echo "Please run: python scripts/data/prepare_sft_data.py first"
    exit 1
fi

torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files="$TRAIN_PATH" \
    data.val_files="$VAL_PATH" \
    data.train_batch_size=32 \
    data.micro_batch_size_per_gpu=2 \
    data.max_length=4096 \
    data.truncation=left \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    model.partial_pretrain="$MODEL_PATH" \
    model.enable_gradient_checkpointing=true \
    model.lora_rank=64 \
    model.lora_alpha=128 \
    model.target_modules=all-linear \
    model.use_liger=false \
    trainer.total_epochs=3 \
    trainer.default_local_dir="$CHECKPOINT_DIR" \
    trainer.save_freq=1 \
    trainer.test_freq=1 \
    trainer.project_name=kernel_sft \
    trainer.experiment_name=qwen25_coder_7b_triton \
    "$@"

echo "=== SFT Training Complete ==="
echo "Checkpoints saved to: $CHECKPOINT_DIR"
echo ""
echo "Next steps:"
echo "  1. Merge LoRA weights (if needed)"
echo "  2. Run RL training: bash scripts/train/run_grpo.sh"
