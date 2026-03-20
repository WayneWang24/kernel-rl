#!/bin/bash
# ============================================================
# CityU HPC SLURM — SFT 训练（KernelBook Triton → ModelNew）
#
# 硬件：1 节点 × 3× A100 40GB
# 数据：data/split/sft/ (KernelBook ModelNew 格式)
# 模型：Qwen/Qwen2.5-Coder-7B-Instruct → LoRA rank=64
#
# 用法：
#   sbatch scripts/slurm/run_sft_cityu.sh
# ============================================================

#SBATCH --job-name=kernel-rl-sft
#SBATCH --partition=gpu3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=3
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euxo pipefail

eval "$(conda shell.bash hook)"
conda activate kernel-rl

PROJECT_DIR="${HOME}/ChenweiWang/workspace/kernel-rl"
mkdir -p "${PROJECT_DIR}/logs"

export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCHDYNAMO_DISABLE=1
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
    echo "Run: python scripts/data/prepare_split_data.py first"
    exit 1
fi

# ===== 模型路径 =====
MODEL_PATH="${HOME}/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-7B-Instruct/snapshots/c03e6d358207e414f1eca0bb1891e29f1db0e242"
if [ ! -d "$MODEL_PATH" ]; then
    MODEL_PATH="Qwen/Qwen2.5-Coder-7B-Instruct"
fi

CHECKPOINT_DIR="${PROJECT_DIR}/checkpoints/sft_modelnew"

echo "=== SFT Training (CityU HPC) ==="
echo "Model:  $MODEL_PATH"
echo "Train:  $TRAIN_PATH"
echo "Output: $CHECKPOINT_DIR"

# 3 GPU, micro_batch=1 (40GB 内存紧张), gradient checkpointing
torchrun --standalone --nnodes=1 --nproc_per_node=3 \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files="$TRAIN_PATH" \
    data.val_files="$VAL_PATH" \
    data.train_batch_size=24 \
    data.micro_batch_size_per_gpu=1 \
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
    +model.override_config.attn_implementation=sdpa \
    trainer.total_epochs=3 \
    trainer.default_local_dir="$CHECKPOINT_DIR" \
    trainer.save_freq=1 \
    trainer.test_freq=1 \
    trainer.project_name=kernel_sft \
    trainer.experiment_name=qwen25_coder_7b_triton_cityu \
    2>&1 | tee "${PROJECT_DIR}/logs/sft_cityu_${SLURM_JOB_ID}.log"

echo "=== SFT Complete ==="
echo "Next: merge LoRA → run CUDA RL"
