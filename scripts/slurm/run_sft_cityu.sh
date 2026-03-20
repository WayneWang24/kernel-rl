#!/bin/bash
# ============================================================
# CityU HPC SLURM — SFT 训练（KernelBook Triton → ModelNew）
#
# 硬件：2 节点 × 3× A100 40GB = 6 GPU
# 数据：data/split/sft/ (KernelBook ModelNew 格式)
# 模型：Qwen/Qwen2.5-Coder-7B-Instruct → LoRA rank=64
#
# 用法：
#   sbatch scripts/slurm/run_sft_cityu.sh
# ============================================================

#SBATCH --job-name=kernel-rl-sft
#SBATCH --partition=gpu3
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=3
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euxo pipefail

# conda activate 期间关闭 -u（cuda-nvcc 激活脚本有未定义变量）
set +u
eval "$(conda shell.bash hook)"
conda activate kernel-rl
set -u

PROJECT_DIR="${HOME}/ChenweiWang/workspace/kernel-rl"
mkdir -p "${PROJECT_DIR}/logs"

export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCHDYNAMO_DISABLE=1
export WANDB_MODE=disabled
unset PYTORCH_CUDA_ALLOC_CONF
unset ROCR_VISIBLE_DEVICES
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# ===== 应用 verl 补丁（PyTorch 2.4 兼容） =====
echo "=== Applying verl patches ==="
python -c "
import sys; sys.path.insert(0, '${PROJECT_DIR}/scripts/train')
from launch_grpo import patch_verl_fsdp_clip_grad
patch_verl_fsdp_clip_grad()
"

# ===== 多节点 torchrun 设置 =====
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# 处理多 IP（IPv6）
if [[ "$head_node_ip" == *" "* ]]; then
    IFS=' ' read -ra ADDR <<< "$head_node_ip"
    if [[ ${#ADDR[0]} -gt 16 ]]; then
        head_node_ip=${ADDR[1]}
    else
        head_node_ip=${ADDR[0]}
    fi
fi

MASTER_PORT=29500
echo "Master node: $head_node ($head_node_ip:$MASTER_PORT)"
echo "Total nodes: $SLURM_JOB_NUM_NODES, GPUs/node: 3, Total GPUs: $((SLURM_JOB_NUM_NODES * 3))"

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

echo "=== SFT Training (CityU HPC, 2 nodes × 3 GPUs) ==="
echo "Model:  $MODEL_PATH"
echo "Train:  $TRAIN_PATH"
echo "Output: $CHECKPOINT_DIR"

# 6 GPU, micro_batch=2, gradient checkpointing
# batch_size=24 / (2 * 6) = 2 gradient accumulation steps
srun torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=3 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$head_node_ip:$MASTER_PORT \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files="$TRAIN_PATH" \
    data.val_files="$VAL_PATH" \
    data.train_batch_size=24 \
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
