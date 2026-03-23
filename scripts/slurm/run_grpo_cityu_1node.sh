#!/bin/bash
# ============================================================
# CityU HPC SLURM 作业脚本 — GRPO 训练（单节点版）
#
# 硬件：1 节点 × 3× A100 40GB = 3 GPU
# 分区：gpu3
#
# 用法：
#   sbatch scripts/slurm/run_grpo_cityu_1node.sh
#
# 说明：
#   先单节点跑通，确认训练逻辑无误后再扩展到多节点。
#   3× A100 40GB 跑 7B 需开 optimizer_offload。
# ============================================================

#SBATCH --job-name=kernel-rl-grpo-1n
#SBATCH --partition=gpu3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=3
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euxo pipefail

# ===== 环境 =====
# conda activate 期间关闭 -u（cuda-nvcc 激活脚本有未定义变量）
set +u
eval "$(conda shell.bash hook)"
conda activate kernel-rl
set -u

PROJECT_DIR="${HOME}/ChenweiWang/workspace/kernel-rl"
mkdir -p "${PROJECT_DIR}/logs"

# ===== 环境变量 =====
# SGLang 使用 flashinfer，无需 VLLM_ATTENTION_BACKEND
export RAY_memory_monitor_refresh_ms=0
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCHDYNAMO_DISABLE=1

# 禁用 expandable_segments（与 SGLang/vLLM memory pool 不兼容）
unset PYTORCH_CUDA_ALLOC_CONF
# 清除 ROCm 变量
unset ROCR_VISIBLE_DEVICES

# GPU 管理由 Ray + verl 的 patch_verl_force_cuda 补丁处理。
# 不设 CUDA_VISIBLE_DEVICES（让 SLURM cgroup 自然管理），不设
# RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES（让 Ray 正常管理每个 worker 的 GPU）。
# 补丁让 TaskRunner（无 GPU 的 Ray actor）通过 nvidia-smi 检测 CUDA，
# 避免级联失败。
export WANDB_MODE=disabled

# NCCL 修复：PCIe A100 之间 P2P 可能不可用，禁用后走 SHM
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN

# ===== Step 0: 清理残留 Ray 集群 =====
ray stop --force 2>/dev/null || true
unset RAY_ADDRESS

# ===== Step 0.5: GPU 健康检查（逐个探测，跳过坏 GPU）=====
echo "=== GPU Health Check ==="
nvidia-smi || echo "WARNING: nvidia-smi not found or failed"

# 用 subprocess 逐个测试每个 GPU，因为 torch.cuda 初始化后不能重新加载
WORKING_GPUS=$(python3 -c "
import subprocess, sys, os
total = int(os.environ.get('SLURM_GPUS_PER_NODE', '0') or '0')
if total == 0:
    import torch
    total = torch.cuda.device_count()
working = []
for i in range(total):
    r = subprocess.run(
        [sys.executable, '-c',
         f'import os; os.environ[\"CUDA_VISIBLE_DEVICES\"]=\"{i}\"; import torch; '
         f'assert torch.cuda.is_available() and torch.cuda.device_count()>0, '
         f'f\"GPU {i}: avail={torch.cuda.is_available()} count={torch.cuda.device_count()}\"'],
        capture_output=True, timeout=30, text=True)
    if r.returncode == 0:
        working.append(str(i))
        print(f'GPU {i}: OK')
    else:
        print(f'GPU {i}: FAILED - {r.stderr.strip().splitlines()[-1] if r.stderr.strip() else \"unknown\"}')
print(f'WORKING_GPUS={len(working)}')
print(','.join(working))
" 2>&1)
echo "$WORKING_GPUS"

# 提取最后一行（逗号分隔的 GPU ID 列表）
GPU_IDS=$(echo "$WORKING_GPUS" | tail -1)
NUM_WORKING=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)

if [ "$NUM_WORKING" -lt 2 ]; then
    echo "ERROR: Need at least 2 working GPUs, found $NUM_WORKING"
    exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
echo "Using $NUM_WORKING working GPUs: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# ===== Step 1: 应用 verl 补丁 =====
echo "=== Applying verl patches ==="
python -c "
import sys; sys.path.insert(0, '${PROJECT_DIR}/scripts/train')
from launch_grpo import apply_all_patches
apply_all_patches('${PROJECT_DIR}')
"

# ===== Step 2: 数据路径探测 =====
# 优先 CUDA 数据 → KernelBook split → KernelBench Triton → KernelBook 原始
if [ -f "${PROJECT_DIR}/data/rl_kernelbench_cuda/train.parquet" ]; then
    TRAIN_PATH="${PROJECT_DIR}/data/rl_kernelbench_cuda/train.parquet"
    VAL_PATH="${PROJECT_DIR}/data/rl_kernelbench_cuda/val.parquet"
    REWARD_FN_NAME="compute_score_auto"
    echo "Using KernelBench CUDA RL data (compile+run reward)"
elif [ -f "${PROJECT_DIR}/data/split/rl/train.parquet" ]; then
    TRAIN_PATH="${PROJECT_DIR}/data/split/rl/train.parquet"
    VAL_PATH="${PROJECT_DIR}/data/split/rl/val.parquet"
    REWARD_FN_NAME="compute_score_auto"
    echo "Using KernelBook split RL data (ModelNew format)"
elif [ -f "${PROJECT_DIR}/data/rl_kernelbench/train.parquet" ]; then
    TRAIN_PATH="${PROJECT_DIR}/data/rl_kernelbench/train.parquet"
    VAL_PATH="${PROJECT_DIR}/data/rl_kernelbench/val.parquet"
    REWARD_FN_NAME="compute_score_auto"
    echo "Using KernelBench RL data (Triton format)"
elif [ -f "${PROJECT_DIR}/data/rl/train.parquet" ]; then
    TRAIN_PATH="${PROJECT_DIR}/data/rl/train.parquet"
    VAL_PATH="${PROJECT_DIR}/data/rl/val.parquet"
    REWARD_FN_NAME="compute_score"
    echo "Using KernelBook RL data (original format)"
else
    echo "ERROR: No RL training data found!"
    exit 1
fi

# 模型路径
SFT_CHECKPOINT="${PROJECT_DIR}/checkpoints/sft_modelnew_merged"
if [ ! -d "$SFT_CHECKPOINT" ]; then
    SFT_CHECKPOINT="${PROJECT_DIR}/checkpoints/sft_merged"
fi
if [ -d "$SFT_CHECKPOINT" ]; then
    MODEL_PATH="$SFT_CHECKPOINT"
    echo "Using SFT checkpoint: $MODEL_PATH"
else
    MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-Coder-7B-Instruct}"
    echo "Using base model: $MODEL_PATH"
fi

REWARD_FN_PATH="${PROJECT_DIR}/src/reward/kernel_reward.py"

# ===== Step 3: 动态计算 batch size =====
# train_batch_size 必须能被 n_gpus 整除
# ppo_mini_batch_size 必须能被 n_gpus 整除
TRAIN_BS=$((NUM_WORKING * 4))    # 每 GPU 4 samples
MINI_BS=$TRAIN_BS                # mini_batch = train_batch（GRPO 不需要多轮）

# ===== Step 4: 启动 GRPO 训练 =====
echo ""
echo "=== Launching GRPO Training (1 node, ${NUM_WORKING} GPUs) ==="
echo "Model:     $MODEL_PATH"
echo "Train:     $TRAIN_PATH"
echo "Reward:    $REWARD_FN_NAME"
echo "GPUs:      $NUM_WORKING (CVD=$CUDA_VISIBLE_DEVICES)"
echo "BatchSize: train=$TRAIN_BS mini=$MINI_BS"
echo ""

# 单节点不需要手动启动 Ray 集群，verl 会自动 ray.init()
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_PATH" \
    data.val_files="$VAL_PATH" \
    data.train_batch_size=$TRAIN_BS \
    data.max_prompt_length=2048 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=true \
    data.truncation=error \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=false \
    actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BS \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    +actor_rollout_ref.actor.optim.override_optimizer_config.foreach=false \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.free_cache_engine=true \
    actor_rollout_ref.rollout.enforce_eager=false \
    actor_rollout_ref.rollout.max_num_seqs=8 \
    "+actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=flashinfer" \
    "+actor_rollout_ref.rollout.engine_kwargs.sglang.cuda_graph_max_bs=8" \
    actor_rollout_ref.rollout.n=3 \
    algorithm.use_kl_in_reward=false \
    custom_reward_function.path="$REWARD_FN_PATH" \
    custom_reward_function.name="$REWARD_FN_NAME" \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name=kernel_rl \
    trainer.experiment_name=grpo_cuda_sglang_1node \
    trainer.default_local_dir="${PROJECT_DIR}/checkpoints/grpo_cuda" \
    trainer.n_gpus_per_node=$NUM_WORKING \
    trainer.nnodes=1 \
    trainer.save_freq=200 \
    trainer.test_freq=200 \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.total_epochs=3 \
    2>&1 | tee "${PROJECT_DIR}/logs/grpo_cityu_1node_${SLURM_JOB_ID}.log"

echo "=== GRPO Training Complete ==="
ray stop --force 2>/dev/null || true
