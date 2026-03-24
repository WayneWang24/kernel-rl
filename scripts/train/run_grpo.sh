#!/bin/bash
# ============================================================
# GRPO RL 训练脚本（PolyU HPC）
#
# 硬件：2× A100 80GB，单节点
# 模型：Qwen2.5-Coder-3B-Instruct（快速验证方法有效性）
#
# 用法：
#   bash scripts/train/run_grpo.sh [额外 override 参数]
#
# 前提：
#   - RL 数据已准备（data/split/rl/ 或 data/rl/）
#   - verl 已安装
# ============================================================

set -euxo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ===== 环境变量 =====
export RAY_memory_monitor_refresh_ms=0
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export TORCHDYNAMO_DISABLE=1
export WANDB_MODE=disabled
unset PYTORCH_CUDA_ALLOC_CONF

# ===== 清理残留 Ray =====
ray stop --force 2>/dev/null || true
unset RAY_ADDRESS

# ===== 应用 verl 补丁 =====
echo "=== Applying verl patches ==="
python3 -c "
import sys; sys.path.insert(0, '${PROJECT_DIR}/scripts/train')
from launch_grpo import apply_all_patches
apply_all_patches('${PROJECT_DIR}')
"

# ===== 数据路径探测 =====
if [ -f "${PROJECT_DIR}/data/rl_kernelbench_cuda/train.parquet" ]; then
    TRAIN_PATH="${PROJECT_DIR}/data/rl_kernelbench_cuda/train.parquet"
    VAL_PATH="${PROJECT_DIR}/data/rl_kernelbench_cuda/val.parquet"
    REWARD_FN_NAME="compute_score_auto"
    echo "Using KernelBench CUDA RL data"
elif [ -f "${PROJECT_DIR}/data/split/rl/train.parquet" ]; then
    TRAIN_PATH="${PROJECT_DIR}/data/split/rl/train.parquet"
    VAL_PATH="${PROJECT_DIR}/data/split/rl/val.parquet"
    REWARD_FN_NAME="compute_score_auto"
    echo "Using KernelBook split RL data"
elif [ -f "${PROJECT_DIR}/data/rl_kernelbench/train.parquet" ]; then
    TRAIN_PATH="${PROJECT_DIR}/data/rl_kernelbench/train.parquet"
    VAL_PATH="${PROJECT_DIR}/data/rl_kernelbench/val.parquet"
    REWARD_FN_NAME="compute_score_auto"
    echo "Using KernelBench RL data"
elif [ -f "${PROJECT_DIR}/data/rl/train.parquet" ]; then
    TRAIN_PATH="${PROJECT_DIR}/data/rl/train.parquet"
    VAL_PATH="${PROJECT_DIR}/data/rl/val.parquet"
    REWARD_FN_NAME="compute_score"
    echo "Using KernelBook RL data"
else
    echo "ERROR: No RL training data found!"
    exit 1
fi

# ===== 模型路径 =====
# 3B 模型用于快速验证方法有效性
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-Coder-3B-Instruct}"
echo "Using model: $MODEL_PATH"

REWARD_FN_PATH="${PROJECT_DIR}/src/reward/kernel_reward.py"

# 检查数据
if [ ! -f "$TRAIN_PATH" ]; then
    echo "ERROR: RL training data not found at $TRAIN_PATH"
    exit 1
fi

echo ""
echo "=== Launching GRPO Training (PolyU, 2× A100 80GB) ==="
echo "Model:  $MODEL_PATH"
echo "Train:  $TRAIN_PATH"
echo "Reward: $REWARD_FN_NAME"
echo ""

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_PATH" \
    data.val_files="$VAL_PATH" \
    data.train_batch_size=8 \
    data.max_prompt_length=2048 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=true \
    data.truncation=error \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=false \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    +actor_rollout_ref.actor.optim.override_optimizer_config.foreach=false \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.max_model_len=6144 \
    actor_rollout_ref.rollout.n=5 \
    algorithm.use_kl_in_reward=false \
    custom_reward_function.path="$REWARD_FN_PATH" \
    custom_reward_function.name="$REWARD_FN_NAME" \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name=kernel_rl \
    trainer.experiment_name=grpo_3b_polyu \
    trainer.default_local_dir="${PROJECT_DIR}/checkpoints/grpo_3b" \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=200 \
    trainer.test_freq=200 \
    trainer.max_actor_ckpt_to_keep=2 \
    trainer.total_epochs=3 \
    "$@"

echo "=== GRPO Training Complete ==="
ray stop --force 2>/dev/null || true
