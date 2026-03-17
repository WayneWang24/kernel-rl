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
eval "$(conda shell.bash hook)"
conda activate kernel-rl

PROJECT_DIR="${HOME}/ChenweiWang/workspace/kernel-rl"
mkdir -p "${PROJECT_DIR}/logs"

# ===== 环境变量 =====
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export RAY_memory_monitor_refresh_ms=0
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCHDYNAMO_DISABLE=1

# 禁用 expandable_segments（与 vLLM memory pool 不兼容）
unset PYTORCH_CUDA_ALLOC_CONF
# 清除 ROCm 变量
unset ROCR_VISIBLE_DEVICES

# ===== Step 1: 应用 verl 补丁 =====
echo "=== Applying verl patches ==="
python -c "
import sys; sys.path.insert(0, '${PROJECT_DIR}/scripts/train')
from launch_grpo import apply_all_patches
apply_all_patches('${PROJECT_DIR}')
"

# ===== Step 2: 数据路径探测 =====
if [ -f "${PROJECT_DIR}/data/split/rl/train.parquet" ]; then
    TRAIN_PATH="${PROJECT_DIR}/data/split/rl/train.parquet"
    VAL_PATH="${PROJECT_DIR}/data/split/rl/val.parquet"
    REWARD_FN_NAME="compute_score_auto"
    echo "Using KernelBook split RL data (ModelNew format)"
elif [ -f "${PROJECT_DIR}/data/rl_kernelbench/train.parquet" ]; then
    TRAIN_PATH="${PROJECT_DIR}/data/rl_kernelbench/train.parquet"
    VAL_PATH="${PROJECT_DIR}/data/rl_kernelbench/val.parquet"
    REWARD_FN_NAME="compute_score_auto"
    echo "Using KernelBench RL data (ModelNew format)"
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

# ===== Step 3: 启动 GRPO 训练 =====
echo ""
echo "=== Launching GRPO Training (1 node, 3 GPUs) ==="
echo "Model:     $MODEL_PATH"
echo "Train:     $TRAIN_PATH"
echo "Reward:    $REWARD_FN_NAME"
echo ""

# 单节点不需要手动启动 Ray 集群，verl 会自动 ray.init()
# 3× A100 40GB 跑 7B 内存计算：
#   - FSDP actor (bf16): 7B × 2 / 3 ≈ 4.7GB/GPU
#   - optimizer offload 到 CPU: 0GB/GPU
#   - vLLM (0.30): ~12GB/GPU
#   - 总计: ~17GB/GPU → 余量充足
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_PATH" \
    data.val_files="$VAL_PATH" \
    data.train_batch_size=12 \
    data.max_prompt_length=2048 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=true \
    data.truncation=error \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=false \
    actor_rollout_ref.actor.ppo_mini_batch_size=12 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    +actor_rollout_ref.actor.optim.override_optimizer_config.foreach=false \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.30 \
    actor_rollout_ref.rollout.max_model_len=6144 \
    actor_rollout_ref.rollout.n=5 \
    algorithm.use_kl_in_reward=false \
    custom_reward_function.path="$REWARD_FN_PATH" \
    custom_reward_function.name="$REWARD_FN_NAME" \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name=kernel_rl \
    trainer.experiment_name=grpo_qwen25_coder_7b_1node \
    trainer.default_local_dir="${PROJECT_DIR}/checkpoints/grpo" \
    trainer.n_gpus_per_node=3 \
    trainer.nnodes=1 \
    trainer.save_freq=200 \
    trainer.test_freq=200 \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.total_epochs=3 \
    2>&1 | tee "${PROJECT_DIR}/logs/grpo_cityu_1node_${SLURM_JOB_ID}.log"

echo "=== GRPO Training Complete ==="
