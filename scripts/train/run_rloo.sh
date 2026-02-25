#!/bin/bash
# ============================================================
# RLOO RL 训练脚本
#
# 从 SFT checkpoint 开始，使用 RLOO 算法
# 硬件：2× A100 80GB
#
# 用法：
#   bash scripts/train/run_rloo.sh [额外 override 参数]
# ============================================================

set -euxo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

TRAIN_PATH="${PROJECT_DIR}/data/rl/train.parquet"
VAL_PATH="${PROJECT_DIR}/data/rl/val.parquet"

SFT_CHECKPOINT="${PROJECT_DIR}/checkpoints/sft_merged"
if [ -d "$SFT_CHECKPOINT" ]; then
    MODEL_PATH="$SFT_CHECKPOINT"
    echo "Using SFT checkpoint: $MODEL_PATH"
else
    MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-Coder-7B-Instruct}"
    echo "WARNING: SFT checkpoint not found, using base model: $MODEL_PATH"
fi

REWARD_FN_PATH="${PROJECT_DIR}/src/reward/kernel_reward.py"

if [ ! -f "$TRAIN_PATH" ]; then
    echo "ERROR: RL training data not found at $TRAIN_PATH"
    exit 1
fi

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=rloo \
    data.train_files="$TRAIN_PATH" \
    data.val_files="$VAL_PATH" \
    data.train_batch_size=32 \
    data.max_prompt_length=2048 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=true \
    data.truncation=error \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=160 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    algorithm.use_kl_in_reward=true \
    algorithm.kl_penalty=kl \
    algorithm.kl_ctrl.type=fixed \
    algorithm.kl_ctrl.kl_coef=0.001 \
    reward.custom_reward_function.path="$REWARD_FN_PATH" \
    reward.custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=kernel_rl \
    trainer.experiment_name=rloo_qwen25_coder_7b \
    trainer.default_local_dir="${PROJECT_DIR}/checkpoints/rloo" \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=40 \
    "$@"

echo "=== RLOO Training Complete ==="
