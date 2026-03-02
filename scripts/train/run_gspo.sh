#!/bin/bash
# ============================================================
# GSPO RL 训练脚本
#
# GSPO = GRPO advantage + 序列级重要性比 + 超小裁剪
# 相比 GRPO：更稳定的训练、更快的收敛
#
# 硬件：2× A100 80GB
#
# 用法：
#   bash scripts/train/run_gspo.sh [额外 override 参数]
#
# 前提：
#   - 已完成 SFT 训练并合并 LoRA（checkpoints/sft_merged/）
#   - 已执行 prepare_rl_data.py 生成 data/rl/*.parquet
# ============================================================

set -euxo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# 数据路径
TRAIN_PATH="${PROJECT_DIR}/data/rl/train.parquet"
VAL_PATH="${PROJECT_DIR}/data/rl/val.parquet"

# 模型：使用 SFT checkpoint（需要先 merge LoRA）
SFT_CHECKPOINT="${PROJECT_DIR}/checkpoints/sft_merged"
if [ -d "$SFT_CHECKPOINT" ]; then
    MODEL_PATH="$SFT_CHECKPOINT"
    echo "Using SFT checkpoint: $MODEL_PATH"
else
    MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-Coder-7B-Instruct}"
    echo "WARNING: SFT checkpoint not found, using base model: $MODEL_PATH"
fi

# Reward 函数
REWARD_FN_PATH="${PROJECT_DIR}/src/reward/kernel_reward.py"

# 检查数据
if [ ! -f "$TRAIN_PATH" ]; then
    echo "ERROR: RL training data not found at $TRAIN_PATH"
    echo "Please run: python scripts/data/prepare_rl_data.py first"
    exit 1
fi

# ===== GSPO 核心参数（论文推荐值）=====
CLIP_RATIO_LOW=3e-4      # 超小裁剪下界（PPO 用 0.2，GSPO 小 ~600 倍）
CLIP_RATIO_HIGH=4e-4     # 超小裁剪上界
LOSS_AGG_MODE="seq-mean-token-mean"  # 序列级聚合

# ===== 训练规模（适配 2× A100 80GB）=====
TRAIN_BATCH_SIZE=16       # 每 step 的 prompt 数
N_RESP=8                  # 每 prompt 生成 8 个候选（2 GPU 用 8 而非 16）
PPO_MINI_BATCH_SIZE=64    # 16×8/2 = 64，保持 ~2 mini-batches
PPO_MICRO_BATCH_SIZE=2    # 每 GPU 微批次

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_PATH" \
    data.val_files="$VAL_PATH" \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=2048 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=true \
    data.truncation=error \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05 \
    actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
    actor_rollout_ref.actor.loss_agg_mode=${LOSS_AGG_MODE} \
    actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO_LOW} \
    actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE} \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=${N_RESP} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    algorithm.use_kl_in_reward=false \
    reward.custom_reward_function.path="$REWARD_FN_PATH" \
    reward.custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=kernel_rl \
    trainer.experiment_name=gspo_qwen25_coder_7b \
    trainer.default_local_dir="${PROJECT_DIR}/checkpoints/gspo" \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=40 \
    "$@"

echo "=== GSPO Training Complete ==="
