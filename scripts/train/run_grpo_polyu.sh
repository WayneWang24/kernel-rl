#!/bin/bash
# ============================================================
# PolyU VM 直接运行 — GRPO 训练 (3B, compile+run reward)
#
# 硬件：2× A100 80GB（虚拟机，非 SLURM）
#
# 用法：
#   nohup bash scripts/train/run_grpo_polyu.sh > logs/grpo_polyu.log 2>&1 &
#
# 前提：
#   - conda env 'kernel-rl' 已激活
#   - rl_kernelbench_cuda 数据已准备：
#     python scripts/data/prepare_rl_kernelbench.py \
#         --kernelbench_dir ~/workspace/my-kernel-bench \
#         --output_dir data/rl_kernelbench_cuda --backend cuda
#   - nvcc 可用（conda install -c nvidia cuda-nvcc=12.1 cuda-cudart-dev=12.1）
# ============================================================

set -euxo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
mkdir -p "${PROJECT_DIR}/logs"

# ===== 环境变量 =====
export RAY_memory_monitor_refresh_ms=0
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export TORCHDYNAMO_DISABLE=1
export WANDB_MODE=disabled

unset PYTORCH_CUDA_ALLOC_CONF
unset ROCR_VISIBLE_DEVICES

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN

# ===== Step 0: 清理残留 Ray 集群 =====
ray stop --force 2>/dev/null || true
unset RAY_ADDRESS

# ===== Step 0.5: GPU 健康检查 =====
echo "=== GPU Health Check ==="
nvidia-smi || { echo "ERROR: nvidia-smi failed"; exit 1; }

NUM_WORKING=$(python3 -c "import torch; print(torch.cuda.device_count())")
echo "Detected $NUM_WORKING GPUs"

if [ "$NUM_WORKING" -lt 2 ]; then
    echo "ERROR: Need at least 2 GPUs, found $NUM_WORKING"
    exit 1
fi

# ===== Step 1: 应用 verl 补丁 =====
echo "=== Applying verl patches ==="
python -c "
import sys; sys.path.insert(0, '${PROJECT_DIR}/scripts/train')
from launch_grpo import apply_all_patches
apply_all_patches('${PROJECT_DIR}')
"

# ===== Step 1.5: 修复 Click Sentinel deepcopy bug =====
python -c "
import click._utils, inspect
src = inspect.getfile(click._utils)
with open(src) as f:
    content = f.read()
if '__deepcopy__' not in content:
    content = content.replace(
        'class Sentinel(enum.Enum):',
        'class Sentinel(enum.Enum):\n    def __deepcopy__(self, memo):\n        return self\n    def __copy__(self):\n        return self\n',
    )
    with open(src, 'w') as f:
        f.write(content)
    print('[patch] Click Sentinel patched for deepcopy')
else:
    print('[patch] Click Sentinel already patched')
"

# ===== Step 2: 数据路径探测 =====
# 优先 CUDA 数据（compile+run reward）
if [ -f "${PROJECT_DIR}/data/rl_kernelbench_cuda/train.parquet" ]; then
    TRAIN_PATH="${PROJECT_DIR}/data/rl_kernelbench_cuda/train.parquet"
    VAL_PATH="${PROJECT_DIR}/data/rl_kernelbench_cuda/val.parquet"
    REWARD_FN_NAME="compute_score_auto"
    echo "Using KernelBench CUDA RL data (compile+run reward)"
elif [ -f "${PROJECT_DIR}/data/split/rl/train.parquet" ]; then
    TRAIN_PATH="${PROJECT_DIR}/data/split/rl/train.parquet"
    VAL_PATH="${PROJECT_DIR}/data/split/rl/val.parquet"
    REWARD_FN_NAME="compute_score_auto"
    echo "WARNING: Using KernelBook split RL data (static reward only!)"
else
    echo "ERROR: No RL training data found!"
    echo "Run: python scripts/data/prepare_rl_kernelbench.py --backend cuda --output_dir data/rl_kernelbench_cuda"
    exit 1
fi

# 模型路径：优先 3B SFT checkpoint → 3B base model
SFT_CHECKPOINT_3B="${PROJECT_DIR}/checkpoints/sft_3b_merged"
if [ -d "$SFT_CHECKPOINT_3B" ]; then
    MODEL_PATH="$SFT_CHECKPOINT_3B"
    echo "Using 3B SFT checkpoint: $MODEL_PATH"
else
    MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-Coder-3B-Instruct}"
    echo "Using 3B base model: $MODEL_PATH"
fi

REWARD_FN_PATH="${PROJECT_DIR}/src/reward/kernel_reward.py"

# ===== Step 3: 动态计算 batch size =====
TRAIN_BS=$((NUM_WORKING * 4))    # 每 GPU 4 samples
MINI_BS=$TRAIN_BS

# ===== Step 4: 启动 GRPO 训练 =====
echo ""
echo "============================================"
echo "  GRPO 3B Training (PolyU VM)"
echo "============================================"
echo "Model:     $MODEL_PATH"
echo "Train:     $TRAIN_PATH"
echo "Reward:    $REWARD_FN_NAME (compile+run)"
echo "GPUs:      $NUM_WORKING"
echo "BatchSize: train=$TRAIN_BS mini=$MINI_BS"
echo "============================================"
echo ""

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
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=sync \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enforce_eager=true \
    actor_rollout_ref.rollout.free_cache_engine=true \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.max_model_len=6144 \
    actor_rollout_ref.rollout.n=5 \
    algorithm.use_kl_in_reward=false \
    custom_reward_function.path="$REWARD_FN_PATH" \
    custom_reward_function.name="$REWARD_FN_NAME" \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name=kernel_rl \
    trainer.experiment_name=grpo_3b_cuda_${NUM_WORKING}gpu \
    trainer.default_local_dir="${PROJECT_DIR}/checkpoints/grpo_3b_cuda" \
    trainer.n_gpus_per_node=$NUM_WORKING \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=-1 \
    trainer.val_before_train=false \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.total_epochs=3 \
    2>&1 | tee "${PROJECT_DIR}/logs/grpo_polyu.log"

echo "=== GRPO 3B Training Complete ==="
ray stop --force 2>/dev/null || true
