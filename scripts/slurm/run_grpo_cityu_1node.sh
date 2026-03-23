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

# 确保 CUDA 设备可见
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES=0,1,2
fi
export WANDB_MODE=disabled

# 【核心修复】禁止 Ray 覆盖 CUDA_VISIBLE_DEVICES
# 根因：verl 创建 TaskRunner actor 时不请求 GPU，Ray 给它设 CUDA_VISIBLE_DEVICES=""，
# 导致 TaskRunner 内 torch.cuda.is_available()=False → get_device_name()="cpu"，
# 级联导致所有 worker 也不请求 GPU 资源。
# 设置此变量后，所有 Ray 进程继承父进程的 CUDA_VISIBLE_DEVICES=0,1,2，
# verl 自己的 _setup_env_cuda_visible_devices() 负责每个 worker 的 GPU 分配。
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1

# NCCL 修复：PCIe A100 之间 P2P 可能不可用，禁用后走 SHM
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN

# ===== Step 0: 清理残留 Ray 集群 =====
ray stop --force 2>/dev/null || true
unset RAY_ADDRESS

# ===== Step 0.5: GPU 诊断 =====
echo "=== GPU Diagnostics ==="
nvidia-smi || echo "WARNING: nvidia-smi not found or failed"
python3 -c "import torch; print(f'torch.cuda.is_available()={torch.cuda.is_available()}, device_count={torch.cuda.device_count()}')"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

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

# ===== Step 3: 启动 GRPO 训练 =====
echo ""
echo "=== Launching GRPO Training (1 node, 3 GPUs) ==="
echo "Model:     $MODEL_PATH"
echo "Train:     $TRAIN_PATH"
echo "Reward:    $REWARD_FN_NAME"
echo ""

# 单节点不需要手动启动 Ray 集群，verl 会自动 ray.init()
# 3× A100 40GB 跑 7B 内存计算（rollout phase）：
#   - FSDP actor shard (bf16): 7B × 2 / 3 ≈ 4.7GB/GPU
#   - SGLang model (full): ~14GB/GPU
#   - SGLang KV cache: 0.60 × 40 - 14 ≈ 10GB/GPU
#   - 总计: ~29GB/GPU → 40GB 内可以
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
    trainer.n_gpus_per_node=3 \
    trainer.nnodes=1 \
    trainer.save_freq=200 \
    trainer.test_freq=200 \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.total_epochs=3 \
    2>&1 | tee "${PROJECT_DIR}/logs/grpo_cityu_1node_${SLURM_JOB_ID}.log"

echo "=== GRPO Training Complete ==="
ray stop --force 2>/dev/null || true
