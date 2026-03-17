#!/bin/bash
# ============================================================
# CityU HPC SLURM 作业脚本 — GRPO 训练
#
# 硬件：3 节点 × 3× A100 40GB = 9 GPU
# 分区：gpu3
#
# 用法：
#   sbatch scripts/slurm/run_grpo_cityu.sh
#
# 前提：
#   - conda env 'kernel-rl' 已创建（bash scripts/slurm/setup_env_cityu.sh）
#   - 训练数据已准备（data/rl/ 或 data/split/rl/）
# ============================================================

#SBATCH --job-name=kernel-rl-grpo
#SBATCH --partition=gpu3
#SBATCH --nodes=3
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

# SLURM 会把脚本拷贝到 /var/spool，不能用 $0 推导路径
PROJECT_DIR="${HOME}/ChenweiWang/workspace/kernel-rl"

mkdir -p "${PROJECT_DIR}/logs"

# ===== 环境变量 =====
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export RAY_memory_monitor_refresh_ms=0
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

# 禁用 expandable_segments（与 vLLM memory pool 不兼容）
unset PYTORCH_CUDA_ALLOC_CONF

# ===== Ray 集群设置 =====
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# 处理 IPv6
if [[ "$head_node_ip" == *" "* ]]; then
    IFS=' ' read -ra ADDR <<< "$head_node_ip"
    if [[ ${#ADDR[0]} -gt 16 ]]; then
        head_node_ip=${ADDR[1]}
    else
        head_node_ip=${ADDR[0]}
    fi
    echo "IPv6 detected, using IPv4: $head_node_ip"
fi

port=6379
ip_head=$head_node_ip:$port
export ip_head

echo "============================================"
echo "  kernel-rl GRPO Training (CityU HPC)"
echo "============================================"
echo "Head node:    $head_node ($head_node_ip)"
echo "Nodes:        $SLURM_JOB_NUM_NODES"
echo "GPUs/node:    $SLURM_GPUS_PER_NODE"
echo "Total GPUs:   $((SLURM_JOB_NUM_NODES * SLURM_GPUS_PER_NODE))"
echo "============================================"

# ===== Step 1: 在 head 节点上应用 verl 补丁 =====
# launch_grpo.py 里的补丁会修改 pip 安装的 verl 文件
# 在共享文件系统上只需执行一次，所有节点都能读到
echo ""
echo "=== Applying verl patches on head node ==="
srun --nodes=1 --ntasks=1 -w "$head_node" \
    python -c "
import sys; sys.path.insert(0, '${PROJECT_DIR}/scripts/train')
from launch_grpo import apply_all_patches
apply_all_patches('${PROJECT_DIR}')
"

# ===== Step 2: 启动 Ray 集群 =====
echo ""
echo "=== Starting Ray HEAD at $head_node ==="
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_NODE}" --block &
sleep 10

worker_num=$((SLURM_JOB_NUM_NODES - 1))
for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting Ray WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray start --address "$ip_head" \
        --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_NODE}" --block &
    sleep 5
done

# 等待所有节点加入
echo "Waiting for Ray cluster to stabilize..."
sleep 15

# 验证 Ray 集群
srun --overlap --nodes=1 --ntasks=1 -w "$head_node" \
    ray status

# ===== Step 3: 数据路径探测 =====
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
    echo "Please run setup_env_cityu.sh first or copy data manually."
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

# ===== Step 4: 启动 GRPO 训练 =====
echo ""
echo "=== Launching GRPO Training ==="
echo "Model:     $MODEL_PATH"
echo "Train:     $TRAIN_PATH"
echo "Reward:    $REWARD_FN_NAME"
echo ""

PYTHONUNBUFFERED=1 srun --overlap --nodes=1 --ntasks=1 -w "$head_node" \
    python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_PATH" \
    data.val_files="$VAL_PATH" \
    data.train_batch_size=32 \
    data.max_prompt_length=2048 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=true \
    data.truncation=error \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=false \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    +actor_rollout_ref.actor.optim.override_optimizer_config.foreach=false \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
    actor_rollout_ref.rollout.max_model_len=6144 \
    actor_rollout_ref.rollout.n=5 \
    algorithm.use_kl_in_reward=false \
    +reward.custom_reward_function.path="$REWARD_FN_PATH" \
    +reward.custom_reward_function.name="$REWARD_FN_NAME" \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name=kernel_rl \
    trainer.experiment_name=grpo_qwen25_coder_7b_9gpu \
    trainer.default_local_dir="${PROJECT_DIR}/checkpoints/grpo" \
    trainer.n_gpus_per_node=3 \
    trainer.nnodes=3 \
    trainer.save_freq=200 \
    trainer.test_freq=200 \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.total_epochs=3 \
    2>&1 | tee "${PROJECT_DIR}/logs/grpo_cityu_${SLURM_JOB_ID}.log"

echo "=== GRPO Training Complete ==="
