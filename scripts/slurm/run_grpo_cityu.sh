#!/bin/bash
# ============================================================
# CityU HPC SLURM 作业脚本 — GRPO 训练
#
# 硬件：2 节点 × 3× A100 40GB = 6 GPU
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
#SBATCH --nodes=2
#SBATCH --nodelist=gpu13,gpu15
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=3
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=720:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euxo pipefail

# ===== 环境 =====
# conda activate 期间关闭 -u（cuda-nvcc 激活脚本有未定义变量）
set +u
eval "$(conda shell.bash hook)"
conda activate kernel-rl
set -u

# SLURM 会把脚本拷贝到 /var/spool，不能用 $0 推导路径
PROJECT_DIR="${HOME}/ChenweiWang/workspace/kernel-rl"

mkdir -p "${PROJECT_DIR}/logs"

# ===== 环境变量 =====
# vLLM rollout (SGLang ABI 不兼容 torch 2.4)
export RAY_memory_monitor_refresh_ms=0
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCHDYNAMO_DISABLE=1

# 禁用 expandable_segments（与 SGLang/vLLM memory pool 不兼容）
unset PYTORCH_CUDA_ALLOC_CONF
# 清除 ROCm 变量（SLURM 可能自动设置，与 CUDA_VISIBLE_DEVICES 冲突）
unset ROCR_VISIBLE_DEVICES

# GPU 管理由 Ray + patch_verl_force_cuda 补丁处理（同 1node 脚本注释）
export WANDB_MODE=disabled

# NCCL：PCIe A100 之间 P2P 不可用，但节点间有 100Gbps IB
export NCCL_P2P_DISABLE=1
# IB 可用（mlx5_0, 100Gbps），不再禁用
export NCCL_DEBUG=WARN

# 清理残留 Ray 集群
ray stop --force 2>/dev/null || true
unset RAY_ADDRESS

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
echo "GPUs/node:    $SLURM_GPUS_PER_NODE (requested)"
echo "============================================"

# ===== Step 0.5: Per-node GPU health check =====
echo ""
echo "=== GPU Health Check (all nodes) ==="
GPU_CHECK_DIR="${PROJECT_DIR}/logs/gpu_check_${SLURM_JOB_ID}"
for node in "${nodes_array[@]}"; do
    echo "Checking $node..."
    srun --nodes=1 --ntasks=1 -w "$node" \
        python3 "${PROJECT_DIR}/scripts/slurm/check_node_gpus.py" "$GPU_CHECK_DIR" "$node"
done

# Read results and find minimum working GPUs across nodes
MIN_GPUS=99
for node in "${nodes_array[@]}"; do
    GPU_FILE="${GPU_CHECK_DIR}/${node}.txt"
    if [ -f "$GPU_FILE" ]; then
        GPU_IDS=$(cat "$GPU_FILE" | tr -d '\n')
        NUM=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l | tr -d ' ')
        echo "  $node: $NUM working GPUs (CVD=$GPU_IDS)"
        if [ "$NUM" -lt "$MIN_GPUS" ]; then MIN_GPUS=$NUM; fi
    else
        echo "  WARNING: No GPU info for $node"
        MIN_GPUS=0
    fi
done

if [ "$MIN_GPUS" -lt 2 ]; then
    echo "ERROR: Need at least 2 working GPUs per node, min found: $MIN_GPUS"
    exit 1
fi

GPUS_PER_NODE=$MIN_GPUS
TOTAL_GPUS=$((GPUS_PER_NODE * SLURM_JOB_NUM_NODES))
echo ""
echo "Using $GPUS_PER_NODE GPUs/node × $SLURM_JOB_NUM_NODES nodes = $TOTAL_GPUS total GPUs"

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

# ===== Step 1.5: 修复 Click Sentinel deepcopy bug =====
# Ray 2.40 的 ray start 会 deepcopy Click 命令对象，Click 的 Sentinel(enum.Enum)
# 用 object() 作值导致 deepcopy 失败。给 Sentinel 加 __deepcopy__ 方法。
echo "=== Patching Click Sentinel for Ray compat ==="
srun --nodes=1 --ntasks=1 -w "$head_node" \
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

# ===== Step 2: 启动 Ray 集群（使用检测到的 GPU）=====
# 每个节点只暴露检测通过的 GPU，截断到 MIN_GPUS 保持一致
HEAD_GPUS_ALL=$(cat "${GPU_CHECK_DIR}/${head_node}.txt" | tr -d '\n')
HEAD_CVD=$(echo "$HEAD_GPUS_ALL" | tr ',' '\n' | head -$GPUS_PER_NODE | paste -sd,)

echo ""
echo "=== Starting Ray HEAD at $head_node (CVD=$HEAD_CVD, num_gpus=$GPUS_PER_NODE) ==="
srun --nodes=1 --ntasks=1 -w "$head_node" \
    env CUDA_VISIBLE_DEVICES="$HEAD_CVD" \
    ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "$GPUS_PER_NODE" --block &
sleep 10

worker_num=$((SLURM_JOB_NUM_NODES - 1))
for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    WORKER_GPUS_ALL=$(cat "${GPU_CHECK_DIR}/${node_i}.txt" | tr -d '\n')
    WORKER_CVD=$(echo "$WORKER_GPUS_ALL" | tr ',' '\n' | head -$GPUS_PER_NODE | paste -sd,)
    echo "Starting Ray WORKER $i at $node_i (CVD=$WORKER_CVD, num_gpus=$GPUS_PER_NODE)"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        env CUDA_VISIBLE_DEVICES="$WORKER_CVD" \
        ray start --address "$ip_head" \
        --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "$GPUS_PER_NODE" --block &
    sleep 5
done

# 等待所有节点加入
echo "Waiting for Ray cluster to stabilize..."
sleep 15

# 验证 Ray 集群
srun --overlap --nodes=1 --ntasks=1 -w "$head_node" \
    ray status

# ===== Step 3: 数据路径探测 =====
# 优先 混合数据 → CUDA 数据 → KernelBook split → KernelBench Triton → KernelBook 原始
if [ -f "${PROJECT_DIR}/data/rl_mixed/train.parquet" ]; then
    TRAIN_PATH="${PROJECT_DIR}/data/rl_mixed/train.parquet"
    VAL_PATH="${PROJECT_DIR}/data/rl_mixed/val.parquet"
    REWARD_FN_NAME="compute_score_auto"
    echo "Using mixed RL data (KernelBook static+compile + KernelBench compile+run)"
elif [ -f "${PROJECT_DIR}/data/rl_kernelbench_cuda/train.parquet" ]; then
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

# ===== Step 4: 动态计算 batch size =====
TRAIN_BS=$((TOTAL_GPUS * 4))
MINI_BS=$TRAIN_BS

# ===== Step 5: 启动 GRPO 训练 =====
echo ""
echo "=== Launching GRPO Training ==="
echo "Model:     $MODEL_PATH"
echo "Train:     $TRAIN_PATH"
echo "Reward:    $REWARD_FN_NAME"
echo "GPUs:      $TOTAL_GPUS ($GPUS_PER_NODE/node × $SLURM_JOB_NUM_NODES nodes)"
echo "BatchSize: train=$TRAIN_BS mini=$MINI_BS"
echo ""

PYTHONUNBUFFERED=1 srun --overlap --nodes=1 --ntasks=1 -w "$head_node" \
    python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_PATH" \
    data.val_files="$VAL_PATH" \
    data.train_batch_size=$TRAIN_BS \
    data.max_prompt_length=2048 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=true \
    data.truncation=error \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr=5e-5 \
    actor_rollout_ref.model.use_remove_padding=false \
    actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BS \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=16 \
    actor_rollout_ref.model.target_modules=all-linear \
    +actor_rollout_ref.model.override_config.attn_implementation=flash_attention_2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    +actor_rollout_ref.actor.optim.override_optimizer_config.foreach=false \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=sync \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enforce_eager=true \
    actor_rollout_ref.rollout.free_cache_engine=true \
    actor_rollout_ref.rollout.load_format=safetensors \
    ++actor_rollout_ref.rollout.layered_summon=true \
    actor_rollout_ref.rollout.max_model_len=6144 \
    actor_rollout_ref.rollout.n=5 \
    algorithm.use_kl_in_reward=false \
    custom_reward_function.path="$REWARD_FN_PATH" \
    custom_reward_function.name="$REWARD_FN_NAME" \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name=kernel_rl \
    trainer.experiment_name=grpo_cuda_vllm_${TOTAL_GPUS}gpu \
    trainer.default_local_dir="${PROJECT_DIR}/checkpoints/grpo_cuda" \
    trainer.n_gpus_per_node=$GPUS_PER_NODE \
    trainer.nnodes=$SLURM_JOB_NUM_NODES \
    trainer.resume_mode=disable \
    trainer.save_freq=50 \
    trainer.test_freq=-1 \
    trainer.val_before_train=false \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.total_epochs=3 \
    2>&1 | tee "${PROJECT_DIR}/logs/grpo_cityu_${SLURM_JOB_ID}.log"

echo "=== GRPO Training Complete ==="
