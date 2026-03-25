#!/bin/bash
# ============================================================
# CityU HPC 环境配置
#
# 用法：
#   bash scripts/slurm/setup_env_cityu.sh
#
# 前提：
#   - 已 clone 代码: git clone https://github.com/WayneWang24/kernel-rl.git
#   - 有 conda/mamba
#
# 兼容性：
#   CUDA Driver 535 → max CUDA 12.2 → cu121 wheels only
#   PyTorch 2.5.1+cu121 + vLLM 0.7.3 + verl 0.4.1
#   verl 0.4.1 使用老 rollout 架构（sync mode, 不需要 vLLM 0.8+）
#   LoRA 支持需要 vLLM >= 0.7.3
# ============================================================

set -euxo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

echo "=== Setting up kernel-rl environment on CityU HPC ==="

# 1. 创建 conda 环境
if conda info --envs | grep -q "kernel-rl"; then
    echo "conda env 'kernel-rl' already exists, skipping creation"
else
    conda create -n kernel-rl python=3.10 -y
fi

eval "$(conda shell.bash hook)"
conda activate kernel-rl

# 2. 安装核心依赖
# CUDA Driver 535 只支持到 CUDA 12.2，必须用 cu121 wheels
# verl 0.4.1 LoRA 需要 vLLM >= 0.7.3, vLLM 0.7.3 需要 PyTorch 2.5.1
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install vllm==0.7.3
pip install verl==0.4.1 --no-deps
pip install flash-attn --no-build-isolation
pip install pandas pyarrow ray tensordict

# 2b. 安装 nvcc（HPC 计算节点无系统 nvcc，load_inline 编译需要）
conda install -c nvidia cuda-nvcc=12.1 cuda-cudart-dev=12.1 -y

# 3. 验证安装
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
python -c "import vllm; print(f'vLLM {vllm.__version__}')" 2>/dev/null || echo "vLLM import warning on login node (OK)"
python -c "import verl; print(f'verl {verl.__version__}')"
nvcc --version || echo "nvcc not available on login node (OK, available on compute nodes)"

# 4. 准备数据（从 cleaned parquet 生成 RL 数据）
echo ""
echo "=== Preparing RL training data ==="

if [ -f "${PROJECT_DIR}/data/rl/train.parquet" ]; then
    echo "RL data already exists, skipping"
elif [ -f "${PROJECT_DIR}/data/split/rl/train.parquet" ]; then
    echo "Split RL data already exists, skipping"
elif [ -f "${PROJECT_DIR}/data/cleaned/kernelbook_clean.parquet" ]; then
    echo "Generating RL data from cleaned parquet..."
    if [ -f "${PROJECT_DIR}/scripts/data/prepare_split_data.py" ]; then
        python "${PROJECT_DIR}/scripts/data/prepare_split_data.py" \
            --input "${PROJECT_DIR}/data/cleaned/kernelbook_clean.parquet" \
            --output_dir "${PROJECT_DIR}/data/split"
    elif [ -f "${PROJECT_DIR}/scripts/data/prepare_rl_data.py" ]; then
        python "${PROJECT_DIR}/scripts/data/prepare_rl_data.py"
    else
        echo "WARNING: No data preparation script found. Please prepare data manually."
    fi
else
    echo "WARNING: No source data found. Please copy data/rl/ or data/split/rl/ from old HPC."
    echo "  Option 1: scp old-hpc:~/workspace/kernel-rl/data/rl/ ${PROJECT_DIR}/data/rl/"
    echo "  Option 2: scp old-hpc:~/workspace/kernel-rl/data/split/ ${PROJECT_DIR}/data/split/"
fi

# 5. 创建日志目录
mkdir -p "${PROJECT_DIR}/logs"

echo ""
echo "=== Setup Complete ==="
echo "Next: sbatch scripts/slurm/run_grpo_cityu.sh"
