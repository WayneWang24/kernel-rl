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

# 2. 安装依赖
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install verl==0.7.0
pip install flash-attn --no-build-isolation
pip install pandas pyarrow

# 2b. 安装 nvcc（HPC 计算节点无系统 nvcc，load_inline 编译需要）
conda install -c nvidia cuda-nvcc=12.1 cuda-cudart-dev=12.1 -y

# 2c. 安装 SGLang（替代 vLLM，GRPO 多采样前缀共享更优）
pip install "sglang[all]"
pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/

# 3. 验证安装
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}')"
python -c "import verl; print(f'verl installed at {verl.__file__}')"
python -c "import sglang; print(f'SGLang {sglang.__version__}')"
nvcc --version
python -c "from torch.utils.cpp_extension import load_inline; print('load_inline OK')"

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
