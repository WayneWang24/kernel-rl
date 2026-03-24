#!/bin/bash
# ============================================================
# CityU HPC — KernelBench 编译验证评测
#
# 两阶段：vLLM 生成 → 编译+运行验证
# 硬件：1 节点, 1 GPU (vLLM TP=1, 7B 单卡可跑)
#
# 用法：
#   sbatch scripts/slurm/run_eval_compile_cityu.sh
# ============================================================

#SBATCH --job-name=kernel-eval
#SBATCH --partition=gpu3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euxo pipefail

set +u
eval "$(conda shell.bash hook)"
conda activate kernel-rl
set -u

PROJECT_DIR="${HOME}/ChenweiWang/workspace/kernel-rl"
KERNELBENCH_DIR="${HOME}/ChenweiWang/workspace/my-kernel-bench"

# ===== 配置 =====
MODEL_PATH="${PROJECT_DIR}/checkpoints/sft_modelnew_merged"
RUN_NAME="sft-modelnew-compile"
LEVELS="1 2"
BACKEND="triton"  # SFT 模型训练在 Triton/ModelNew 格式上

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "============================================"
echo "  KernelBench Compile Evaluation (CityU)"
echo "============================================"
echo "Model:   $MODEL_PATH"
echo "Levels:  $LEVELS"
echo "Backend: $BACKEND"
echo "============================================"

# ===== 阶段 A: vLLM 生成 =====
echo ""
echo "=== Phase A: Generation (vLLM TP=1) ==="

CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 10240 \
    --dtype half \
    --enforce-eager \
    --port 8000 &
VLLM_PID=$!

# 等待 vLLM 就绪（enforce-eager 跳过 cudagraph，启动更快）
echo "Waiting for vLLM server..."
for i in $(seq 1 300); do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "vLLM ready! (${i}s)"
        break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "ERROR: vLLM died"; exit 1
    fi
    if [ "$i" -eq 300 ]; then
        echo "ERROR: vLLM timeout"; kill $VLLM_PID 2>/dev/null; exit 1
    fi
    sleep 1
done

# 生成
python "${PROJECT_DIR}/scripts/eval/eval_with_compile.py" \
    --model_path "$MODEL_PATH" \
    --run_name "$RUN_NAME" \
    --levels $LEVELS \
    --kernelbench_dir "$KERNELBENCH_DIR" \
    --output_dir "${PROJECT_DIR}/results" \
    --api_base "http://localhost:8000/v1" \
    --model_name "$MODEL_PATH" \
    --max_tokens 8192 \
    --backend "$BACKEND" \
    --generate_only

# 关闭 vLLM（释放 GPU 给阶段 B）
echo "Shutting down vLLM..."
kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true
sleep 5

echo "Phase A complete."

# ===== 阶段 B: 编译验证 =====
echo ""
echo "=== Phase B: Compile + Verify ==="

python "${PROJECT_DIR}/scripts/eval/eval_with_compile.py" \
    --run_name "$RUN_NAME" \
    --levels $LEVELS \
    --kernelbench_dir "$KERNELBENCH_DIR" \
    --output_dir "${PROJECT_DIR}/results" \
    --num_workers 4 \
    --num_gpus 1 \
    --eval_only

echo ""
echo "=== Evaluation Complete ==="
echo "Results: ${PROJECT_DIR}/results/${RUN_NAME}/"
