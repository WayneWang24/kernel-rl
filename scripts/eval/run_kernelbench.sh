#!/bin/bash
# ============================================================
# KernelBench 评测脚本
#
# 对指定模型在 KernelBench 上运行评测，输出 fast_p 指标。
#
# 用法：
#   bash scripts/eval/run_kernelbench.sh <model_path> <run_name> [levels]
#
# 示例：
#   # 评测 baseline
#   bash scripts/eval/run_kernelbench.sh Qwen/Qwen2.5-Coder-7B-Instruct baseline
#
#   # 评测 SFT checkpoint
#   bash scripts/eval/run_kernelbench.sh checkpoints/sft_merged sft-only
#
#   # 评测 GRPO checkpoint
#   bash scripts/eval/run_kernelbench.sh checkpoints/grpo/epoch_40 grpo-epoch40
#
#   # 指定 level
#   bash scripts/eval/run_kernelbench.sh checkpoints/grpo/epoch_40 grpo-l1 "1"
#
# 前提：
#   - KernelBench 已克隆到 KERNELBENCH_DIR（默认 /tmp/KernelBench）
#   - GPU 可用
# ============================================================

set -euxo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# 参数
MODEL_PATH="${1:?Usage: $0 <model_path> <run_name> [levels]}"
RUN_NAME="${2:?Usage: $0 <model_path> <run_name> [levels]}"
LEVELS="${3:-1 2}"

# KernelBench 目录
KERNELBENCH_DIR="${KERNELBENCH_DIR:-/tmp/KernelBench}"
RESULTS_DIR="${PROJECT_DIR}/results/${RUN_NAME}"

# 检查 KernelBench
if [ ! -d "$KERNELBENCH_DIR" ]; then
    echo "KernelBench not found. Cloning..."
    git clone https://github.com/ScalingIntelligence/KernelBench "$KERNELBENCH_DIR"
fi

mkdir -p "$RESULTS_DIR"

echo "=== KernelBench Evaluation ==="
echo "Model: $MODEL_PATH"
echo "Run name: $RUN_NAME"
echo "Levels: $LEVELS"
echo "Results dir: $RESULTS_DIR"

# 方式 1：使用 vLLM 启动模型服务，然后用 KernelBench 评测
# 这是推荐的方式，因为可以复用 vLLM 的推理优化

# 启动 vLLM 服务（后台）
echo "Starting vLLM server..."
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 6144 \
    --port 8000 &
VLLM_PID=$!

# 等待服务就绪
echo "Waiting for vLLM server to start..."
for i in $(seq 1 60); do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "vLLM server ready!"
        break
    fi
    if [ $i -eq 60 ]; then
        echo "ERROR: vLLM server failed to start"
        kill $VLLM_PID 2>/dev/null || true
        exit 1
    fi
    sleep 5
done

# 对每个 level 运行评测
for level in $LEVELS; do
    echo ""
    echo "=== Evaluating Level $level ==="

    python "${PROJECT_DIR}/scripts/eval/eval_kernelbench.py" \
        --kernelbench_dir "$KERNELBENCH_DIR" \
        --level "$level" \
        --api_base "http://localhost:8000/v1" \
        --model_name "$MODEL_PATH" \
        --output_dir "$RESULTS_DIR" \
        --run_name "$RUN_NAME" \
        --n_samples 1 \
        --temperature 0.0 \
        --max_tokens 4096
done

# 关闭 vLLM 服务
echo "Shutting down vLLM server..."
kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true

echo ""
echo "=== Evaluation Complete ==="
echo "Results saved to: $RESULTS_DIR"
echo ""
echo "To compare results: python scripts/eval/compare_results.py --results_dir results/"
