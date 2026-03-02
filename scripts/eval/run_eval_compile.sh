#!/bin/bash
# ============================================================
# 编译验证评测驱动脚本
#
# 分两阶段避免 GPU 冲突：
#   阶段 A: 启动 vLLM (2 GPU) → 生成 ModelNew 代码 → 关闭 vLLM
#   阶段 B: 用 my-kernel-bench 编译 + GPU 验证 (2 GPU)
#
# 用法：
#   bash scripts/eval/run_eval_compile.sh <model_path> <run_name> [levels]
#
# 示例：
#   # 评测 baseline（zero-shot）
#   bash scripts/eval/run_eval_compile.sh Qwen/Qwen2.5-Coder-7B-Instruct baseline
#
#   # 评测 SFT 后模型
#   bash scripts/eval/run_eval_compile.sh checkpoints/sft_merged sft-only
#
#   # 评测 GRPO checkpoint，只评 level 1
#   bash scripts/eval/run_eval_compile.sh checkpoints/grpo/epoch_20 grpo-e20 "1"
#
#   # 仅做评测（已有生成结果）
#   bash scripts/eval/run_eval_compile.sh SKIP grpo-e20 "1 2 3 4"
#
# 前提：
#   - my-kernel-bench 在 ~/Code/my-kernel-bench（或设置 KERNELBENCH_DIR）
#   - 2× GPU 可用
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# 参数
MODEL_PATH="${1:?Usage: $0 <model_path|SKIP> <run_name> [levels]}"
RUN_NAME="${2:?Usage: $0 <model_path|SKIP> <run_name> [levels]}"
LEVELS="${3:-1 2 3 4}"

# 配置
KERNELBENCH_DIR="${KERNELBENCH_DIR:-$HOME/Code/my-kernel-bench}"
RESULTS_DIR="${PROJECT_DIR}/results"
VLLM_PORT="${VLLM_PORT:-8000}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
NUM_COMPILE_WORKERS="${NUM_COMPILE_WORKERS:-4}"
NUM_GPUS="${NUM_GPUS:-2}"

# 检查 my-kernel-bench
if [ ! -d "$KERNELBENCH_DIR/data" ]; then
    echo "ERROR: my-kernel-bench not found at $KERNELBENCH_DIR"
    echo "Please clone it or set KERNELBENCH_DIR"
    exit 1
fi

# 构建 levels 参数
LEVEL_ARGS=""
for level in $LEVELS; do
    LEVEL_ARGS="$LEVEL_ARGS $level"
done

echo "============================================"
echo "  Compile + Verify Evaluation"
echo "============================================"
echo "Model:        $MODEL_PATH"
echo "Run name:     $RUN_NAME"
echo "Levels:       $LEVELS"
echo "KernelBench:  $KERNELBENCH_DIR"
echo "Results:      $RESULTS_DIR/$RUN_NAME"
echo "============================================"

# ======== 阶段 A: 生成 ========
if [ "$MODEL_PATH" = "SKIP" ]; then
    echo ""
    echo "=== Phase A: SKIPPED (eval_only mode) ==="
else
    echo ""
    echo "=== Phase A: Generation (vLLM) ==="

    # 启动 vLLM 服务
    echo "Starting vLLM server on port $VLLM_PORT..."
    python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_PATH" \
        --tensor-parallel-size "$NUM_GPUS" \
        --gpu-memory-utilization 0.8 \
        --max-model-len 10240 \
        --port "$VLLM_PORT" &
    VLLM_PID=$!

    # 确保退出时清理 vLLM
    cleanup_vllm() {
        echo "Cleaning up vLLM server (PID: $VLLM_PID)..."
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    }
    trap cleanup_vllm EXIT

    # 等待 vLLM 就绪
    echo "Waiting for vLLM server..."
    for i in $(seq 1 120); do
        if curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
            echo "vLLM server ready! (${i}s)"
            break
        fi
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "ERROR: vLLM server process died"
            exit 1
        fi
        if [ "$i" -eq 120 ]; then
            echo "ERROR: vLLM server failed to start after 120s"
            exit 1
        fi
        sleep 1
    done

    # 生成
    python "${PROJECT_DIR}/scripts/eval/eval_with_compile.py" \
        --model_path "$MODEL_PATH" \
        --run_name "$RUN_NAME" \
        --levels $LEVEL_ARGS \
        --kernelbench_dir "$KERNELBENCH_DIR" \
        --output_dir "$RESULTS_DIR" \
        --api_base "http://localhost:${VLLM_PORT}/v1" \
        --model_name "$MODEL_PATH" \
        --max_tokens "$MAX_TOKENS" \
        --generate_only

    # 关闭 vLLM（释放 GPU 给阶段 B）
    echo "Shutting down vLLM server..."
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
    trap - EXIT  # 移除 cleanup trap
    sleep 5  # 等待 GPU 内存释放

    echo "Phase A complete."
fi

# ======== 阶段 B: 编译验证 ========
echo ""
echo "=== Phase B: Compile + Verify ==="

python "${PROJECT_DIR}/scripts/eval/eval_with_compile.py" \
    --run_name "$RUN_NAME" \
    --levels $LEVEL_ARGS \
    --kernelbench_dir "$KERNELBENCH_DIR" \
    --output_dir "$RESULTS_DIR" \
    --num_workers "$NUM_COMPILE_WORKERS" \
    --num_gpus "$NUM_GPUS" \
    --eval_only

echo ""
echo "============================================"
echo "  Evaluation Complete"
echo "============================================"
echo "Results: $RESULTS_DIR/$RUN_NAME/"
echo ""
echo "Files:"
for level in $LEVELS; do
    result_file="$RESULTS_DIR/$RUN_NAME/level${level}.json"
    if [ -f "$result_file" ]; then
        echo "  level${level}.json ✓"
    else
        echo "  level${level}.json (not found)"
    fi
done
echo "  compile_eval_metrics.json"
echo ""
