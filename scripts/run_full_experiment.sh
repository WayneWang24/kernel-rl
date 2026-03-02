#!/bin/bash
# ============================================================
# 端到端实验管线
#
# 完整流程：
#   Step 1: 数据准备（SFT ModelNew + RL KernelBench）
#   Step 2: Baseline 评测（Qwen2.5-Coder-7B zero-shot）
#   Step 3: SFT 训练（ModelNew 格式）
#   Step 4: LoRA 合并
#   Step 5: SFT 评测
#   Step 6: GRPO 训练（KernelBench 任务）
#   Step 7: GRPO 评测
#   Step 8: 结果对比
#
# 用法：
#   bash scripts/run_full_experiment.sh [--from STEP] [--to STEP]
#
# 示例：
#   # 运行全部
#   bash scripts/run_full_experiment.sh
#
#   # 从 step 3 开始（已有数据 + baseline）
#   bash scripts/run_full_experiment.sh --from 3
#
#   # 只运行 step 1-2（数据准备 + baseline）
#   bash scripts/run_full_experiment.sh --to 2
#
# 硬件：2× A100 80GB
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# 默认参数
FROM_STEP=1
TO_STEP=8
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}"
KERNELBENCH_DIR="${KERNELBENCH_DIR:-$HOME/Code/my-kernel-bench}"
EVAL_LEVELS="${EVAL_LEVELS:-1 2 3}"

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --from) FROM_STEP="$2"; shift 2 ;;
        --to) TO_STEP="$2"; shift 2 ;;
        --model) BASE_MODEL="$2"; shift 2 ;;
        --kernelbench) KERNELBENCH_DIR="$2"; shift 2 ;;
        --levels) EVAL_LEVELS="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "============================================"
echo "  kernel-rl Full Experiment Pipeline"
echo "============================================"
echo "Base model:   $BASE_MODEL"
echo "KernelBench:  $KERNELBENCH_DIR"
echo "Eval levels:  $EVAL_LEVELS"
echo "Steps:        $FROM_STEP → $TO_STEP"
echo "============================================"
echo ""

run_step() {
    local step=$1
    local name=$2
    if [ "$step" -ge "$FROM_STEP" ] && [ "$step" -le "$TO_STEP" ]; then
        echo ""
        echo "╔══════════════════════════════════════════╗"
        echo "║  Step $step: $name"
        echo "╚══════════════════════════════════════════╝"
        echo ""
        return 0
    else
        echo "  [SKIP] Step $step: $name"
        return 1
    fi
}

# ======== Step 1: 数据准备 ========
if run_step 1 "Data Preparation"; then
    echo "--- 1a: Prepare SFT data (ModelNew format) ---"
    python "${PROJECT_DIR}/scripts/data/prepare_sft_modelnew.py" \
        --input "${PROJECT_DIR}/data/cleaned/kernelbook_clean.parquet" \
        --output_dir "${PROJECT_DIR}/data/sft_modelnew"

    echo ""
    echo "--- 1b: Prepare RL data (KernelBench tasks) ---"
    python "${PROJECT_DIR}/scripts/data/prepare_rl_kernelbench.py" \
        --kernelbench_dir "$KERNELBENCH_DIR" \
        --output_dir "${PROJECT_DIR}/data/rl_kernelbench"

    echo ""
    echo "Step 1 complete. Data ready at:"
    echo "  SFT: data/sft_modelnew/{train,val,test}.parquet"
    echo "  RL:  data/rl_kernelbench/{train,val,test}.parquet"
fi

# ======== Step 2: Baseline 评测 ========
if run_step 2 "Baseline Evaluation"; then
    bash "${PROJECT_DIR}/scripts/eval/run_eval_compile.sh" \
        "$BASE_MODEL" \
        "baseline" \
        "$EVAL_LEVELS"

    echo "Step 2 complete. Baseline results at: results/baseline/"
fi

# ======== Step 3: SFT 训练 ========
if run_step 3 "SFT Training (ModelNew)"; then
    # 使用 ModelNew 格式数据
    TRAIN_PATH="${PROJECT_DIR}/data/sft_modelnew/train.parquet"
    VAL_PATH="${PROJECT_DIR}/data/sft_modelnew/val.parquet"

    if [ ! -f "$TRAIN_PATH" ]; then
        echo "ERROR: SFT data not found. Run step 1 first."
        exit 1
    fi

    torchrun --standalone --nnodes=1 --nproc_per_node=2 \
        -m verl.trainer.fsdp_sft_trainer \
        data.train_files="$TRAIN_PATH" \
        data.val_files="$VAL_PATH" \
        data.train_batch_size=32 \
        data.micro_batch_size_per_gpu=2 \
        data.max_length=6144 \
        data.truncation=left \
        data.multiturn.enable=true \
        data.multiturn.messages_key=messages \
        model.partial_pretrain="$BASE_MODEL" \
        model.enable_gradient_checkpointing=true \
        model.lora_rank=64 \
        model.lora_alpha=128 \
        model.target_modules=all-linear \
        model.use_liger=false \
        trainer.total_epochs=3 \
        trainer.default_local_dir="${PROJECT_DIR}/checkpoints/sft_modelnew" \
        trainer.save_freq=1 \
        trainer.test_freq=1 \
        trainer.project_name=kernel_sft \
        trainer.experiment_name=qwen25_coder_7b_modelnew

    echo "Step 3 complete. SFT checkpoints at: checkpoints/sft_modelnew/"
fi

# ======== Step 4: LoRA 合并 ========
if run_step 4 "Merge LoRA Weights"; then
    SFT_CKPT="${PROJECT_DIR}/checkpoints/sft_modelnew/epoch_3"
    if [ ! -d "$SFT_CKPT" ]; then
        # 尝试找到最新的 epoch
        SFT_CKPT=$(ls -d "${PROJECT_DIR}/checkpoints/sft_modelnew/epoch_"* 2>/dev/null | sort -t_ -k2 -n | tail -1)
    fi

    if [ -z "$SFT_CKPT" ] || [ ! -d "$SFT_CKPT" ]; then
        echo "ERROR: SFT checkpoint not found. Run step 3 first."
        exit 1
    fi

    echo "Merging LoRA from: $SFT_CKPT"
    python "${PROJECT_DIR}/scripts/train/merge_lora.py" \
        --base_model "$BASE_MODEL" \
        --lora_path "$SFT_CKPT" \
        --output_dir "${PROJECT_DIR}/checkpoints/sft_modelnew_merged"

    echo "Step 4 complete. Merged model at: checkpoints/sft_modelnew_merged/"
fi

# ======== Step 5: SFT 评测 ========
if run_step 5 "SFT Evaluation"; then
    SFT_MERGED="${PROJECT_DIR}/checkpoints/sft_modelnew_merged"
    if [ ! -d "$SFT_MERGED" ]; then
        echo "ERROR: Merged SFT model not found. Run step 4 first."
        exit 1
    fi

    bash "${PROJECT_DIR}/scripts/eval/run_eval_compile.sh" \
        "$SFT_MERGED" \
        "sft-modelnew" \
        "$EVAL_LEVELS"

    echo "Step 5 complete. SFT results at: results/sft-modelnew/"
fi

# ======== Step 6: GRPO 训练 ========
if run_step 6 "GRPO Training (KernelBench)"; then
    bash "${PROJECT_DIR}/scripts/train/run_grpo.sh"

    echo "Step 6 complete. GRPO checkpoints at: checkpoints/grpo/"
fi

# ======== Step 7: GRPO 评测 ========
if run_step 7 "GRPO Evaluation"; then
    GRPO_CKPT="${PROJECT_DIR}/checkpoints/grpo/epoch_20"
    if [ ! -d "$GRPO_CKPT" ]; then
        # 尝试找到最新的 epoch
        GRPO_CKPT=$(ls -d "${PROJECT_DIR}/checkpoints/grpo/epoch_"* 2>/dev/null | sort -t_ -k2 -n | tail -1)
    fi

    if [ -z "$GRPO_CKPT" ] || [ ! -d "$GRPO_CKPT" ]; then
        echo "ERROR: GRPO checkpoint not found. Run step 6 first."
        exit 1
    fi

    echo "Evaluating GRPO checkpoint: $GRPO_CKPT"
    bash "${PROJECT_DIR}/scripts/eval/run_eval_compile.sh" \
        "$GRPO_CKPT" \
        "grpo-modelnew" \
        "$EVAL_LEVELS"

    echo "Step 7 complete. GRPO results at: results/grpo-modelnew/"
fi

# ======== Step 8: 结果对比 ========
if run_step 8 "Results Comparison"; then
    echo "=== Compile + Verify Results ==="
    echo ""

    for run_name in baseline sft-modelnew grpo-modelnew; do
        metrics_file="${PROJECT_DIR}/results/${run_name}/compile_eval_metrics.json"
        if [ -f "$metrics_file" ]; then
            echo "--- $run_name ---"
            python3 -c "
import json
with open('$metrics_file') as f:
    m = json.load(f)
for level, metrics in sorted(m.items()):
    print(f'  {level}: compile={metrics[\"compile_pass_rate\"]}, verify={metrics[\"verify_pass_rate\"]}')
"
            echo ""
        else
            echo "--- $run_name --- (no results found)"
            echo ""
        fi
    done

    # 也运行原始的 compare_results.py（如果有静态评测结果）
    if ls "${PROJECT_DIR}/results"/*/level*_results.json > /dev/null 2>&1; then
        echo "=== Static Analysis Results ==="
        python "${PROJECT_DIR}/scripts/eval/compare_results.py" \
            --results_dir "${PROJECT_DIR}/results/" 2>/dev/null || true
    fi

    echo ""
    echo "Step 8 complete."
fi

echo ""
echo "============================================"
echo "  Experiment Pipeline Complete!"
echo "============================================"
echo "Results directory: ${PROJECT_DIR}/results/"
echo ""
