# Data Split Redesign: SFT 学格式 + RL 学质量

**Date**: 2026-03-03
**Status**: Approved

## Goal

Redesign the data pipeline to:
1. Use full KernelBook 18K data (no cleaning)
2. Split into SFT (65%) and RL (35%) with complementary characteristics
3. Both use ModelNew format
4. Evaluate on KernelBench Level 1-4 (completely held-out)

## Data Split Strategy (Scheme C)

### Input
- KernelBook raw: 18,162 records
- Only filter: remove rows with empty python_code or triton_code
- No dedup, no syntax filter, no length filter

### Feature Computation (per record)
```python
complexity_score = normalize(python_code_tokens) + num_triton_kernels * 0.3 + has_nn_module * 0.2
quality_score = has_autotune * 0.3 + has_block_size * 0.2 + has_wrapper * 0.3 + is_non_trivial * 0.2
modelnew_convertible = try_convert_to_modelnew() is not None
```

### Split Rules (by repo group)
- Group by `repo_name` to prevent data leakage

**SFT Pool (~65%, ~11,800)**:
- All ModelNew-convertible samples prioritized
- High-stars repos prioritized
- Cover full complexity range
- Purpose: Learn ModelNew format + basic Triton writing

**RL Pool (~35%, ~6,360)**:
- Higher complexity_score samples prioritized
- Higher quality_score samples prioritized (autotune, optimization)
- Still use ModelNew format prompts
- Purpose: Learn code quality and optimization via reward signal

### Format

| | SFT | RL |
|---|---|---|
| Prompt | ModelNew template (Model → ModelNew) | Same |
| Response | Reference Triton code (teacher) | Self-generated |
| Loss/Reward | Cross-entropy | compute_score_modelnew |

### Evaluation (Completely Held-Out)
- KernelBench Level 1: 100 tasks (single op)
- KernelBench Level 2: 100 tasks (combined ops)
- KernelBench Level 3: 50 tasks (small networks)
- KernelBench Level 4: 20 tasks (full models)

## File Changes

| File | Action | Description |
|------|--------|-------------|
| `scripts/data/prepare_split_data.py` | **NEW** | Unified SFT+RL split from raw KernelBook |
| `scripts/train/run_sft.sh` | MODIFY | Update data paths to data/split/sft/ |
| `scripts/train/run_grpo.sh` | MODIFY | Use KernelBook RL data + compute_score_modelnew |
| `scripts/eval/run_eval_compile.sh` | MODIFY | Default eval levels 1-4 |
| `scripts/run_full_experiment.sh` | MODIFY | Update pipeline |
| `src/reward/kernel_reward.py` | MODIFY | compute_score_modelnew handle string ground_truth |

## Training Pipeline

```
Step 1: prepare_split_data.py → data/split/{sft,rl}/{train,val}.parquet
Step 2: Baseline eval (Qwen2.5-Coder-7B zero-shot) → Level 1-4
Step 3: SFT training (3 epoch, LoRA rank=64, ~11.8K samples)
Step 4: LoRA merge
Step 5: SFT eval → Level 1-4
Step 6: GRPO training (20 epoch, ~6.4K samples, compute_score_modelnew)
Step 7: GRPO eval → Level 1-4
Step 8: Compare Baseline vs SFT vs GRPO
```
