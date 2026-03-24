# Kernel-RL 实验记录

## Exp-1: SFT Baseline (CityU HPC)

| 项目 | 值 |
|------|-----|
| 模型 | Qwen2.5-Coder-7B-Instruct |
| 方法 | SFT (LoRA rank=64, alpha=16, target=all-linear) |
| 数据 | KernelBook 清洗后 → ModelNew 格式, ~10k train / ~3k val |
| 硬件 | CityU HPC, 3× A100 40GB, 单节点 |
| Checkpoint | `checkpoints/sft_modelnew_merged/` |

### KernelBench 静态评测结果 (2026-03-23)

| 指标 | Level 1 | Level 2 |
|------|---------|---------|
| 总题数 | 100 | 100 |
| 平均分 | 0.965 | 0.922 |
| 完整结构 (≥0.8) | 96% | 91% |
| 有 Triton kernel (≥0.4) | 96% | 92% |
| 语法错误 (=0.2) | 4 | 8 |

注意：此为静态分析评分，仅检查代码结构完整度（有 triton import、jit decorator、wrapper 等），不验证编译/运行正确性。

---

## Exp-2: GRPO 3B 方法验证 (PolyU HPC)

**目的**：用小模型快速验证 RL 方法有效性

| 项目 | 值 |
|------|-----|
| 模型 | Qwen2.5-Coder-3B-Instruct (base, 无 SFT) |
| 方法 | GRPO, full fine-tuning (无 LoRA) |
| 数据 | KernelBook split RL data (ModelNew 格式) |
| Reward | compute_score_auto (静态分析) |
| 硬件 | PolyU HPC, 2× A100 80GB, 单节点 |
| Rollout | vLLM, TP=1, gpu_memory_utilization=0.5 |
| Checkpoint | `checkpoints/grpo_3b/` |

### 超参数

| 参数 | 值 |
|------|-----|
| lr | 1e-6 |
| train_batch_size | 8 |
| ppo_mini_batch_size | 8 |
| ppo_micro_batch_size_per_gpu | 2 |
| rollout.n | 5 |
| max_prompt_length | 2048 |
| max_response_length | 4096 |
| gradient_checkpointing | true |
| optimizer_offload | false |
| param_offload | false |
| total_epochs | 3 |
| save_freq | 200 |

### 状态
- [ ] 训练中 (启动于 2026-03-24)
- [ ] KernelBench 评测

---

## Exp-3: GRPO 7B (CityU HPC)

**目的**：7B 模型正式 RL 训练

| 项目 | 值 |
|------|-----|
| 模型 | Qwen2.5-Coder-7B-Instruct → SFT checkpoint (`sft_modelnew_merged`) |
| 方法 | GRPO, LoRA rank=64, alpha=16, target=all-linear |
| 数据 | KernelBook split RL data (ModelNew 格式) |
| Reward | compute_score_auto (静态分析) |
| 硬件 | CityU HPC, 2 节点 × 3× A100 40GB = 6 GPU |
| Rollout | SGLang, flashinfer, gpu_memory_utilization=0.8 |
| Checkpoint | `checkpoints/grpo_cuda/` |

### 超参数

| 参数 | 值 |
|------|-----|
| lr | 5e-5 |
| train_batch_size | 动态 (TOTAL_GPUS × 4) |
| ppo_mini_batch_size | = train_batch_size |
| ppo_micro_batch_size_per_gpu | 1 |
| rollout.n | 5 |
| max_prompt_length | 2048 |
| max_response_length | 4096 |
| gradient_checkpointing | true |
| optimizer_offload | false |
| param_offload | false |
| total_epochs | 3 |
| save_freq | 50 |

### 状态
- [ ] 等待 SLURM 资源分配
- [ ] KernelBench 评测

---

## 评测计划

每个实验的 checkpoint 在 KernelBench Level 1-2 上评测：
1. **Baseline**: Qwen2.5-Coder-7B-Instruct (原始模型)
2. **SFT**: sft_modelnew_merged (Exp-1)
3. **GRPO 3B**: grpo_3b best checkpoint (Exp-2)
4. **GRPO 7B**: grpo_cuda best checkpoint (Exp-3)

指标：静态分析 avg_score + has_complete_pct (≥0.8)

## Google Drive 备份

路径：`gdrive:kernel-rl/checkpoints/`

| Checkpoint | 来源 | 上传状态 |
|-----------|------|---------|
| sft_modelnew_merged | CityU | [ ] |
| grpo_3b | PolyU | [ ] |
| grpo_cuda | CityU | [ ] |
