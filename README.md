# kernel-rl

使用强化学习提升 LLM 的 GPU Kernel 生成能力。

## 概述

本项目基于 [KernelBook](https://huggingface.co/datasets/GPUMODE/KernelBook) 数据集和 [verl](https://github.com/volcengine/verl) 强化学习框架，通过 SFT + RL 两阶段训练提升开源 7B 模型的 Triton kernel 生成能力，并在 [KernelBench](https://github.com/ScalingIntelligence/KernelBench) 上评测。

## 实验设计

### 训练策略

```
Phase 1: SFT (Supervised Fine-Tuning)
  └── 在 KernelBook 上学习 PyTorch → Triton 转换

Phase 2: RL (Reinforcement Learning)
  ├── GRPO (Group Relative Policy Optimization)
  ├── REINFORCE++
  └── RLOO (REINFORCE Leave-One-Out)
```

### 对比实验

| 模型 | 说明 |
|------|------|
| Qwen2.5-Coder-7B-Instruct | Baseline (zero-shot) |
| + SFT | SFT on KernelBook |
| + SFT + GRPO | SFT → GRPO RL |
| + SFT + REINFORCE++ | SFT → REINFORCE++ RL |
| + SFT + RLOO | SFT → RLOO RL |

### 硬件要求

- 2× NVIDIA A100 80GB

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 数据准备

```bash
# 下载 KernelBook 数据集
python scripts/data/download_kernelbook.py

# 清洗数据
python scripts/data/clean_kernelbook.py \
    --input data/raw/kernelbook_raw.parquet \
    --output data/cleaned/kernelbook_clean.parquet

# 准备 SFT 格式数据
python scripts/data/prepare_sft_data.py \
    --input data/cleaned/kernelbook_clean.parquet \
    --output_dir data/sft

# 准备 RL 格式数据
python scripts/data/prepare_rl_data.py \
    --input data/cleaned/kernelbook_clean.parquet \
    --output_dir data/rl
```

### 3. SFT 训练

```bash
bash scripts/train/run_sft.sh
```

### 4. 合并 LoRA 权重

```bash
python scripts/train/merge_lora.py \
    --lora_path checkpoints/sft/epoch_3 \
    --output_dir checkpoints/sft_merged
```

### 5. RL 训练

```bash
# 选择一种或多种 RL 算法
bash scripts/train/run_grpo.sh
bash scripts/train/run_reinforce_pp.sh
bash scripts/train/run_rloo.sh
```

### 6. 评测

```bash
# 评测各个 checkpoint
bash scripts/eval/run_kernelbench.sh Qwen/Qwen2.5-Coder-7B-Instruct baseline
bash scripts/eval/run_kernelbench.sh checkpoints/sft_merged sft-only
bash scripts/eval/run_kernelbench.sh checkpoints/grpo/epoch_40 grpo

# 对比结果
python scripts/eval/compare_results.py --results_dir results/
```

## 项目结构

```
kernel-rl/
├── configs/                    # 配置文件
├── scripts/
│   ├── data/                   # 数据处理脚本
│   │   ├── download_kernelbook.py
│   │   ├── clean_kernelbook.py
│   │   ├── prepare_sft_data.py
│   │   └── prepare_rl_data.py
│   ├── train/                  # 训练脚本
│   │   ├── run_sft.sh
│   │   ├── merge_lora.py
│   │   ├── run_grpo.sh
│   │   ├── run_reinforce_pp.sh
│   │   └── run_rloo.sh
│   └── eval/                   # 评测脚本
│       ├── run_kernelbench.sh
│       ├── eval_kernelbench.py
│       └── compare_results.py
├── src/
│   ├── reward/
│   │   └── kernel_reward.py    # Reward 函数
│   └── utils/
│       ├── dedup.py            # 去重工具
│       └── tokenizer_utils.py  # Token 计算工具
├── data/                       # 数据目录（不入 git）
├── checkpoints/                # 模型检查点（不入 git）
└── results/                    # 评测结果
```

## Reward 函数设计

### 训练时（静态分析，快速）

| 等级 | 分数 | 条件 |
|------|------|------|
| R0 | 0.0 | 无代码块 |
| R1 | 0.1 | 有代码但无函数/类定义 |
| R2 | 0.2 | 有定义但语法错误 |
| R3 | 0.4 | 语法正确但无 Triton kernel |
| R4 | 0.6 | 有 @triton.jit 但缺 wrapper |
| R5 | 0.8 | 完整 kernel + wrapper |
| +0.1 | | 包含性能优化（autotune 等） |
| +0.1 | | 非简单复制 |

### 评测时（编译+执行，准确）

- 编译成功 → 0.6
- 正确性验证 → 1.0

## 参考

- [KernelBook](https://huggingface.co/datasets/GPUMODE/KernelBook) - GPUMODE PyTorch→Triton 数据集
- [KernelBench](https://github.com/ScalingIntelligence/KernelBench) - Stanford GPU kernel 评测基准
- [verl](https://github.com/volcengine/verl) - RL 训练框架
- [Kevin-32B](https://arxiv.org/abs/2507.11948) - GRPO 多轮 RL 训练 kernel 生成
- [CUDA-L1](https://arxiv.org/abs/2507.14111) - 对比 RL kernel 优化
