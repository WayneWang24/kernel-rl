#!/usr/bin/env python3
"""
GRPO 训练启动器（Python 版，替代 bash 脚本避免行续接问题）。

用法：
    python scripts/train/launch_grpo.py [额外 Hydra override ...]

硬件：2× A100 80GB
"""
import os
import sys
import subprocess

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# === 数据路径（自动探测） ===
if os.path.isfile(f"{PROJECT_DIR}/data/split/rl/train.parquet"):
    train_path = f"{PROJECT_DIR}/data/split/rl/train.parquet"
    val_path = f"{PROJECT_DIR}/data/split/rl/val.parquet"
    reward_fn_name = "compute_score_auto"
    print("Using KernelBook split RL data (ModelNew format)")
elif os.path.isfile(f"{PROJECT_DIR}/data/rl_kernelbench/train.parquet"):
    train_path = f"{PROJECT_DIR}/data/rl_kernelbench/train.parquet"
    val_path = f"{PROJECT_DIR}/data/rl_kernelbench/val.parquet"
    reward_fn_name = "compute_score_auto"
    print("Using KernelBench RL data (ModelNew format)")
else:
    train_path = f"{PROJECT_DIR}/data/rl/train.parquet"
    val_path = f"{PROJECT_DIR}/data/rl/val.parquet"
    reward_fn_name = "compute_score"
    print("Using KernelBook RL data (original format)")

# === 模型路径（自动探测） ===
model_path = None
for ckpt in ["checkpoints/sft_modelnew_merged", "checkpoints/sft_merged"]:
    full = os.path.join(PROJECT_DIR, ckpt)
    if os.path.isdir(full):
        model_path = full
        print(f"Using SFT checkpoint: {model_path}")
        break
if model_path is None:
    model_path = os.environ.get("MODEL_PATH", "Qwen/Qwen2.5-Coder-7B-Instruct")
    print(f"WARNING: SFT checkpoint not found, using base model: {model_path}")

# === Reward 函数 ===
reward_fn_path = f"{PROJECT_DIR}/src/reward/kernel_reward.py"

# === 检查数据 ===
if not os.path.isfile(train_path):
    print(f"ERROR: RL training data not found at {train_path}")
    sys.exit(1)

# === 构建 Hydra overrides ===
overrides = [
    "algorithm.adv_estimator=grpo",
    f"data.train_files={train_path}",
    f"data.val_files={val_path}",
    "data.train_batch_size=16",
    "data.max_prompt_length=4096",
    "data.max_response_length=8192",
    "data.filter_overlong_prompts=true",
    "data.truncation=error",
    f"actor_rollout_ref.model.path={model_path}",
    "actor_rollout_ref.actor.optim.lr=5e-7",
    "actor_rollout_ref.model.use_remove_padding=true",
    "actor_rollout_ref.actor.ppo_mini_batch_size=16",
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4",
    "actor_rollout_ref.actor.use_kl_loss=true",
    "actor_rollout_ref.actor.kl_loss_coef=0.005",
    "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
    "actor_rollout_ref.actor.entropy_coeff=0",
    "actor_rollout_ref.model.enable_gradient_checkpointing=true",
    "actor_rollout_ref.actor.fsdp_config.param_offload=false",
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=false",
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4",
    "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
    "actor_rollout_ref.rollout.name=vllm",
    "actor_rollout_ref.rollout.gpu_memory_utilization=0.35",
    "actor_rollout_ref.rollout.n=5",
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4",
    "actor_rollout_ref.ref.fsdp_config.param_offload=true",
    "algorithm.use_kl_in_reward=false",
    f"+reward.custom_reward_function.path={reward_fn_path}",
    f"+reward.custom_reward_function.name={reward_fn_name}",
    "trainer.critic_warmup=0",
    'trainer.logger=["console"]',
    "trainer.project_name=kernel_rl",
    "trainer.experiment_name=grpo_qwen25_coder_7b_modelnew",
    f"trainer.default_local_dir={PROJECT_DIR}/checkpoints/grpo",
    "trainer.n_gpus_per_node=2",
    "trainer.nnodes=1",
    "trainer.save_freq=10",
    "trainer.test_freq=5",
    "trainer.total_epochs=20",
]

# 追加用户额外参数
overrides.extend(sys.argv[1:])

# === 执行 ===
cmd = [sys.executable, "-m", "verl.trainer.main_ppo"] + overrides
print(f"\n=== Launching GRPO Training ===")
print(f"Overrides: {len(overrides)} params")
print(f"Command: {cmd[0]} -m verl.trainer.main_ppo [...]")
print()

os.execvp(cmd[0], cmd)
