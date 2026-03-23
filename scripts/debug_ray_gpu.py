#!/usr/bin/env python3
"""诊断 Ray worker 内的 GPU 可见性。在 srun 交互节点上运行。"""
import os
import subprocess
import ray

def check_gpu_env():
    """在当前进程中检查 GPU 环境。"""
    info = {}
    info["pid"] = os.getpid()
    info["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
    info["SLURM_GPUS_PER_NODE"] = os.environ.get("SLURM_GPUS_PER_NODE", "<unset>")
    info["SLURM_JOB_GPUS"] = os.environ.get("SLURM_JOB_GPUS", "<unset>")
    info["SLURM_STEP_GPUS"] = os.environ.get("SLURM_STEP_GPUS", "<unset>")

    # /dev/nvidia* 检查
    nvidia_devs = [f for f in os.listdir("/dev/") if f.startswith("nvidia")] if os.path.exists("/dev/") else []
    info["dev_nvidia_files"] = nvidia_devs[:10]

    # nvidia-smi
    for path in ["nvidia-smi", "/usr/bin/nvidia-smi"]:
        try:
            r = subprocess.run([path], capture_output=True, timeout=10, text=True)
            info[f"nvidia-smi({path})"] = f"rc={r.returncode}, first_line={r.stdout.split(chr(10))[0][:80]}"
        except Exception as e:
            info[f"nvidia-smi({path})"] = f"FAIL: {e}"

    # torch.cuda
    import torch
    info["torch.cuda.is_available()"] = torch.cuda.is_available()
    info["torch.cuda.device_count()"] = torch.cuda.device_count() if torch.cuda.is_available() else "N/A (not available)"
    info["torch.version.cuda"] = torch.version.cuda

    return info


@ray.remote
class GpuTestActor:
    """不请求 GPU 的 actor（模拟 TaskRunner）。"""
    def check(self):
        return {"role": "no-gpu actor", **check_gpu_env()}


@ray.remote(num_gpus=1)
class GpuWorkerActor:
    """请求 1 GPU 的 actor（模拟 WorkerDict）。"""
    def check(self):
        return {"role": "1-gpu actor", **check_gpu_env()}


if __name__ == "__main__":
    print("=" * 60)
    print("=== Main process GPU check ===")
    print("=" * 60)
    for k, v in check_gpu_env().items():
        print(f"  {k}: {v}")

    print("\n=== Starting Ray ===")
    ray.init()
    print(f"Ray resources: {ray.available_resources()}")

    print("\n=== No-GPU actor (simulates TaskRunner) ===")
    actor1 = GpuTestActor.remote()
    result1 = ray.get(actor1.check.remote())
    for k, v in result1.items():
        print(f"  {k}: {v}")

    print("\n=== 1-GPU actor (simulates WorkerDict) ===")
    try:
        actor2 = GpuWorkerActor.remote()
        result2 = ray.get(actor2.check.remote())
        for k, v in result2.items():
            print(f"  {k}: {v}")
    except Exception as e:
        print(f"  FAILED: {e}")

    ray.shutdown()
    print("\n=== Done ===")
