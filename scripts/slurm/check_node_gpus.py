#!/usr/bin/env python3
"""Check working GPUs on current node, write results to shared directory.

Usage:
    python3 check_node_gpus.py <output_dir> <node_name>

Writes comma-separated working GPU IDs to <output_dir>/<node_name>.txt
"""
import os
import subprocess
import sys


def main():
    output_dir = sys.argv[1]
    node_name = sys.argv[2] if len(sys.argv) > 2 else os.uname()[1]

    total = int(os.environ.get("SLURM_GPUS_PER_NODE", "0") or "0")
    if total == 0:
        try:
            import torch
            total = torch.cuda.device_count()
        except Exception:
            total = 3

    working = []
    for i in range(total):
        try:
            r = subprocess.run(
                [sys.executable, "-c",
                 f'import os; os.environ["CUDA_VISIBLE_DEVICES"]="{i}"; '
                 f"import torch; "
                 f"assert torch.cuda.is_available() and torch.cuda.device_count()>0"],
                capture_output=True, timeout=30, text=True,
            )
            if r.returncode == 0:
                working.append(str(i))
                print(f"  GPU {i}: OK")
            else:
                err = r.stderr.strip().split("\n")[-1] if r.stderr.strip() else "unknown"
                print(f"  GPU {i}: FAILED - {err}")
        except subprocess.TimeoutExpired:
            print(f"  GPU {i}: TIMEOUT")

    os.makedirs(output_dir, exist_ok=True)
    result_file = os.path.join(output_dir, f"{node_name}.txt")
    with open(result_file, "w") as f:
        f.write(",".join(working))

    print(f"  {node_name}: {len(working)}/{total} GPUs working")


if __name__ == "__main__":
    main()
