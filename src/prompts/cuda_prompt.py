"""
集中管理的 Prompt 模板。

包含：
- CUDA_PROMPT_TEMPLATE: CUDA load_inline 后端（新默认）
- TRITON_PROMPT_TEMPLATE: Triton 后端（向后兼容）
- TRITON_ORIGINAL_PROMPT_TEMPLATE: Triton 原始格式（KernelBook 格式）
"""


# ============================================================
# CUDA load_inline Prompt（新默认）
# ============================================================

CUDA_PROMPT_TEMPLATE = """You are an expert CUDA programmer. Your task is to optimize the given PyTorch `Model` by replacing its standard operators with custom CUDA kernels using `torch.utils.cpp_extension.load_inline`.

### RULES
1. Name your optimized class `ModelNew`, inheriting from `nn.Module`.
2. Preserve the `__init__` signature and all `nn.Module` sub-layers so that `state_dict` remains compatible with the original `Model`.
3. In `forward`, access underlying parameters (e.g., `self.linear.weight`) and pass them to your custom CUDA kernel. Do NOT call module objects directly for the operations you are replacing.
4. **Memory operations are allowed** in standard PyTorch: `reshape`, `view`, `contiguous`, `permute`, `transpose`, `indexing`, `cat`, `stack`, `split`. Only the **compute-heavy** operations (matmul, conv, activation, normalization, pooling, reduction, elementwise math) must use CUDA kernels.
5. Use `torch.utils.cpp_extension.load_inline` with three components:
   - `cpp_sources`: C++ declarations (function signatures exposed to Python)
   - `cuda_sources`: CUDA implementation (kernels + launch wrappers)
   - `functions`: list of function names to export
6. Each CUDA kernel must handle **contiguous** tensors. Call `.contiguous()` on inputs before passing to the kernel.
7. Use `AT_DISPATCH_FLOATING_TYPES_AND_HALF` in CUDA code to support multiple dtypes.
8. Give the `load_inline` call a descriptive `name` argument (e.g., `name="fused_gelu"`).
9. Output the **complete, runnable** Python code inside a single code block. Include all imports.

### EXAMPLE (fused bias + ReLU)
```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

fused_bias_relu_source = \"\"\"
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_bias_relu_kernel(const float* __restrict__ input,
                                        const float* __restrict__ bias,
                                        float* __restrict__ output,
                                        int N, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * C) {
        int c = idx % C;
        float val = input[idx] + bias[c];
        output[idx] = val > 0.0f ? val : 0.0f;
    }
}

torch::Tensor fused_bias_relu_cuda(torch::Tensor input, torch::Tensor bias) {
    auto output = torch::empty_like(input);
    int total = input.numel();
    int C = bias.size(0);
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    fused_bias_relu_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), total / C, C);
    return output;
}
\"\"\"

fused_bias_relu_cpp = "torch::Tensor fused_bias_relu_cuda(torch::Tensor input, torch::Tensor bias);"

fused_bias_relu = load_inline(
    name="fused_bias_relu",
    cpp_sources=fused_bias_relu_cpp,
    cuda_sources=fused_bias_relu_source,
    functions=["fused_bias_relu_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = x.contiguous()
        weight = self.linear.weight
        bias = self.linear.bias
        out = torch.nn.functional.linear(x, weight)  # or your custom matmul kernel
        out = fused_bias_relu.fused_bias_relu_cuda(out, bias)
        return out
```

### Input Architecture
```python
{model_code}
```

### Optimized CUDA Implementation:"""


# ============================================================
# Triton ModelNew Prompt（向后兼容）
# ============================================================

TRITON_PROMPT_TEMPLATE = """You are an expert Performance Engineer specializing in Triton and PyTorch internals.

### TASK
Optimize the provided architecture named `Model` by replacing standard PyTorch operators with custom Triton kernels.

### RULES
1. Name the optimized output architecture `ModelNew`.
2. Preserve `__init__` structure (nn.Module definitions) for state_dict compatibility.
3. In `forward`, access underlying parameters (e.g., self.conv.weight) and pass them to your custom Triton kernels. Do NOT call module objects directly.
4. Use @triton.jit for kernel functions, include proper wrapper functions.
5. Generate REAL, compilable code with all imports. Output ONLY the code block.

### Input Architecture
```python
{model_code}
```

### Optimized Triton Implementation:"""


# ============================================================
# Triton 原始格式 Prompt（KernelBook 格式，向后兼容）
# ============================================================

TRITON_ORIGINAL_PROMPT_TEMPLATE = """You are an expert GPU programmer specializing in Triton kernel development. Your task is to convert the following PyTorch code into an optimized Triton kernel implementation.

## Requirements:
1. The Triton kernel must be functionally equivalent to the PyTorch code
2. Use @triton.jit decorator for kernel functions
3. Include a Python wrapper function that calls the Triton kernel
4. Optimize for GPU performance (memory coalescing, optimal block sizing, etc.)
5. Include proper imports (triton, triton.language as tl, torch, etc.)

## PyTorch Code:
```python
{python_code}
```

## Optimized Triton Implementation:"""
