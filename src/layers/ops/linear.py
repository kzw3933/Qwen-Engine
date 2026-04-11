import torch
import torch.nn.functional as F

import triton
import triton.language as tl

from typing import Tuple, Optional

@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "STEP_K": 32, "GROUP_M": 8},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 128, "STEP_K": 32, "GROUP_M": 8},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "STEP_K": 32, "GROUP_M": 8},
            num_warps=4,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "STEP_K": 32, "GROUP_M": 8},
            num_warps=4,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "STEP_K": 32, "GROUP_M": 8},
            num_warps=4,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "STEP_K": 32, "GROUP_M": 8},
            num_warps=2,
            num_stages=5,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "STEP_K": 64, "GROUP_M": 8},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 128, "STEP_K": 64, "GROUP_M": 8},
            num_warps=8,
            num_stages=3,
        ),
    ],
    key=["n_cols", "hidden_size"],
)
@triton.jit
def _triton_linear_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    n_rows,
    n_cols,
    hidden_size,
    seq_len,
    x_stride_b, x_stride_s, x_stride_d,
    weight_stride_o, weight_stride_i,
    output_stride_b, output_stride_s, output_stride_o,
    bias_stride,
    use_bias: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STEP_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    
    num_pid_m = tl.cdiv(n_rows, BLOCK_M)
    num_pid_n = tl.cdiv(n_cols, BLOCK_N)
    
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    
    pid_m_start = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - pid_m_start, GROUP_M)
    
    pid_m = pid_m_start + (pid % num_pid_in_group) % group_size_m
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    row_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = row_offsets < n_rows
    batch_offsets = row_offsets // seq_len
    seq_offsets = row_offsets % seq_len
            
    col_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    col_mask = col_offsets < n_cols

    output_offsets = (
        batch_offsets[:, None] * output_stride_b
        + seq_offsets[:, None] * output_stride_s
        + col_offsets[None, :] * output_stride_o
    )
    output_mask = row_mask[:, None] & col_mask[None, :]

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    x_base_offsets = (
        batch_offsets[:, None] * x_stride_b
        + seq_offsets[:, None] * x_stride_s
    )

    for k_start in range(0, hidden_size, STEP_K):
        k_offsets = k_start + tl.arange(0, STEP_K)
        k_mask = k_offsets < hidden_size

        x_offsets = x_base_offsets + k_offsets[None, :] * x_stride_d
        x_mask = row_mask[:, None] & k_mask[None, :]

        weight_offsets = (
            col_offsets[None, :] * weight_stride_o
            + k_offsets[:, None] * weight_stride_i
        )
        weight_mask = k_mask[:, None] & col_mask[None, :]

        block_x = tl.load(x_ptr + x_offsets, mask=x_mask, other=0.0)
        block_weight = tl.load(weight_ptr + weight_offsets, mask=weight_mask, other=0.0)

        accumulator = tl.dot(block_x, block_weight, acc=accumulator)

    if use_bias:
        bias = tl.load(bias_ptr + col_offsets * bias_stride, mask=col_mask, other=0.0).to(tl.float32)
        accumulator += bias[None, :]

    tl.store(
        output_ptr + output_offsets,
        accumulator,
        mask=output_mask,
    )

# x: [B, S, D]
# weight: [output_size, input_size] input_size = D
# bias: [output_size]
def triton_linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None):
    B, S, hidden_size = x.shape
    output_size = weight.shape[0]
    
    assert x.dim() == 3, "triton_linear only supports [B, S, hidden_size]"
    assert weight.dim() == 2, "weight must be [output_size, hidden_size]"
    assert x.is_cuda and weight.is_cuda, "triton_linear only supports CUDA tensors"
    assert weight.shape[1] == hidden_size, "weight input size must match x hidden size"

    n_rows = B * S
    n_cols = output_size
    
    output = torch.empty((B, S, n_cols), dtype=x.dtype, device=x.device)
    
    grid = lambda meta: (triton.cdiv(n_rows, meta["BLOCK_M"]) * triton.cdiv(n_cols, meta["BLOCK_N"]), )
        
    _triton_linear_kernel[grid](
        x,
        weight,
        bias,
        output,
        n_rows,
        n_cols,
        hidden_size,
        S,
        x.stride(0), x.stride(1), x.stride(2),
        weight.stride(0), weight.stride(1),
        output.stride(0), output.stride(1), output.stride(2),
        bias.stride(0) if bias is not None else 0,
        use_bias=bias is not None,
    )
    
    return output

# x: [B, S, hidden_size]
# weight: [output_size, input_size] input_size = hidden_size
# bias: [output_size]
def torch_linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]):
    # if bias is not None:
    #     output = x @ weight.T + bias
    # else:
    #     output = x @ weight.T 
    
    # return output
    
    return F.linear(
        x,
        weight,
        bias
    )
        
        
def test_linear(x_size: Tuple[int], weight_size: Tuple[int], use_bias: bool=True, dtype=torch.bfloat16, rtol=1e-3, atol=1e-3):

    device = torch.cuda.current_device()

    base_x = torch.randn((x_size[1], x_size[0], x_size[2]), dtype=dtype, device=device)
    x = base_x.transpose(0, 1)
    base_weight = torch.randn((weight_size[1], weight_size[0]), dtype=dtype, device=device)
    weight = base_weight.transpose(0, 1)
    bias = torch.randn((weight_size[0],), dtype=dtype, device=device) if use_bias else None

    torch_outputs = torch_linear(x, weight, bias)
    triton_outputs = triton_linear(x, weight, bias)

    torch.testing.assert_close(triton_outputs, torch_outputs, rtol=rtol, atol=atol)

    print("PASSED!")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['n_tokens'],
        x_vals=[2**i for i in range(8, 16)],
        x_log=True,
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'Torch'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='TFLOPs',
        plot_name='linear-performance',
        args={
            'hidden_size': 2048,
            'output_size': 2048,
            'use_bias': True,
        },
    )
)
def benchmark(
    n_tokens: int,
    hidden_size: int,
    output_size: int,
    use_bias: bool,
    provider: str,
    dtype=torch.bfloat16,
):
    device = torch.cuda.current_device()

    batch_size = 32
    assert n_tokens % batch_size == 0, "n_tokens must be a multiple of 32"
    seq_len = n_tokens // batch_size

    x = torch.randn((batch_size, seq_len, hidden_size), dtype=dtype, device=device)
    weight = torch.randn((output_size, hidden_size), dtype=dtype, device=device)
    bias = torch.randn((output_size,), dtype=dtype, device=device) if use_bias else None

    if provider == 'torch':
        fn = lambda: torch_linear(x, weight, bias)
    else:
        fn = lambda: triton_linear(x, weight, bias)

    ms = triton.testing.do_bench(fn)

    # GEMM FLOPs = 2 * M * N * K
    M = n_tokens
    N = output_size
    K = hidden_size
    tflops = lambda ms: (2.0 * M * N * K) * 1e-12 / (ms * 1e-3) 

    return tflops(ms)
    
    
    
if __name__ == '__main__':
    import os
    test_linear((16,16,512), (1024, 512), False)
    os.makedirs('./benchmark_results', exist_ok=True)
    benchmark.run(save_path='./benchmark_results', print_data=False)
