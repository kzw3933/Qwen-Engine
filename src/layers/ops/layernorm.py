import torch
import torch.nn as nn
import torch.nn.functional as F

import triton
import triton.language as tl

from typing import Tuple
from dataclasses import dataclass



@triton.jit
def _triton_rmsnorm_kernel(
    x_ptr,
    weight_ptr,
    output_ptr,
    n_rows,
    n_cols,
    num_heads,
    x_stride_t, x_stride_h, x_stride_d,
    weight_stride,
    output_stride_t, output_stride_h, output_stride_d,
    HAS_HEAD_DIM: tl.constexpr,
    eps,
    BLOCK_N: tl.constexpr,
):
    row_start = tl.program_id(axis=0)
    row_step = tl.num_programs(axis=0)
    
    for row_idx in tl.range(row_start, n_rows, row_step):
        if HAS_HEAD_DIM:
            token_idx = row_idx // num_heads
            head_idx = row_idx % num_heads
        else:
            token_idx = row_idx
            head_idx = 0

        row_ptr = (
            x_ptr
            + token_idx * x_stride_t
            + head_idx * x_stride_h
        )
        out_ptr = (
            output_ptr
            + token_idx * output_stride_t
            + head_idx * output_stride_h
        )
        
        sum_sq = tl.zeros((), dtype=tl.float32)
        for col_start in tl.range(0, n_cols, BLOCK_N):
            col_offsets = col_start + tl.arange(0, BLOCK_N)
            col_mask = col_offsets < n_cols
            
            part_row = tl.load(
                row_ptr + col_offsets * x_stride_d,
                mask=col_mask,
                other=0.0,
            )
            part_row_fp32 = part_row.to(tl.float32)
            sum_sq += tl.sum(part_row_fp32 * part_row_fp32, axis=0)
            
        mean_sq = sum_sq / n_cols
        rstd = tl.rsqrt(mean_sq + eps)
        
        for col_start in tl.range(0, n_cols, BLOCK_N):
            col_offsets = col_start + tl.arange(0, BLOCK_N)
            col_mask = col_offsets < n_cols

            part_row = tl.load(
                row_ptr + col_offsets * x_stride_d, 
                mask=col_mask, 
                other=0.0,
            )
            weight = tl.load(
                weight_ptr + col_offsets * weight_stride, 
                mask=col_mask, 
                other=0.0,
            )

            part_row_fp32 = part_row.to(tl.float32)
            weight_fp32 = weight.to(tl.float32)

            output_fp32 = part_row_fp32 * rstd * weight_fp32
            tl.store(
                out_ptr + col_offsets * output_stride_d, 
                output_fp32, 
                mask=col_mask
            )     


@dataclass
class GPUProps:
    sms: int
    warp_size: int
    regs_per_sm: int
    shared_memory_per_sm: int
    max_threads_per_sm: int
    

def get_gpu_props():
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    sms = props.multi_processor_count
    warp_size = props.warp_size
    regs_per_sm = props.regs_per_multiprocessor
    shared_memory_per_sm = props.shared_memory_per_multiprocessor
    max_threads_per_sm = props.max_threads_per_multi_processor
    
    return GPUProps(
        sms=sms,
        warp_size=warp_size,
        regs_per_sm=regs_per_sm,
        shared_memory_per_sm=shared_memory_per_sm,
        max_threads_per_sm=max_threads_per_sm
    )


# x: [T, D] or [T, H, D]
# weight: [D]
# output: [T, D] or [T, H, D]
def triton_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    assert x.dim() in (2, 3), "triton_rmsnorm only supports [T, D] or [T, H, D]"
    assert weight.dim() == 1, "weight must be 1D"
    assert x.is_cuda and weight.is_cuda, "triton_rmsnorm only supports CUDA tensors"
    assert weight.shape[0] == x.shape[-1], "weight size must match last dimension of x"

    if x.dim() == 3:
        T, H, D = x.shape
        n_rows = T * H
        num_heads = H
    else:
        T, D = x.shape
        n_rows = T
        num_heads = 1

    n_cols = D
    y = torch.empty_strided(x.shape, x.stride(), dtype=x.dtype, device=x.device)
    
    BLOCK_N = 1024
    
    num_warps = 8

        
    props = get_gpu_props()
    
    # kernel = _triton_rmsnorm_kernel.warmup(
    #     x,
    #     weight,
    #     y,
    #     n_rows,
    #     n_cols,
    #     num_heads,
    #     x.stride(0), x.stride(1) if x.dim() == 3 else 0, x.stride(-1),
    #     weight.stride(0),
    #     y.stride(0), y.stride(1) if x.dim() == 3 else 0, y.stride(-1),
    #     HAS_HEAD_DIM=x.dim() == 3,
    #     eps=eps,
    #     BLOCK_N=BLOCK_N,
    #     num_warps=num_warps,
    #     grid=(1,)
    # )
    
    # kernel._init_handles()
    # n_regs_per_thread = kernel.n_regs
    # sram_per_program = kernel.metadata.shared
    
    # reg_occupancy = props.regs_per_sm // (n_regs_per_thread * props.warp_size * num_warps)
    # sram_occupancy = props.shared_memory_per_sm // sram_per_program if sram_per_program > 0 else float('inf')
    # thread_occupancy = props.max_threads_per_sm // (props.warp_size * num_warps)
    
    # programs_per_sm = min(reg_occupancy, sram_occupancy, thread_occupancy)
    n_programs_gpu = 6 * props.sms
    
    grid_row = min(n_programs_gpu, n_rows)
    grid = (grid_row, 1, 1)
    
    _triton_rmsnorm_kernel[grid](
        x,
        weight,
        y,
        n_rows,
        n_cols,
        num_heads,
        x.stride(0), x.stride(1) if x.dim() == 3 else 0, x.stride(-1),
        weight.stride(0),
        y.stride(0), y.stride(1) if x.dim() == 3 else 0, y.stride(-1),
        HAS_HEAD_DIM=x.dim() == 3,
        eps=eps,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
    )
    
    return y

# x: [T, D] or [T, H, D]
# weight: [D]
# output: [T, D] or [T, H, D]
def torch_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        
    # x_fp32 = x.float()
    # w_fp32 = weight.float()

    # mean_square = (x_fp32 * x_fp32).mean(dim=-1, keepdim=True)
    # x_norm = x_fp32 * torch.rsqrt(mean_square + eps)
    # output = x_norm * w_fp32

    # return output.to(dtype=x.dtype)
    
    return F.rms_norm(
        x,
        normalized_shape=(weight.shape[-1],),
        weight=weight,
        eps=eps
    )  
        
def test_rmsnorm(x_size: Tuple[int], eps: float=1e-6, dtype=torch.bfloat16, rtol=1e-3, atol=1e-3):
    
    device = torch.cuda.current_device()
    x = torch.rand(x_size, dtype=dtype, device=device)
    weight = torch.rand(x_size[-1], dtype=dtype, device=device)
    torch_outputs = torch_rmsnorm(x, weight, eps=eps)
    triton_outputs = triton_rmsnorm(x, weight=weight, eps=eps)
    
    torch.testing.assert_close(triton_outputs, torch_outputs, rtol=rtol, atol=atol)
    
    print("PASSED!")
    
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['n_tokens'],
        x_vals=[2**i for i in range(8, 16)],   # 256 ~ 32768
        x_log=True,
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'Torch'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='GB/s',
        plot_name='rmsnorm-performance',
        args={
            'hidden_size': 2048
        },
    )
)
def benchmark(n_tokens: int, hidden_size: int, provider: str, dtype=torch.bfloat16):
    device = torch.cuda.current_device()

    x = torch.randn((n_tokens, hidden_size), dtype=dtype, device=device)
    weight = torch.randn((hidden_size,), dtype=dtype, device=device)
    eps = 1e-6

    if provider == 'torch':
        fn = lambda: torch_rmsnorm(x, weight, eps)
    else:
        fn = lambda: triton_rmsnorm(x, weight, eps)

    ms = triton.testing.do_bench(fn)

    bytes_x = x.numel() * x.element_size()
    bytes_w = weight.numel() * weight.element_size()
    bytes_y = x.numel() * x.element_size()

    gbps = (bytes_x + bytes_w + bytes_y) * 1e-9 / (ms * 1e-3)
    
    return gbps
    
    
    
if __name__ == '__main__':
    import os
    test_rmsnorm((32,2048))
    os.makedirs('./benchmark_results', exist_ok=True)
    benchmark.run(save_path='./benchmark_results', print_data=False)
