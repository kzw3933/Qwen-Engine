import torch
import torch.nn as nn
import torch.nn.functional as F

import triton
import triton.language as tl

from typing import Tuple
from dataclasses import dataclass


@triton.jit
def _triton_silu_kernel(
    x_ptr,
    output_ptr,
    n_rows,
    n_cols,
    seq_len,
    x_stride_b, x_stride_s, x_stride_d,
    output_stride_b, output_stride_s, output_stride_d,
    BLOCK_N: tl.constexpr,
):
    row_start = tl.program_id(axis=0)
    row_step = tl.num_programs(axis=0)

    for row_idx in tl.range(row_start, n_rows, row_step):
        batch_idx = row_idx // seq_len
        seq_idx = row_idx % seq_len

        row_ptr = x_ptr + batch_idx * x_stride_b + seq_idx * x_stride_s
        out_ptr = output_ptr + batch_idx * output_stride_b + seq_idx * output_stride_s

        for col_start in tl.range(0, n_cols, BLOCK_N):
            col_offsets = col_start + tl.arange(0, BLOCK_N)
            col_mask = col_offsets < n_cols

            part_row = tl.load(
                row_ptr + col_offsets * x_stride_d,
                mask=col_mask,
                other=0.0,
            )
            part_row_fp32 = part_row.to(tl.float32)
            sigmoid = 1.0 / (1.0 + tl.exp(-part_row_fp32))
            output_fp32 = part_row_fp32 * sigmoid

            tl.store(
                out_ptr + col_offsets * output_stride_d,
                output_fp32,
                mask=col_mask,
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

    return GPUProps(
        sms=props.multi_processor_count,
        warp_size=props.warp_size,
        regs_per_sm=props.regs_per_multiprocessor,
        shared_memory_per_sm=props.shared_memory_per_multiprocessor,
        max_threads_per_sm=props.max_threads_per_multi_processor,
    )


# x: [B, S, D]
# output: [B, S, D]
def triton_silu(x: torch.Tensor) -> torch.Tensor:
    assert x.dim() == 3, "triton_silu only supports [B, S, D]"
    assert x.is_cuda, "triton_silu only supports CUDA tensors"

    B, S, hidden_size = x.shape

    n_rows = B * S
    n_cols = hidden_size
    y = torch.empty_strided(x.shape, x.stride(), dtype=x.dtype, device=x.device)

    BLOCK_N = 1024
    num_warps = 8

    props = get_gpu_props()
    
    # kernel = _triton_silu_kernel.warmup(
    #     x,
    #     y,
    #     n_rows,
    #     n_cols,
    #     S,
    #     x.stride(0), x.stride(1), x.stride(2), 
    #     y.stride(0), y.stride(1), y.stride(2),
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

    grid_row = min(max(1, n_programs_gpu), n_rows)
    grid = (grid_row, 1, 1)

    _triton_silu_kernel[grid](
        x,
        y,
        n_rows,
        n_cols,
        S,
        x.stride(0), x.stride(1), x.stride(2),
        y.stride(0), y.stride(1), y.stride(2),
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
    )

    return y

# x: [B, S, D]
# output: [B, S, D]
def torch_silu(x: torch.Tensor) -> torch.Tensor:
    # x_fp32 = x.float()
    # sigmoid = 1.0 / (1.0 + torch.exp(-x_fp32))
    # output = x_fp32 * sigmoid
    # return output.to(dtype=x.dtype)

    return F.silu(x)

def test_silu(x_size: Tuple[int], dtype=torch.bfloat16, rtol=1e-3, atol=1e-3):
    device = torch.cuda.current_device()
    base_x = torch.randn((x_size[1], x_size[0], x_size[2]), dtype=dtype, device=device)
    x = base_x.transpose(0, 1)

    torch_outputs = torch_silu(x)
    triton_outputs = triton_silu(x)

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
        ylabel='GB/s',
        plot_name='silu-performance',
        args={
            'hidden_size': 2048,
        },
    )
)
def benchmark(n_tokens: int, hidden_size: int, provider: str, dtype=torch.bfloat16):
    device = torch.cuda.current_device()

    batch_size = 32
    assert n_tokens % batch_size == 0, "n_tokens must be a multiple of 32"
    seq_len = n_tokens // batch_size

    x = torch.randn((batch_size, seq_len, hidden_size), dtype=dtype, device=device)

    if provider == 'torch':
        fn = lambda: torch_silu(x)
    else:
        fn = lambda: triton_silu(x)

    ms = triton.testing.do_bench(fn)

    bytes_x = x.numel() * x.element_size()
    bytes_y = x.numel() * x.element_size()

    gbps = (bytes_x + bytes_y) * 1e-9 / (ms * 1e-3)

    return gbps


if __name__ == '__main__':
    import os
    test_silu((16, 16, 2048))
    os.makedirs('./benchmark_results', exist_ok=True)
    benchmark.run(save_path='./benchmark_results', print_data=False)
