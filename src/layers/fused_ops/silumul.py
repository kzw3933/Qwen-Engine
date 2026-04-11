import torch
import torch.nn.functional as F


from dataclasses import dataclass
from typing import Tuple


import triton
import triton.language as tl
    
    
@triton.jit
def _triton_silumul_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_rows,
    n_cols,
    seq_len,
    x_stride_b, x_stride_s, x_stride_d,
    y_stride_b, y_stride_s, y_stride_d,
    output_stride_b, output_stride_s, output_stride_d,
    BLOCK_N: tl.constexpr,
):
    row_start = tl.program_id(axis=0)
    row_step = tl.num_programs(axis=0)
    
    for row_idx in tl.range(row_start, n_rows, row_step):
        batch_idx = row_idx // seq_len
        seq_idx = row_idx % seq_len
        
        x_row_ptr = x_ptr + batch_idx * x_stride_b + seq_idx * x_stride_s
        y_row_ptr = y_ptr + batch_idx * y_stride_b + seq_idx * y_stride_s
        out_ptr = output_ptr + batch_idx * output_stride_b + seq_idx * output_stride_s
        
        for col_start in tl.range(0, n_cols, BLOCK_N):
            col_offsets = col_start + tl.arange(0, BLOCK_N)
            col_mask = col_offsets < n_cols
            
            part_x = tl.load(
                x_row_ptr + col_offsets * x_stride_d,
                mask=col_mask,
                other=0.0
            )
            part_y = tl.load(
                y_row_ptr + col_offsets * y_stride_d,
                mask=col_mask,
                other=0.0
            )
            
            part_x_fp32 = part_x.to(tl.float32)
            sigmoid = 1.0 / (1.0 + tl.exp(-part_x_fp32))
            silu_x = (part_x_fp32 * sigmoid).to(part_x.dtype)
            output = silu_x * part_y
        
            tl.store(
                out_ptr + col_offsets * output_stride_d,
                output,
                col_mask,
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
    
# x: [B, S, D] y: [B, S, D]
# output: [B, S, D]
def triton_silumul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.dim() == 3, "triton_silumul only supports [B, S, D]"
    assert y.dim() == 3, "triton_silumul only supports [B, S, D]"
    assert x.shape == y.shape, "x and y must have the same shape"
    assert x.is_cuda and y.is_cuda, "triton_silumul only supports CUDA tensors"
    assert x.device == y.device, "x and y must be on the same device"
    assert x.dtype == y.dtype, "x and y must have the same dtype"
    assert x.is_contiguous() and y.is_contiguous(), "triton_silumul expects contiguous inputs"

    B, S, D = x.shape
    n_rows = B * S
    n_cols = D
    
    num_warps = 8
    BLOCK_N = 1024
    
    output = torch.empty_strided(x.shape, x.stride(), dtype=x.dtype, device=x.device)
    
    props = get_gpu_props()
    
    # kernel = _triton_silumul_kernel.warmup(
    #     x,
    #     weight,
    #     y,
    #     n_rows,
    #     n_cols,
    #     S,
    #     x.stride(0), x.stride(1), x.stride(2),
    #     y.stride(0), y.stride(1), y.stride(2),
    #     output.stride(0), output.stride(1), output.stride(2),
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
    
    _triton_silumul_kernel[grid](
        x,
        y,
        output,
        n_rows,
        n_cols,
        S,
        x.stride(0), x.stride(1), x.stride(2),
        y.stride(0), y.stride(1), y.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
    )
    
    return output
    

# x: [B, S, D] y: [B, S, D]
# output: [B, S, D]
def torch_silumul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.silu(x) * y


def test_silumul(
    x_size: Tuple[int],
    dtype: torch.dtype = torch.bfloat16,
    rtol: float = 1e-3,
    atol: float = 1e-3,
):
    device = torch.cuda.current_device()

    base_x = torch.randn((x_size[1], x_size[0], x_size[2]), dtype=dtype, device=device)
    base_y = torch.randn((x_size[1], x_size[0], x_size[2]), dtype=dtype, device=device)
    x = base_x.transpose(0, 1).contiguous()
    y = base_y.transpose(0, 1).contiguous()

    torch_outputs = torch_silumul(x, y)
    triton_outputs = triton_silumul(x, y)

    torch.testing.assert_close(triton_outputs, torch_outputs, rtol=rtol, atol=atol)

    print("PASSED!")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["n_tokens"],
        x_vals=[2**i for i in range(8, 16)],
        x_log=True,
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "Torch"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="GB/s",
        plot_name="silumul-performance",
        args={
            "hidden_size": 2048,
        },
    )
)
def benchmark(
    n_tokens: int,
    hidden_size: int,
    provider: str,
    dtype: torch.dtype = torch.bfloat16,
):
    device = torch.cuda.current_device()

    batch_size = 32
    assert n_tokens % batch_size == 0, "n_tokens must be a multiple of 32"
    seq_len = n_tokens // batch_size

    x = torch.randn((batch_size, seq_len, hidden_size), dtype=dtype, device=device)
    y = torch.randn((batch_size, seq_len, hidden_size), dtype=dtype, device=device)

    if provider == "torch":
        fn = lambda: torch_silumul(x, y)
    else:
        fn = lambda: triton_silumul(x, y)

    ms = triton.testing.do_bench(fn)

    bytes_x = x.numel() * x.element_size()
    bytes_y = y.numel() * y.element_size()
    bytes_out = x.numel() * x.element_size()

    gbps = (bytes_x + bytes_y + bytes_out) * 1e-9 / (ms * 1e-3)

    return gbps


if __name__ == "__main__":
    import os

    test_silumul((16, 16, 2048))
    os.makedirs("./benchmark_results", exist_ok=True)
    benchmark.run(save_path="./benchmark_results", print_data=False)

