import torch
import torch.nn.functional as F

import triton
import triton.language as tl

from typing import Tuple
from dataclasses import dataclass


@triton.jit
def _triton_embedding_kernel(
    x_ptr,
    weight_ptr,
    output_ptr,
    n_rows,
    n_cols,
    seq_len,
    x_stride_b, x_stride_s,
    weight_stride_vocab, weight_stride_dim,
    output_stride_b, output_stride_s, output_stride_d,
    BLOCK_N: tl.constexpr,
):
    row_start = tl.program_id(axis=0)
    row_step = tl.num_programs(axis=0)

    for row_idx in tl.range(row_start, n_rows, row_step):
        
        batch_idx = row_idx // seq_len
        seq_idx = row_idx % seq_len
        token = tl.load(x_ptr + batch_idx * x_stride_b + seq_idx * x_stride_s)
        
        for col_start in tl.range(0, n_cols, BLOCK_N):
            col_offsets = col_start + tl.arange(0, BLOCK_N)
            col_mask = col_offsets < n_cols

            embedding = tl.load(
                weight_ptr + token * weight_stride_vocab + col_offsets * weight_stride_dim,
                mask=col_mask,
                other=0.0,
            )

            tl.store(
                output_ptr + batch_idx * output_stride_b + seq_idx * output_stride_s + col_offsets * output_stride_d,
                embedding,
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


# x: [B, S]
# weight: [vocab_size, embedding_dim]
# output: [B, S, embedding_dim]
def triton_embedding(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    assert x.dim() == 2, "triton_embedding only supports [B, S]"
    assert weight.dim() == 2, "weight must be [vocab_size, embedding_dim]"
    assert x.is_cuda and weight.is_cuda, "triton_embedding only supports CUDA tensors"

    B, S = x.shape
    vocab_size, embedding_dim = weight.shape

    n_rows = B * S
    n_cols = embedding_dim
    y = torch.empty((B, S, embedding_dim), dtype=weight.dtype, device=x.device)

    BLOCK_N = 1024
    num_warps = 8

    props = get_gpu_props()

    # kernel = _triton_embedding_kernel.warmup(
    #     x,
    #     weight,
    #     y,
    #     n_rows,
    #     n_cols,
    #     S,
    #     x.stride(0), x.stride(1),
    #     weight.stride(0), weight.stride(1),
    #     y.stride(0), y.stride(1), y.stride(2),
    #     BLOCK_SIZE=BLOCK_SIZE,
    #     num_warps=num_warps,
    #     grid=(1, 1, 1),
    # )

    # kernel._init_handles()
    # n_regs_per_thread = kernel.n_regs
    # sram_per_program = kernel.metadata.shared
    #
    # reg_occupancy = props.regs_per_sm // (n_regs_per_thread * num_warps * props.warp_size) if n_regs_per_thread > 0 else float('inf')
    # sram_occupancy = props.shared_memory_per_sm // sram_per_program if sram_per_program > 0 else float('inf')
    # thread_occupancy = props.max_threads_per_sm // (num_warps * props.warp_size)
    #
    # programs_per_sm = min(reg_occupancy, sram_occupancy, thread_occupancy)
    # n_programs_gpu = programs_per_sm * props.sms

    n_programs_gpu = 6 * props.sms
    grid_row = min(n_programs_gpu, n_rows)
    
    grid = (grid_row, 1, 1)

    _triton_embedding_kernel[grid](
        x,
        weight,
        y,
        n_rows,
        n_cols,
        S,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        y.stride(0), y.stride(1), y.stride(2),
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
    )

    return y


# x: [B, S]
# weight: [vocab_size, embedding_dim]
# output: [B, S, embedding_dim]
def torch_embedding(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return F.embedding(x, weight=weight)


def test_embedding(x_size: Tuple[int], weight_size: Tuple[int], weight_dtype=torch.bfloat16, atol=1e-3, rtol=1e-3):
    device = torch.cuda.current_device()

    base_x = torch.randint(0, weight_size[0], (x_size[1], x_size[0]), dtype=torch.int32, device=device)
    x = base_x.transpose(0, 1)
    weight = torch.randn(weight_size, dtype=weight_dtype, device=device)

    triton_outputs = triton_embedding(x, weight)
    torch_outputs = torch_embedding(x, weight)

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
        plot_name="embedding-performance",
        args={
            "embedding_dim": 2048,
            "vocab_size": 151936,
        },
    )
)
def benchmark(n_tokens: int, embedding_dim: int, vocab_size: int, provider: str, weight_dtype=torch.bfloat16):
    device = torch.cuda.current_device()

    batch_size = 32
    seq_len = n_tokens // batch_size
    x = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.int32, device=device)
    weight = torch.randn((vocab_size, embedding_dim), dtype=weight_dtype, device=device)

    if provider == "torch":
        fn = lambda: torch_embedding(x, weight)
    else:
        fn = lambda: triton_embedding(x, weight)

    ms = triton.testing.do_bench(fn)

    bytes_x = n_tokens * x.element_size()
    bytes_weight = n_tokens * embedding_dim * weight.element_size()
    bytes_out = n_tokens * embedding_dim * weight.element_size()

    gbs = lambda ms: (bytes_x + bytes_weight + bytes_out) * 1e-9 / (ms * 1e-3)

    return gbs(ms)


if __name__ == "__main__":
    import os

    test_embedding((4, 8), (1024, 2048))
    os.makedirs("./benchmark_results", exist_ok=True)
    benchmark.run(save_path="./benchmark_results", print_data=False)
