import torch
import torch.nn as nn


import triton
import triton.language as tl

from typing import Tuple
from dataclasses import dataclass


@triton.jit
def _triton_rotary_embedding_kernel(
    cos_sin_cache_ptr,
    positions_ptr,
    q_ptr,
    k_ptr,
    q_out_ptr,
    k_out_ptr,
    n_tokens,
    num_q_heads,
    num_k_heads,
    head_dim,
    pos_stride_t,
    cache_stride_pos, cache_stride_d,
    q_stride_t, q_stride_h, q_stride_d,
    k_stride_t, k_stride_h, k_stride_d,
    q_out_stride_t, q_out_stride_h, q_out_stride_d,
    k_out_stride_t, k_out_stride_h, k_out_stride_d,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token_start = tl.program_id(axis=0)
    token_step = tl.num_programs(axis=0)
    head_block_idx = tl.program_id(axis=1)
    
    rotary_half = head_dim // 2
    q_head_offsets = head_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    k_head_offsets = head_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    q_head_mask = q_head_offsets < num_q_heads
    k_head_mask = k_head_offsets < num_k_heads
    
    for token_idx in tl.range(token_start, n_tokens, token_step):

        position = tl.load(
            positions_ptr + token_idx * pos_stride_t
        )
        cache_row_ptr = cos_sin_cache_ptr + position * cache_stride_pos
        
        for dim_start in tl.range(0, rotary_half, BLOCK_D):
            dim_offsets = dim_start + tl.arange(0, BLOCK_D)
            dim_mask = dim_offsets < rotary_half
            
            cos = tl.load(
                cache_row_ptr + dim_offsets * cache_stride_d,
                mask=dim_mask,
                other=0.0,
            ).to(tl.float32)
            sin = tl.load(
                cache_row_ptr + rotary_half * cache_stride_d + dim_offsets * cache_stride_d,
                mask=dim_mask,
                other=0.0,
            ).to(tl.float32)
            
            q_base_ptr = (
                q_ptr
                + token_idx * q_stride_t
            )
            q_out_base_ptr = (
                q_out_ptr
                + token_idx * q_out_stride_t
            )
            q_first_ptr = q_base_ptr + q_head_offsets[:, None] * q_stride_h + dim_offsets[None, :] * q_stride_d
            q_second_ptr = q_first_ptr + rotary_half * q_stride_d
            q_mask = q_head_mask[:,None] & dim_mask[None,:]
            
            q_first = tl.load(q_first_ptr, mask=q_mask, other=0.0).to(tl.float32)
            q_second = tl.load(q_second_ptr, mask=q_mask, other=0.0).to(tl.float32)
            
            q_out_first = cos[None, :] * q_first - sin[None, :] * q_second
            q_out_second = sin[None, :] * q_first + cos[None, :] * q_second
            
            tl.store(
                q_out_base_ptr + q_head_offsets[:, None] * q_out_stride_h + dim_offsets[None, :] * q_out_stride_d,
                q_out_first,
                mask=q_mask,
            )
            tl.store(
                q_out_base_ptr + q_head_offsets[:, None] * q_out_stride_h + (rotary_half + dim_offsets[None, :]) * q_out_stride_d,
                q_out_second,
                mask=q_mask,
            )
            
            k_base_ptr = (
                k_ptr
                + token_idx * k_stride_t
            )
            k_out_base_ptr = (
                k_out_ptr
                + token_idx * k_out_stride_t
            )
            k_first_ptr = k_base_ptr + k_head_offsets[:, None] * k_stride_h + dim_offsets[None, :] * k_stride_d
            k_second_ptr = k_first_ptr + rotary_half * k_stride_d
            k_mask = k_head_mask[:,None] & dim_mask[None,:]
            
            k_first = tl.load(k_first_ptr, mask=k_mask, other=0.0).to(tl.float32)
            k_second = tl.load(k_second_ptr, mask=k_mask, other=0.0).to(tl.float32)
            
            k_out_first = cos[None, :] * k_first - sin[None, :] * k_second
            k_out_second = sin[None, :] * k_first + cos[None, :] * k_second
            
            tl.store(
                k_out_base_ptr + k_head_offsets[:, None] * k_out_stride_h + dim_offsets[None, :] * k_out_stride_d,
                k_out_first,
                mask=k_mask,
            )
            tl.store(
                k_out_base_ptr + k_head_offsets[:, None] * k_out_stride_h + (rotary_half + dim_offsets[None, :]) * k_out_stride_d,
                k_out_second,
                mask=k_mask,
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

# cos_sin_cache: [max_position, rotary_embedding]
# positions: [T]
# q: [T, num_q_heads, head_dim]
# k: [T, num_k_heads, head_dim]  head_dim = rotary_embedding
def triton_rotary_embedding(
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
):
    T, = positions.shape
    n_tokens = T
    num_q_heads = q.shape[1]
    num_k_heads = k.shape[1]
    head_dim = q.shape[-1]

    assert q.dim() == 3 and k.dim() == 3, "q and k must be [T, num_heads, head_dim]"
    assert q.shape[:1] == positions.shape, "q must match positions shape on [T]"
    assert k.shape[:1] == positions.shape, "k must match positions shape on [T]"
    assert q.shape[-1] == k.shape[-1], "q and k must have the same head_dim"
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    assert cos_sin_cache.shape[-1] == head_dim, "cos_sin_cache last dim must equal head_dim"
    assert cos_sin_cache.shape[0] > 0, "cos_sin_cache must not be empty"
    assert positions.dtype in (torch.int32, torch.int64), "positions must be integer dtype"
    
    q_out = torch.empty_strided(q.shape, q.stride(), dtype=q.dtype, device=q.device)
    k_out = torch.empty_strided(k.shape, k.stride(), dtype=k.dtype, device=k.device)
    
    BLOCK_H = 4 if max(num_q_heads, num_k_heads) >= 4 else 1
    BLOCK_D = 128
    
    q_head_blocks = triton.cdiv(num_q_heads, BLOCK_H)
    k_head_blocks = triton.cdiv(num_k_heads, BLOCK_H)
    qk_max_head_blocks = max(q_head_blocks, k_head_blocks)
    
    num_warps = 8
    
    props = get_gpu_props()
    
    # kernel = _triton_rotary_embedding_kernel.warmup(
    #     cos_sin_cache,
    #     positions,
    #     q,
    #     k,
    #     q_out,
    #     k_out,
    #     n_tokens,
    #     num_q_heads,
    #     num_k_heads,
    #     head_dim,
    #     positions.stride(0),
    #     cos_sin_cache.stride(0), cos_sin_cache.stride(1),
    #     q.stride(0), q.stride(1), q.stride(2),
    #     k.stride(0), k.stride(1), k.stride(2),
    #     q_out.stride(0), q_out.stride(1), q_out.stride(2),
    #     k_out.stride(0), k_out.stride(1), k_out.stride(2),
    #     BLOCK_H=BLOCK_H,
    #     BLOCK_D=BLOCK_D,
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
    
    n_programs_gpu = 9 * props.sms
    
    grid_row = min(max(1, n_programs_gpu // qk_max_head_blocks), n_tokens)
    grid = (grid_row, qk_max_head_blocks)
    

    _triton_rotary_embedding_kernel[grid](
        cos_sin_cache,
        positions,
        q,
        k,
        q_out,
        k_out,
        n_tokens,
        num_q_heads,
        num_k_heads,
        head_dim,
        positions.stride(0),
        cos_sin_cache.stride(0), cos_sin_cache.stride(1),
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        q_out.stride(0), q_out.stride(1), q_out.stride(2),
        k_out.stride(0), k_out.stride(1), k_out.stride(2),
        BLOCK_H=BLOCK_H,
        BLOCK_D=BLOCK_D,
        num_warps=num_warps,
    )

    return q_out, k_out


# cos_sin_cache: [max_position, rotary_embedding]
# positions: [T]
# q, k: [T, num_heads, head_dim] head_dim = rotary_embedding
def torch_rotary_embedding(
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
):
    cos, sin = cos_sin_cache[positions].chunk(2, dim=-1)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    q_fp32 = q.float()
    q1, q2 = q_fp32.chunk(2, dim=-1)
    q_out = torch.cat(
        [cos * q1 - sin * q2, sin * q1 + cos * q2],
        dim=-1,
    ).to(q.dtype)

    k_fp32 = k.float()
    k1, k2 = k_fp32.chunk(2, dim=-1)
    k_out = torch.cat(
        [cos * k1 - sin * k2, sin * k1 + cos * k2],
        dim=-1,
    ).to(k.dtype)

    return q_out, k_out  
        
def test_rotary_embedding(
    positions_size: Tuple[int],
    q_size: Tuple[int],
    k_size: Tuple[int],
    rotary_embedding: int,
    max_position: int = 2048,
    dtype=torch.bfloat16,
    atol=1e-3,
    rtol=1e-3,
):

    device = torch.cuda.current_device()

    positions = torch.randint(0, max_position, positions_size, dtype=torch.int32, device=device)
    cos_sin_cache = torch.randn((max_position, rotary_embedding), dtype=torch.float32, device=device)
    q = torch.randn(q_size, dtype=dtype, device=device)
    k = torch.randn(k_size, dtype=dtype, device=device)

    torch_q, torch_k = torch_rotary_embedding(cos_sin_cache, positions, q, k)
    triton_q, triton_k = triton_rotary_embedding(cos_sin_cache, positions, q, k)
    
    torch.testing.assert_close(triton_q, torch_q, rtol=rtol, atol=atol)
    torch.testing.assert_close(triton_k, torch_k, rtol=rtol, atol=atol)

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
        plot_name='rotary-embedding-performance',
        args={
            'num_q_heads': 16,
            'num_k_heads': 16,
            'head_dim': 128,
            'max_position': 2048,
        },
    )
)
def benchmark(
    n_tokens: int,
    num_q_heads: int,
    num_k_heads: int,
    head_dim: int,
    max_position: int,
    provider: str,
    dtype=torch.bfloat16,
):
    device = torch.cuda.current_device()

    positions = torch.randint(0, max_position, (n_tokens,), dtype=torch.int32, device=device)
    cos_sin_cache = torch.randn((max_position, head_dim), dtype=torch.float32, device=device)
    q = torch.randn((n_tokens, num_q_heads, head_dim), dtype=dtype, device=device)
    k = torch.randn((n_tokens, num_k_heads, head_dim), dtype=dtype, device=device)

    if provider == 'torch':
        fn = lambda: torch_rotary_embedding(cos_sin_cache, positions, q, k)
    else:
        fn = lambda: triton_rotary_embedding(cos_sin_cache, positions, q, k)

    ms = triton.testing.do_bench(fn)

    bytes_positions = positions.numel() * positions.element_size()
    bytes_cache = n_tokens * head_dim * cos_sin_cache.element_size()
    bytes_q = q.numel() * q.element_size()
    bytes_k = k.numel() * k.element_size()
    bytes_q_out = q.numel() * q.element_size()
    bytes_k_out = k.numel() * k.element_size()

    gbps = lambda ms: (bytes_positions + bytes_cache + bytes_q + bytes_k + bytes_q_out + bytes_k_out) * 1e-9 / (ms * 1e-3)

    return gbps(ms)


if __name__ == '__main__':
    import os
    test_rotary_embedding(
        positions_size=(32,),
        q_size=(32, 32, 128),
        k_size=(32, 16, 128),
        rotary_embedding=128,
    )
    os.makedirs('./benchmark_results', exist_ok=True)
    benchmark.run(save_path='./benchmark_results', print_data=False)
