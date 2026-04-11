from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F

import triton
import triton.language as tl


@triton.jit
def _triton_flash_attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    qo_stride_b, qo_stride_s, qo_stride_h, qo_stride_d,
    kv_stride_b, kv_stride_s, kv_stride_h, kv_stride_d,
    scale,
    seq_len,
    head_dim: tl.constexpr,
    NUM_KV_GROUPS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    batch_idx = tl.program_id(axis=2)
    head_idx = tl.program_id(axis=1)
    seq_block_idx = tl.program_id(axis=0)
    
    kv_head_idx = head_idx // NUM_KV_GROUPS
    
    seq_block_offsets = seq_block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    seq_block_mask = seq_block_offsets < seq_len
    
    d_offsets = tl.arange(0, head_dim)
    
    q = tl.load(
        q_ptr + batch_idx * qo_stride_b + seq_block_offsets[:, None] * qo_stride_s + head_idx * qo_stride_h + d_offsets[None, :] * qo_stride_d,
        mask=seq_block_mask[:, None],
        other=0.0
    )
    
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - 1e10
    acc = tl.zeros([BLOCK_M, head_dim], dtype=tl.float32)
    
    for kv_seq_block_start in tl.range(0, seq_len, BLOCK_N):
        kv_seq_block_offsets = kv_seq_block_start +  tl.arange(0, BLOCK_N)
        
        kv_seq_block_mask = kv_seq_block_offsets < seq_len
        
        k = tl.load(
            k_ptr + d_offsets[:, None] * kv_stride_d + batch_idx * kv_stride_b + kv_seq_block_offsets[None, :] * kv_stride_s + kv_head_idx * kv_stride_h,
            mask=kv_seq_block_mask[None, :],
            other=0.0
        )
        
        qk = tl.dot(q, k) * scale
        
        casual_mask = seq_block_offsets[:, None] >= kv_seq_block_offsets[None, :]
        qk = tl.where(casual_mask & seq_block_mask[:, None] & kv_seq_block_mask[None, :], qk, -1e10)
        
        m_ij = tl.max(qk, axis=1)
        m_i_new = tl.maximum(m_i, m_ij)
        
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])
        
        v = tl.load(
            v_ptr + batch_idx * kv_stride_b + kv_seq_block_offsets[:, None] * kv_stride_s + kv_head_idx * kv_stride_h + d_offsets[None, :] * kv_stride_d,
            mask=kv_seq_block_mask[:, None],
            other=0.0
        )
        
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_i_new
        
    acc = acc / l_i[:, None]
    
    tl.store(
        o_ptr + batch_idx * qo_stride_b + seq_block_offsets[:, None] * qo_stride_s + head_idx * qo_stride_h + d_offsets[None, :] * qo_stride_d,
        acc,
        mask=seq_block_mask[:, None]
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


# q: [B, S, Hq, D] k: [B, S, Hk, D] v: [B, S, Hk, D]
def torch_spda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    
    # B, Sq, Hq, D = q.shape
    # _, Sk, Hk, _ = k.shape
    
    # assert Hq % Hk == 0, "Hq must be divisible by Hk"
    
    # q_fp32 = q.float()
    # k_fp32 = k.float()
    # v_fp32 = v.float()
    
    # repeat = Hq // Hk
    # k_fp32 = k_fp32.repeat_interleave(repeat, dim=2)
    # v_fp32 = v_fp32.repeat_interleave(repeat, dim=2)
    
    # q_fp32 = q_fp32.transpose(1, 2)
    # k_fp32 = k_fp32.transpose(1, 2)
    # v_fp32 = v_fp32.transpose(1, 2)
    
    # attn_scores = torch.matmul(q_fp32, k_fp32.transpose(-2, -1)) * scale
    # mask = torch.tril(
    #     torch.ones((Sq, Sk), device=attn_scores.device, dtype=torch.bool)
    # )
    # attn_scores.masked_fill_(~mask, float("-inf"))
    # attn_probs = F.softmax(attn_scores, dim=-1)
    
    # out = torch.matmul(attn_probs, v_fp32).transpose(1, 2).to(q.dtype)
    # return out
    
    
    B, Sq, Hq, D = q.shape
    _, Sk, Hk, _ = k.shape

    assert Hq % Hk == 0, "Hq must be divisible by Hk"

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    out = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=True,
        scale=scale,
        enable_gqa=(Hq != Hk),
    )

    return out.transpose(1, 2)


# q: [B, Sq, Hq, D] k: [B, Sk, Hk, D] v: [B, Sk, Hk, D]
# output: [B, Sq, Hq, D]
def triton_spda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    assert q.dim() == 4, "q must be [B, Sq, Hq, D]"
    assert k.dim() == 4, "k must be [B, Sk, Hk, D]"
    assert v.dim() == 4, "v must be [B, Sk, Hk, D]"
    assert q.is_cuda and k.is_cuda and v.is_cuda, "triton_attention only supports CUDA tensors"
    assert q.device == k.device == v.device, "q, k, v must be on the same device"
    assert q.dtype == k.dtype == v.dtype, "q, k, v must have the same dtype"
    assert q.shape[0] == k.shape[0] == v.shape[0], "batch size mismatch"
    assert k.shape == v.shape, "k and v must have the same shape"
    assert q.shape[-1] == k.shape[-1] == v.shape[-1], "head_dim mismatch"
    assert q.shape[-1] <= 256, "this FlashAttention-style skeleton assumes head_dim <= 256"
    assert q.shape[2] % k.shape[2] == 0, "Hq must be divisible by Hk for GQA"
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous(), "triton_attention expects contiguous inputs"

    B, Sq, Hq, D = q.shape
    _, Sk, Hk, _ = k.shape
    num_kv_groups = Hq // Hk

    output = torch.empty_like(q)

    BLOCK_M = 64
    BLOCK_N = 64
    num_warps = 4 if D <= 64 else 8
    props = get_gpu_props()

    # FlashAttention-2 style decomposition usually assigns one program to
    # one (batch, head, query-block) tile and streams over K/V blocks.
    grid = (triton.cdiv(Sq, BLOCK_M), Hq, B)

    # lse stores per-row log-sum-exp statistics for online softmax.
    # A backward pass or split-K reduction path can reuse this buffer later.
    _triton_flash_attention_kernel[grid](
        q,
        k,
        v,
        output,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        scale,
        Sq,
        D,
        NUM_KV_GROUPS=num_kv_groups,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
    )

    return output


def test_attention(
    q_size: Tuple[int],
    k_size: Tuple[int],
    v_size: Tuple[int],
    scale: float = 1.0,
    dtype: torch.dtype = torch.bfloat16,
    rtol: float = 1e-2,
    atol: float = 1e-2,
):
    device = torch.cuda.current_device()

    q = torch.randn(q_size, dtype=dtype, device=device).contiguous()
    k = torch.randn(k_size, dtype=dtype, device=device).contiguous()
    v = torch.randn(v_size, dtype=dtype, device=device).contiguous()

    torch_outputs = torch_spda(q, k, v, scale=scale)
    triton_outputs = triton_spda(q, k, v, scale=scale)

    torch.testing.assert_close(triton_outputs, torch_outputs, rtol=rtol, atol=atol)

    print("PASSED!")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["n_tokens"],
        x_vals=[2**i for i in range(8, 13)],
        x_log=True,
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "Torch"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="TFLOPs",
        plot_name="attention-performance",
        args={
            "num_q_heads": 16,
            "num_kv_heads": 8,
            "head_dim": 128,
            "causal": True,
        },
    )
)
def benchmark(
    n_tokens: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    causal: bool,
    provider: str,
    dtype: torch.dtype = torch.bfloat16,
):
    if not causal:
        raise ValueError("this benchmark currently assumes causal attention")

    device = torch.cuda.current_device()
    batch_size = 4
    assert n_tokens % batch_size == 0, "n_tokens must be a multiple of batch_size"
    seq_len = n_tokens // batch_size

    q = torch.randn((batch_size, seq_len, num_q_heads, head_dim), dtype=dtype, device=device).contiguous()
    k = torch.randn((batch_size, seq_len, num_kv_heads, head_dim), dtype=dtype, device=device).contiguous()
    v = torch.randn((batch_size, seq_len, num_kv_heads, head_dim), dtype=dtype, device=device).contiguous()
    scale = head_dim ** -0.5

    if provider == "torch":
        fn = lambda: torch_spda(q, k, v, scale=scale)
    else:
        fn = lambda: triton_spda(q, k, v, scale=scale)

    ms = triton.testing.do_bench(fn)

    # Approximate causal attention FLOPs:
    # qk^T ~= B * Hq * S * S * D
    # pv   ~= B * Hq * S * S * D
    # multiply-add counts as 2 FLOPs.
    B = batch_size
    S = seq_len
    H = num_q_heads
    D = head_dim
    flops = 4.0 * B * H * S * S * D

    return flops * 1e-12 / (ms * 1e-3)


if __name__ == "__main__":
    import os

    test_attention(
        q_size=(2, 16, 16, 128),
        k_size=(2, 16, 8, 128),
        v_size=(2, 16, 8, 128),
        scale=128 ** -0.5,
    )
    os.makedirs("./benchmark_results", exist_ok=True)
    benchmark.run(save_path="./benchmark_results", print_data=False)