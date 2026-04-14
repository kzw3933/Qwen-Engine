from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F

import triton
import triton.language as tl

from engine import Context, KVCachePool

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
    
@triton.jit
def _triton_store_kv_cache_kernel(
    src_k_ptr, src_v_ptr,
    dst_k_ptr, dst_v_ptr,
    physical_block_ids_ptr,
    block_offsets_ptr,
    src_stride_t, src_stride_h, src_stride_d,
    dst_stride_layer, dst_stride_block, dst_stride_pos, dst_stride_h, dst_stride_d,
    layer_idx,
    head_dim,
    BLOCK_D: tl.constexpr
):
    token_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    
    block_id = tl.load(physical_block_ids_ptr + token_idx)
    block_offset = tl.load(block_offsets_ptr + token_idx)
    
    d_offsets = tl.arange(0, BLOCK_D)
    d_mask = d_offsets < head_dim
    
    k_vec = tl.load(
        src_k_ptr
        + token_idx * src_stride_t
        + kv_head_idx * src_stride_h
        + d_offsets * src_stride_d
    )
    
    v_vec = tl.load(
        src_v_ptr
        + token_idx * src_stride_t
        + kv_head_idx * src_stride_h
        + d_offsets * src_stride_d
    )
    
    tl.store(
        dst_k_ptr
        + layer_idx * dst_stride_layer
        + block_id * dst_stride_block
        + block_offset * dst_stride_pos
        + kv_head_idx * dst_stride_h
        + d_offsets * dst_stride_d,
        k_vec,
        mask=d_mask
    )
    
    tl.store(
        dst_v_ptr
        + layer_idx * dst_stride_layer
        + block_id * dst_stride_block
        + block_offset * dst_stride_pos
        + kv_head_idx * dst_stride_h
        + d_offsets * dst_stride_d,
        v_vec,
        mask=d_mask
    )

    
# k: [T, Hk, D] v: [T, Hv, D] Hk = Hv 
def triton_store_kv_cache(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    physical_block_ids: torch.Tensor,
    block_offsets: torch.Tensor,
    layer_idx: int,
) -> None:
    assert k.shape == v.shape, "k and v must have the same shape"
    assert k.dim() == 3, "k and v must be [T, Hk, D]"
    assert k_cache.shape == v_cache.shape, "k_cache and v_cache must have the same shape"
    assert k_cache.dim() == 5, "cache tensors must be [L, B, P, Hk, D]"
    
    assert physical_block_ids.dim() == 1, "physical_block_ids must be [T]"
    assert block_offsets.dim() == 1, "block_offsets must be [T]"
    assert physical_block_ids.numel() == k.shape[0], "physical_block_ids size must match T"
    assert block_offsets.numel() == k.shape[0], "block_offsets size must match T"
    
    assert k.is_cuda and v.is_cuda, "source k/v must be CUDA tensors"
    assert k_cache.is_cuda and v_cache.is_cuda, "cache tensors must be CUDA tensors"
    assert physical_block_ids.is_cuda and block_offsets.is_cuda, "physical_block_ids/block_offsets must be CUDA tensors"
    
    assert k.dtype == v.dtype, "k and v dtypes must match"
    assert k_cache.dtype == v_cache.dtype, "cache dtypes must match"
    assert k.dtype == k_cache.dtype, "source and cache dtypes must match"
    
    assert physical_block_ids.dtype in (torch.int32, torch.int64), "physical_block_ids must be int32 or int64"
    assert block_offsets.dtype in (torch.int32, torch.int64), "block_offsets must be int32 or int64"
    
    
    n_tokens, num_kv_heads, head_dim = k.shape
    num_layers, _num_blocks, _block_size, cache_num_kv_heads, cache_head_dim = k_cache.shape
    
    assert num_kv_heads == cache_num_kv_heads, "num_kv_heads mismatch"
    assert head_dim == cache_head_dim, "head_dim mismatch"
    assert 0 <= layer_idx < num_layers, f"Invalid layer_idx: {layer_idx}"
    
    if head_dim <= 64:
        BLOCK_D = 64
    elif head_dim <= 128:
        BLOCK_D = 128
    elif head_dim <= 256:
        BLOCK_D = 256
    else:
        raise ValueError(f"Unsupported head_dim for Triton store kernel: {head_dim}")
    
    grid = (n_tokens, num_kv_heads)
    
    _triton_store_kv_cache_kernel[grid](
        k, v,
        k_cache, v_cache,
        physical_block_ids, block_offsets,
        k.stride(0), k.stride(1), k.stride(2),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3), k_cache.stride(4), 
        layer_idx,
        head_dim,
        BLOCK_D=BLOCK_D
    )
        
    
@triton.jit
def _triton_prefill_flash_attention_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    cu_seqlens_ptr, 
    q_stride_t, q_stride_h, q_stride_d,
    k_stride_t, k_stride_h, k_stride_d,
    v_stride_t, v_stride_h, v_stride_d,
    o_stride_t, o_stride_h, o_stride_d,
    scale,
    num_q_heads,
    num_kv_heads,
    head_dim,
    max_seqlen,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    seq_block_idx = tl.program_id(axis=0)
    q_head_idx = tl.program_id(axis=1)
    seq_idx = tl.program_id(axis=2)
    
    num_kv_groups = num_q_heads // num_kv_heads
    kv_head_idx = q_head_idx // num_kv_groups
    
    seq_start = tl.load(cu_seqlens_ptr + seq_idx)
    seq_end = tl.load(cu_seqlens_ptr + seq_idx + 1)
    seq_len = seq_end - seq_start
    
    m_offsets = seq_block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_offsets < seq_len
    d_offsets = tl.arange(0, BLOCK_D)
    d_mask = d_offsets < head_dim
    
    q = tl.load(
        q_ptr
        + (seq_start + m_offsets[:, None]) * q_stride_t
        + q_head_idx * q_stride_h
        + d_offsets[None, :] * q_stride_d,
        mask=m_mask[:, None] & d_mask[None, :],
        other=0.0
    )
    
    m_i = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    
    for n_start in tl.range(0, max_seqlen, BLOCK_N):
        n_offsets = n_start + tl.arange(0, BLOCK_N)
        n_mask = n_offsets < seq_len
        
        k = tl.load(
            k_ptr
            + (seq_start + n_offsets[None, :]) * k_stride_t
            + kv_head_idx * k_stride_h
            + d_offsets[:, None] * k_stride_d,
            mask=d_mask[:, None] & n_mask[None, :],
            other=0.0
        )
        
        qk = tl.dot(q, k)
        qk = qk * scale
        
        casual_mask = m_offsets[:, None] >= n_offsets[None, :]
        qk = tl.where(
            m_mask[:, None] & n_mask[None, :] & casual_mask,
            qk,
            -float('inf') 
        )
        
        m_ij = tl.max(qk, axis=1)
        m_i_new = tl.maximum(m_i, m_ij)
        
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])
        
        v = tl.load(
            v_ptr
            + (seq_start + n_offsets[:, None]) * v_stride_t
            + kv_head_idx * v_stride_h
            + d_offsets[None, :] * v_stride_d,
            mask=n_mask[:, None] & d_mask[None, :],
            other=0.0
        )
        
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_i_new
    
    acc = acc / l_i[:, None]
    
    tl.store(
        o_ptr
        + (seq_start + m_offsets[:, None]) * o_stride_t
        + q_head_idx * o_stride_h
        + d_offsets[None, :] * o_stride_d,
        acc,
        mask=m_mask[:, None] & d_mask[None, :]
    )
    

# q: [T, Hq, D] k: [T, Hk, D] v: [T, Hv, D] Hk = Hv
# output: [T, Hq, D] 
def triton_prefill_spda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    context: Context,
    kv_cache_pool: KVCachePool,
    layer_idx: int,
    scale: float
) -> torch.Tensor:
    assert q.dim() == 3 and k.dim() == 3 and v.dim() == 3, "q, k, v must be [T, H, D]"
    assert k.shape == v.shape, "k and v must have the same shape"
    assert q.shape[0] == k.shape[0] == v.shape[0], "token count mismatch"
    assert q.is_cuda and k.is_cuda and v.is_cuda, "triton_prefill_spda only supports CUDA tensors"
    assert q.device == k.device == v.device, "q, k, v must be on the same device"
    assert q.dtype == k.dtype == v.dtype, "q, k, v must have the same dtype"

    assert context.cu_seqlens is not None, "prefill requires context.cu_seqlens"
    assert context.slot_mapping is not None, "prefill requires context.slot_mapping"
    assert context.block_offsets is not None, "prefill requires context.block_offsets"
    assert context.max_seqlen is not None, "prefill requires context.max_seqlen"

    _n_tokens, num_q_heads, head_dim = q.shape
    _, num_kv_heads, _ = k.shape
    assert num_q_heads % num_kv_heads == 0, "Hq must be divisible by Hk for GQA"

    num_seqs = len(context.block_tables)
    cu_seqlens = context.cu_seqlens

    triton_store_kv_cache(
        k, v,
        kv_cache_pool.k_caches,
        kv_cache_pool.v_caches,
        physical_block_ids=context.slot_mapping,
        block_offsets=context.block_offsets,
        layer_idx=layer_idx,
    )
    
    output = torch.empty_like(q)
    
    if head_dim <= 64:
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_D = 64
    elif head_dim <= 128:
        BLOCK_M = 32
        BLOCK_N = 32
        BLOCK_D = 128
    elif head_dim <= 256:
        BLOCK_M = 16
        BLOCK_N = 16
        BLOCK_D = 256
    else:
        raise ValueError(f"Unsupported head_dim for Triton prefill kernel: {head_dim}")

    max_seqlen = int(context.max_seqlen)
    
    grid = (
        triton.cdiv(max_seqlen, BLOCK_M),
        num_q_heads,
        num_seqs
    )
    
    _triton_prefill_flash_attention_kernel[grid](
        q, k, v, output,
        cu_seqlens,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        scale,
        num_q_heads,
        num_kv_heads,
        head_dim,
        max_seqlen,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D
    )
    
    return output


@triton.jit
def _triton_decode_splitkv_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    partial_acc_ptr,
    partial_m_ptr,
    partial_l_ptr,
    cache_lens_ptr,
    block_tables_ptr,
    num_splits_ptr,
    q_stride_b, q_stride_h, q_stride_d,
    cache_stride_layer, cache_stride_block, cache_stride_pos, cache_stride_h, cache_stride_d,
    block_table_stride_b, block_table_stride_blk,
    pacc_stride_b, pacc_stride_h, pacc_stride_s, pacc_stride_d,
    pm_stride_b, pm_stride_h, pm_stride_s,
    pl_stride_b, pl_stride_h, pl_stride_s,
    layer_idx,
    scale,
    split_size,
    block_size,
    num_q_heads,
    num_kv_heads,
    head_dim,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    split_idx = tl.program_id(0)
    q_head_idx = tl.program_id(1)
    batch_idx = tl.program_id(2)
    
    num_kv_groups = num_q_heads // num_kv_heads
    kv_head_idx = q_head_idx // num_kv_groups
    
    cache_len = tl.load(cache_lens_ptr + batch_idx)
    num_splits = tl.load(num_splits_ptr + batch_idx)
    kv_len = cache_len + 1
    
    d_offsets = tl.arange(0, BLOCK_D)
    d_mask = d_offsets < head_dim
    
    pm_ptr = (
        partial_m_ptr
        + batch_idx * pm_stride_b
        + q_head_idx * pm_stride_h
        + split_idx * pm_stride_s
    )
    
    pl_ptr = (
        partial_l_ptr
        + batch_idx * pl_stride_b
        + q_head_idx * pl_stride_h
        + split_idx * pl_stride_s
    )
    
    pacc_ptrs = (
        partial_acc_ptr
        + batch_idx * pacc_stride_b
        + q_head_idx * pacc_stride_h
        + split_idx * pacc_stride_s
        + d_offsets * pacc_stride_d
    )
    
    if split_idx >= num_splits:
        tl.store(pm_ptr, -float('inf'))
        tl.store(pl_ptr, 0.0)
        tl.store(pacc_ptrs, 0.0, mask=d_mask)
        return
    
    split_start = split_idx * split_size
    split_end = tl.minimum(split_start + split_size, kv_len)
    
    if split_start >= split_end:
        tl.store(pm_ptr, -float('inf'))
        tl.store(pl_ptr, 0.0)
        tl.store(pacc_ptrs, 0.0, mask=d_mask)
        return
    
    q_vec = tl.load(
        q_ptr
        + batch_idx * q_stride_b
        + q_head_idx * q_stride_h
        + d_offsets * q_stride_d,
        mask=d_mask,
        other=0.0
    )
    
    m_i = -float('inf')
    l_i = 0.0
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    
    for n_start in tl.range(split_start, split_end, BLOCK_N):
        n_offsets = n_start + tl.arange(0, BLOCK_N)
        n_mask = n_offsets < split_end
        
        logical_block_idx = n_offsets // block_size
        block_offsets = n_offsets % block_size
        
        physical_block_ids = tl.load(
            block_tables_ptr
            + batch_idx * block_table_stride_b
            + logical_block_idx * block_table_stride_blk,
            mask=n_mask,
            other=0,
        )
        
        k_block = tl.load(
            k_cache_ptr
            + layer_idx * cache_stride_layer
            + physical_block_ids[:, None] * cache_stride_block
            + block_offsets[:, None] * cache_stride_pos
            + kv_head_idx * cache_stride_h
            + d_offsets[None, :] * cache_stride_d,
            mask=n_mask[:, None] & d_mask[None, :],
            other=0.0
        ).to(tl.float32)
        
        scores = tl.sum(k_block * q_vec[None, :], axis=1)
        scores = scores * scale
        scores = tl.where(n_mask, scores, -float('inf'))
        
        m_ij = tl.max(scores, axis=0)
        m_i_new = tl.maximum(m_i, m_ij)
        
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(scores - m_i_new)
        
        v_block = tl.load(
            v_cache_ptr
            + layer_idx * cache_stride_layer
            + physical_block_ids[:, None] * cache_stride_block
            + block_offsets[:, None] * cache_stride_pos
            + kv_head_idx * cache_stride_h
            + d_offsets[None, :] * cache_stride_d,
            mask=n_mask[:, None] & d_mask[None, :],
            other=0.0
        ).to(tl.float32)
        
        acc = acc * alpha + tl.sum(p[:, None] * v_block, axis=0)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = m_i_new
        
    tl.store(pm_ptr, m_i)
    tl.store(pl_ptr, l_i)
    tl.store(pacc_ptrs, acc, mask=d_mask)

@triton.jit
def _triton_decode_merge_kernel(
    partial_acc_ptr,
    partial_m_ptr,
    partial_l_ptr,
    out_ptr,
    num_splits_ptr,
    pacc_stride_b, pacc_stride_h, pacc_stride_s, pacc_stride_d,
    pm_stride_b, pm_stride_h, pm_stride_s,
    pl_stride_b, pl_stride_h, pl_stride_s,
    out_stride_b, out_stride_h, out_stride_d,
    head_dim,
    max_num_splits,
    BLOCK_D: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    q_head_idx = tl.program_id(1)
    
    d_offsets = tl.arange(0, BLOCK_D)
    d_mask = d_offsets < head_dim
    
    num_splits = tl.load(num_splits_ptr + batch_idx)
    m = -float('inf')
    
    for split_idx in tl.range(0, max_num_splits):
        if split_idx < num_splits:
            m_s = tl.load(
                partial_m_ptr
                + batch_idx * pm_stride_b
                + q_head_idx * pm_stride_h
                + split_idx * pm_stride_s
            )
            m = tl.maximum(m, m_s)
            
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    l = 0.0
    
    for split_idx in tl.range(0, max_num_splits):
        if split_idx < num_splits:
            m_s = tl.load(
                partial_m_ptr
                + batch_idx * pm_stride_b
                + q_head_idx * pm_stride_h
                + split_idx * pm_stride_s
            )
            l_s = tl.load(
                partial_l_ptr
                + batch_idx * pl_stride_b
                + q_head_idx * pl_stride_h
                + split_idx * pl_stride_s
            )
            acc_s = tl.load(
                partial_acc_ptr
                + batch_idx * pacc_stride_b
                + q_head_idx * pacc_stride_h
                + split_idx * pacc_stride_s
                + d_offsets * pacc_stride_d,
                mask=d_mask,
                other=0.0
            ).to(tl.float32)
            
            alpha = tl.exp(m_s - m)
            acc = acc + alpha * acc_s
            l = l + alpha * l_s
            
    out = acc / l
    
    tl.store(
        out_ptr
        + batch_idx * out_stride_b
        + q_head_idx * out_stride_h
        + d_offsets * out_stride_d,
        out,
        mask=d_mask
    )


def triton_decode_spda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    context: Context,
    kv_cache_pool: KVCachePool,
    layer_idx: int,
    scale: float,
    split_size: int = 256,
) -> torch.Tensor:
    assert q.dim() == 3 and k.dim() == 3 and v.dim() == 3, "q, k, v must be [T, H, D]"
    assert k.shape == v.shape, "k and v must have the same shape"
    assert q.shape[0] == k.shape[0] == v.shape[0], "token count mismatch"
    assert context.slot_mapping is not None, "decode requires context.slot_mapping"
    assert context.block_offsets is not None, "decode requires context.block_offsets"
    assert q.is_cuda and k.is_cuda and v.is_cuda, "triton_decode_spda only supports CUDA tensors"
    
    batch_size, num_q_heads, head_dim = q.shape
    _, num_kv_heads, _ = k.shape
    assert num_q_heads % num_kv_heads == 0, "Hq must be divisible by Hk for GQA"    
    
    triton_store_kv_cache(
        k,
        v,
        kv_cache_pool.k_caches,
        kv_cache_pool.v_caches,
        physical_block_ids=context.slot_mapping,
        block_offsets=context.block_offsets,
        layer_idx=layer_idx,
    )
    
    kv_lens = context.cache_lens + 1
    num_splits_per_seq = (kv_lens + split_size - 1) // split_size
    max_num_splits = int(num_splits_per_seq.max().item())
    
    partial_acc = torch.empty(
        (batch_size, num_q_heads, max_num_splits, head_dim),
        dtype=q.dtype,
        device=q.device
    )
    
    partial_m = torch.empty(
        (batch_size, num_q_heads, max_num_splits),
        dtype=torch.float32,
        device=q.device
    )
    
    partial_l = torch.empty(
        (batch_size, num_q_heads, max_num_splits),
        dtype=torch.float32,
        device=q.device
    )
    
    output = torch.empty_like(q)
    
    if head_dim <= 64:
        BLOCK_D = 64
        BLOCK_N = 64
    elif head_dim <= 128:
        BLOCK_D = 128
        BLOCK_N = 64
    elif head_dim <= 256:
        BLOCK_D = 256
        BLOCK_N = 32
    else:
        raise ValueError(f"Unsupported head_dim for Triton decode kernel: {head_dim}")
    
    split_grid = (max_num_splits, num_q_heads, batch_size)
    
    _triton_decode_splitkv_kernel[split_grid](
        q,
        kv_cache_pool.k_caches,
        kv_cache_pool.v_caches,
        partial_acc,
        partial_m,
        partial_l,
        context.cache_lens,
        context.block_tables,
        num_splits_per_seq,
        q.stride(0), q.stride(1), q.stride(2),
        kv_cache_pool.k_caches.stride(0), kv_cache_pool.k_caches.stride(1), kv_cache_pool.k_caches.stride(2), kv_cache_pool.k_caches.stride(3), kv_cache_pool.k_caches.stride(4),
        context.block_tables.stride(0), context.block_tables.stride(1),
        partial_acc.stride(0), partial_acc.stride(1), partial_acc.stride(2), partial_acc.stride(3),
        partial_m.stride(0), partial_m.stride(1), partial_m.stride(2), 
        partial_l.stride(0), partial_l.stride(1), partial_l.stride(2), 
        layer_idx,
        scale,
        split_size,
        context.block_size,
        num_q_heads,
        num_kv_heads,
        head_dim,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
    )
    
    merge_grid = (batch_size, num_q_heads)
    
    _triton_decode_merge_kernel[merge_grid](
        partial_acc,
        partial_m,
        partial_l,
        output,
        num_splits_per_seq,
        partial_acc.stride(0), partial_acc.stride(1), partial_acc.stride(2), partial_acc.stride(3),
        partial_m.stride(0), partial_m.stride(1), partial_m.stride(2), 
        partial_l.stride(0), partial_l.stride(1), partial_l.stride(2), 
        output.stride(0), output.stride(1), output.stride(2), 
        head_dim,
        max_num_splits,
        BLOCK_D=BLOCK_D,
    )
    
    return output
    

# q: [T, Hq, D] k: [T, Hk, D] v: [T, Hv, D] Hk = Hv
# output: [T, Hq, D] 
def torch_prefill_spda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    context: Context,
    kv_cache_pool: KVCachePool,
    layer_idx: int,
    scale: float,
) -> torch.Tensor:
    batch_size = len(context.block_tables)
    max_seqlen = context.max_seqlen
    cu_seqlens = context.cu_seqlens
    
    num_q_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    head_dim = q.shape[2]
    
    q_padded = torch.zeros((batch_size, max_seqlen, num_q_heads, head_dim), dtype=q.dtype, device=q.device)
    k_padded = torch.zeros((batch_size, max_seqlen, num_kv_heads, head_dim), dtype=k.dtype, device=k.device)
    v_padded = torch.zeros((batch_size, max_seqlen, num_kv_heads, head_dim), dtype=v.dtype, device=v.device)
    
    
    for i in range(batch_size):
        start = cu_seqlens[i].item()
        end = cu_seqlens[i+1].item()
        seq_len = end - start
        
        q_seq = q[start:end]
        k_seq = k[start:end]
        v_seq = v[start:end]
        
        q_padded[i, :seq_len] = q_seq
        k_padded[i, :seq_len] = k_seq
        v_padded[i, :seq_len] = v_seq
        
        block_table = context.block_tables[i]
        valid_block_ids = block_table[block_table >= 0].tolist()
        kv_cache_pool.append_tokens(
            layer_idx=layer_idx,
            block_ids=valid_block_ids,
            start_pos=0,
            k=k_seq,
            v=v_seq
        )
        
    seq_lens = context.seq_lens
        
    q_spda = q_padded.transpose(1, 2)
    k_spda = k_padded.transpose(1, 2)
    v_spda = v_padded.transpose(1, 2)
    
    q_pos = torch.arange(max_seqlen, device=q.device)
    k_pos = torch.arange(max_seqlen, device=k.device)
    
    casual = q_pos[:, None] >= k_pos[None, :]
    key_valid = k_pos[None, :] < seq_lens[:, None]
    query_valid = q_pos[None, :] < seq_lens[:, None]
    
    attn_mask = (
        casual.unsqueeze(0)
        & query_valid[:, :, None]
        & key_valid[:, None, :]
    ).unsqueeze(1)
    
    out = F.scaled_dot_product_attention(
        q_spda,
        k_spda,
        v_spda,
        attn_mask=attn_mask,
        dropout_p=0.0,
        is_causal=False,
        scale=scale,
        enable_gqa=(num_q_heads != num_kv_heads),
    )
    
    out = out.transpose(1, 2)
        
    outputs = []
    for i in range(batch_size):
        seq_len = seq_lens[i].item()
        outputs.append(out[i, :seq_len])
        
    return torch.cat(outputs, dim=0)
        
       
# q: [T, Hq, D] k: [T, Hk, D] v: [T, Hv, D] Hk = Hv
# output: [T, Hq, D]  
def torch_decode_spda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    context: Context,
    kv_cache_pool: KVCachePool,
    layer_idx: int,
    scale: float,
) -> torch.Tensor:
    
    batch_size, num_q_heads, head_dim = q.shape
    _, num_kv_heads, _ = k.shape
    
    kv_lens = context.seq_lens
    max_kv_len = int(kv_lens.max().item())
    
    k_padded = torch.zeros(
        (batch_size, max_kv_len, num_kv_heads, head_dim),
        dtype=k.dtype,
        device=k.device
    )
    
    v_padded = torch.zeros_like(k_padded)
    
    for seq_idx in range(batch_size):
        cache_len = int(context.cache_lens[seq_idx].item())
        full_len = int(kv_lens[seq_idx].item())
        
        block_table = context.block_tables[seq_idx]
        block_ids = block_table[block_table >= 0].tolist()
        
        k_new = k[seq_idx:seq_idx+1]
        v_new = v[seq_idx:seq_idx+1]
        
        kv_cache_pool.append_tokens(
            layer_idx=layer_idx,
            block_ids=block_ids,
            start_pos=cache_len,
            k=k_new,
            v=v_new
        )
        
        k_full, v_full = kv_cache_pool.gather_sequence(
            layer_idx,
            block_ids=block_ids,
            seq_len=full_len
        )
        
        k_padded[seq_idx, :full_len] = k_full
        v_padded[seq_idx, :full_len] = v_full
        
    q_spda = q.unsqueeze(2)
    k_spda = k_padded.transpose(1, 2)
    v_spda = v_padded.transpose(1, 2)
    
    kv_positions = torch.arange(max_kv_len, device=k.device)
    key_valid = kv_positions[None, :] < kv_lens[:, None]
    attn_mask = key_valid[:, None, None, :]
    
    out = F.scaled_dot_product_attention(
        q_spda,
        k_spda,
        v_spda,
        attn_mask=attn_mask,
        dropout_p=0.0,
        is_causal=False,
        scale=scale,
        enable_gqa=(num_q_heads != num_kv_heads),
    )
    
    return out.squeeze(2)


