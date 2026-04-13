import torch

from dataclasses import dataclass
from typing import Tuple


@dataclass
class SequenceCacheView:
    slot: int
    length: int = 0
    
class KVCachePool:
    def __init__(
        self,
        num_layers: int,
        max_num_seqs: int,
        max_model_len: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.num_layers = num_layers
        self.max_num_seqs = max_num_seqs
        self.max_model_len = max_model_len
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device
        
        self.k_caches = torch.empty(
            num_layers,
            max_num_seqs,
            max_model_len,
            num_kv_heads,
            head_dim,
            dtype=dtype,
            device=device,
        )

        self.v_caches = torch.empty(
            num_layers,
            max_num_seqs,
            max_model_len,
            num_kv_heads,
            head_dim,
            dtype=dtype,
            device=device,
        )
        
        self.free_slots = list(range(max_num_seqs-1, -1, -1))
        
    def allocate_slot(self) -> int:
        if not self.free_slots:
            raise RuntimeError("No free KV cache slots available")
        return self.free_slots.pop()
    
    def free_slot(self, slot: int) -> None:
        if slot < 0 or slot >= self.max_num_seqs:
            raise ValueError(f"Invalid KV cache slot: {slot}")
        self.free_slots.append(slot)
        
        
    def write(
        self,
        layer_idx: int,
        slot: int,
        start_pos: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise ValueError(f"Invalid layer_idx: {layer_idx}")
        if slot < 0 or slot >= self.max_num_seqs:
            raise ValueError(f"Invalid KV cache slot: {slot}")
        if k.shape != v.shape:
            raise ValueError("k and v must have the same shape")
        if k.dim() != 3:
            raise ValueError("k and v must be [T, Hk, D]")
        
        t, h, d = k.shape
        if h != self.num_kv_heads or d != self.head_dim:
            raise ValueError(
                f"Expected [T, {self.num_kv_heads}, {self.head_dim}], got {tuple(k.shape)}"
            )
            
        end_pos = start_pos + t
        
        if end_pos > self.max_model_len:
            raise ValueError(
                f"Cache write exceeds max_model_len: {end_pos} > {self.max_model_len}"
            )
            
        self.k_caches[layer_idx, slot, start_pos:end_pos].copy_(k)
        self.v_caches[layer_idx, slot, start_pos:end_pos].copy_(v)
        
    def read(self, layer_idx: int, slot: int, end_pos: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise ValueError(f"Invalid layer_idx: {layer_idx}")
        if slot < 0 or slot >= self.max_num_seqs:
            raise ValueError(f"Invalid KV cache slot: {slot}")
        
        if end_pos < 0 or end_pos > self.max_model_len:
            raise ValueError(f"Invalid end_pos: {end_pos}")
        
        k = self.k_caches[layer_idx, slot, :end_pos]
        v = self.v_caches[layer_idx, slot, :end_pos]
        
        return k, v
    
    
    def reset(self) -> None:
        self.free_slots = list(range(self.max_num_seqs-1, -1, -1))