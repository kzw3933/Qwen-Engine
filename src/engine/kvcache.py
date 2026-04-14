import torch

from dataclasses import dataclass
from typing import Tuple, List


@dataclass
class SequenceCacheView:
    slot: int
    length: int = 0
    
class KVCachePool:
    def __init__(
        self,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device
        
        self.k_caches = torch.empty(
            num_layers,
            num_blocks,
            block_size,
            num_kv_heads,
            head_dim,
            dtype=dtype,
            device=device,
        )

        self.v_caches = torch.empty(
            num_layers,
            num_blocks,
            block_size,
            num_kv_heads,
            head_dim,
            dtype=dtype,
            device=device,
        )
        
        self.free_blocks = list(range(num_blocks-1, -1, -1))
        
    def get_num_required_blocks(self, seq_len: int) -> int:
        if seq_len < 0:
            raise ValueError(f"Invalid seq_len: {seq_len}")
        return (seq_len + self.block_size - 1) // self.block_size
    
    def allocate_block(self) -> int:
        if not self.free_blocks:
            raise RuntimeError("No free KV cache blocks available")
        return self.free_blocks.pop()
    
    def allocate_blocks(self, num_blocks: int) -> List[int]:
        if num_blocks < 0:
            raise ValueError(f"Invalid num_blocks: {num_blocks}")
        if len(self.free_blocks) < num_blocks:
            raise RuntimeError(
                f"Not enough free KV cache blocks: need {num_blocks}, have {len(self.free_blocks)}"
            )
        return [self.free_blocks.pop() for _ in range(num_blocks)]
    
    def free_block(self, block_id: int) -> None:
        if block_id < 0 or block_id >= self.num_blocks:
            raise ValueError(f"Invalid KV cache block id: {block_id}")
        self.free_blocks.append(block_id)
        
        
    def free_blocks_by_ids(self, block_ids: List[int]) -> None:
        for block_id in block_ids:
            self.free_block(block_id)
            
            
    def append_tokens(
        self,
        layer_idx: int,
        block_ids: List[int],
        start_pos: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise ValueError(f"Invalid layer_idx: {layer_idx}")
        if k.shape != v.shape:
            raise ValueError(f"k and v must have the same shape")
        if k.dim() != 3:
            raise ValueError("k and v must be [T, Hk, D]")
        
        t, h, d = k.shape
        end_pos = start_pos + t
        
        required_blocks = self.get_num_required_blocks(end_pos)
        if len(block_ids) < required_blocks:
            raise ValueError(
                f"Insufficient block_ids: need {required_blocks}, got {len(block_ids)}"
            )
        for i in range(t):
            token_pos = start_pos + i
            logical_block_idx = token_pos // self.block_size
            block_offset = token_pos % self.block_size
            physical_block_id = block_ids[logical_block_idx]
            
            self.k_caches[layer_idx, physical_block_id, block_offset].copy_(k[i])
            self.v_caches[layer_idx, physical_block_id, block_offset].copy_(v[i])

    def gather_sequence(
        self,
        layer_idx: int,
        block_ids: List[int],
        seq_len: int
    )  -> Tuple[torch.Tensor, torch.Tensor]:
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise ValueError(f"Invalid layer_idx: {layer_idx}")
        if seq_len < 0:
            raise ValueError(f"Invalid seq_len: {seq_len}")
        
        k_out = torch.empty(
            seq_len,
            self.num_kv_heads,
            self.head_dim,
            dtype=self.dtype,
            device=self.device
        )
        
        v_out = torch.empty_like(k_out)
        
        for token_pos in range(seq_len):
            logical_block_idx = token_pos // self.block_size
            block_offset = token_pos % self.block_size
            physical_block_id = block_ids[logical_block_idx]
            
            k_out[token_pos].copy_(self.k_caches[layer_idx, physical_block_id, block_offset])
            v_out[token_pos].copy_(self.v_caches[layer_idx, physical_block_id, block_offset])
            
        return k_out, v_out
    
    
    def build_block_table(self, block_ids: List[int], max_num_blocks: int) -> List[int]:
        if len(block_ids) > max_num_blocks:
            raise ValueError("block_ids exceeds max_num_blocks")
        table = [-1] * max_num_blocks
        table[:len(block_ids)] = block_ids
        
        return table
            
    
    def reset(self) -> None:
        self.free_blocks = list(range(self.num_blocks-1, -1, -1))