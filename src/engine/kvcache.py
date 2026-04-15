import torch

import hashlib

from dataclasses import dataclass
from typing import Tuple, List, Dict

from engine.sequence import Sequence


@dataclass
class PrefixBlockEntry:
    block_hash: int
    block_id: int
    block_tokens: Tuple[int,...] 
    block_index: int
    prefix_len: int
    
@dataclass
class PrefixSharedResult:
    block_hashes: List[int]
    shared_block_ids: List[int]
    shared_token_len: int
    
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
        self.shared_blocks: Dict[int, PrefixBlockEntry] = {} 
        self.block_refcounts = [0 for _ in range(num_blocks)]
        self.block_id2hash: Dict[int, int] = {}
        
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
        
        
    def get_num_full_blocks(self, seq_len: int) -> int:
        if seq_len < 0:
            raise ValueError(f"Invalid seq_len: {seq_len}")
        return seq_len // self.block_size
    
    def get_num_required_blocks(self, seq_len: int) -> int:
        if seq_len < 0:
            raise ValueError(f"Invalid seq_len: {seq_len}")
        return (seq_len + self.block_size - 1) // self.block_size
    
    
    def hash_block_tokens(self, parent_hash: int, block_tokens: List[int]) -> int:
        h = hashlib.blake2b(digest_size=8)
        h.update(parent_hash.to_bytes(8, byteorder="little", signed=False))
        for token in block_tokens:
            h.update(int(token).to_bytes(8, byteorder="little", signed=True))
        return int.from_bytes(h.digest(), byteorder="little", signed=False)
    
    
    def compute_block_hashes(self, token_ids: List[int]) -> List[int]:
        num_full_blocks = self.get_num_full_blocks(len(token_ids))
        block_hashes: List[int] = []
        parent_hash = 0
        
        for block_idx in range(num_full_blocks):
            start = block_idx * self.block_size
            end = start + self.block_size
            block_tokens = token_ids[start:end]
            block_hash = self.hash_block_tokens(parent_hash, block_tokens)
            block_hashes.append(block_hash)
            parent_hash = block_hash
            
        return block_hashes
    
    def find_longest_prefix_result(self, seq: Sequence) -> PrefixSharedResult:
        token_ids = seq.prompt_token_ids
        num_full_blocks = self.get_num_full_blocks(len(token_ids))
        shared_block_ids: List[int] = []
        
        block_hashes = self.compute_block_hashes(token_ids)
        
        for block_idx in range(num_full_blocks):
            start = block_idx * self.block_size
            end = start + self.block_size
            block_hash = block_hashes[block_idx]    
            block_tokens = token_ids[start:end]    
            entry = self.shared_blocks.get(block_hash)
            if entry is None or entry.block_tokens != block_tokens:
                break
            shared_block_ids.append(entry.block_id)
            
        return PrefixSharedResult(
            block_hashes=block_hashes,
            shared_block_ids=shared_block_ids,
            shared_token_len=len(shared_block_ids) * self.block_size
        )
        
    def retain_blocks(self, block_ids: List[int]) -> None:
        for block_id in block_ids:
            if block_id < 0 or block_id >= self.num_blocks:
                raise ValueError(f"Invalid KV cache block id: {block_id}")
            if self.block_refcounts[block_id] <= 0:
                raise RuntimeError(
                    f"Cannot retain block {block_id} with invalid refcount={self.block_refcounts[block_id]}"
                )
            self.block_refcounts[block_id] += 1
            
    def release_blocks(self, block_ids: List[int]) -> None:
        for block_id in block_ids:
            if block_id < 0 or block_id >= self.num_blocks:
                raise ValueError(f"Invalid KV cache block id: {block_id}")
            if self.block_refcounts[block_id] <= 0:
                raise RuntimeError(
                    f"Cannot release block {block_id} with refcount={self.block_refcounts[block_id]}"
                )
            self.block_refcounts[block_id] -= 1
            if self.block_refcounts[block_id] == 0:
                self.free_block(block_id)
    
    def allocate_block(self) -> int:
        if not self.free_blocks:
            raise RuntimeError("No free KV cache blocks available")
        block_id = self.free_blocks.pop()
        self.block_refcounts[block_id] = 1
        return block_id
    
    def free_block(self, block_id: int) -> None:
        if block_id < 0 or block_id >= self.num_blocks:
            raise ValueError(f"Invalid KV cache block id: {block_id}")
        if self.block_refcounts[block_id] != 0:
            raise RuntimeError(
                f"Cannot free block {block_id} with non-zero refcount={self.block_refcounts[block_id]}"
            )
        block_hash = self.block_id2hash.pop(block_id, None)
        if block_hash is not None:
            self.shared_blocks.pop(block_hash, None)
        self.free_blocks.append(block_id)
        
        
    def allocate_prefill_sequence(self, seq: Sequence) -> None:
        if seq.block_ids:
            raise ValueError("seq.block_ids must be empty before allocate_sequence")
        
        prefix_result = self.find_longest_prefix_result(seq)
        
        if prefix_result.shared_block_ids:
            self.retain_blocks(prefix_result.shared_block_ids)
            seq.block_ids.extend(prefix_result.shared_block_ids)
            
        num_required_blocks = self.get_num_required_blocks(seq.prompt_len)
        num_shared_blocks = len(prefix_result.shared_block_ids)
        num_missing_blocks = num_required_blocks - num_shared_blocks
        
        if num_missing_blocks > 0:
            for i in range(num_missing_blocks):
                block_id = self.allocate_block()
                seq.block_ids.append(block_id)
                if seq.prompt_len >= prefix_result.shared_token_len + (i+1) * self.block_size:                        
                    block_hash = prefix_result.block_hashes[num_shared_blocks+i]
                    self.shared_blocks[block_hash] = PrefixBlockEntry(
                        block_hash=block_hash,
                        block_id=block_id,
                        block_tokens=seq.get_tokens(prefix_result.shared_token_len+i*self.block_size, self.block_size),
                        block_index=num_shared_blocks+i,
                        prefix_len=(num_shared_blocks+i+1) * self.block_size
                    )

                    
        seq.shared_prefix_len = prefix_result.shared_token_len
        seq.shared_block_count = len(prefix_result.shared_block_ids)
        seq.used_prefix_cache = len(prefix_result.shared_block_ids) > 0
        seq.block_hashes = prefix_result.block_hashes
        
        
    def allocate_decode_sequence(self, seq: Sequence) -> None:
        target_len = seq.num_kv_tokens + 1
        
        num_required_blocks = self.get_num_required_blocks(target_len)
        num_missing_blocks = num_required_blocks - len(seq.block_ids)
        
        if num_missing_blocks > 0:
            for _ in range(num_missing_blocks):
                seq.block_ids.append(self.allocate_block())
        
        
        
        
    def release_sequence(self, sequence: Sequence) -> None:
        if sequence.block_ids:
            self.release_blocks(sequence.block_ids)
            sequence.block_ids.clear()
            
    def write_tokens(
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
        
        t, _h, _d = k.shape
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
    
    def register_shared_blocks(self, sequences: List[Sequence]) -> None:
        for seq in sequences:
            num_full_blocks = self.get_num_full_blocks(seq.total_len)
            if num_full_blocks > len(seq.block_hashes):
                assert num_full_blocks - len(seq.block_hashes) == 1
                block_hashes = self.compute_block_hashes(seq.prompt_token_ids + seq.output_token_ids)
                block_idx = len(seq.block_hashes)
                block_id = seq.block_ids[block_idx]
                self.shared_blocks[block_hashes[block_idx]] = PrefixBlockEntry(
                    block_hash=block_hashes[block_idx],
                    block_id=block_id,
                    block_tokens=seq.get_tokens(block_idx * self.block_size, self.block_size),
                    block_index=block_idx,
                    prefix_len=(block_idx+1) * self.block_size
                )
                seq.block_hashes.append(block_hashes[block_idx])
        
    def reset(self) -> None:
        self.free_blocks = list(range(self.num_blocks - 1, -1, -1))
        self.block_refcounts = [0 for _ in range(self.num_blocks)]
        self.shared_blocks.clear()
        self.block_id2hash.clear()

    
            
    