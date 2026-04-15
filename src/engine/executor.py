import torch

from engine.context import Context
from engine.sequence import Sequence, SequenceStatus
from engine.kvcache import KVCachePool

from typing import List, Tuple


class ModelExecutor:
    def __init__(self, model, kv_cache_pool: KVCachePool, device: torch.device, mode: str = "torch"):
        self.model = model
        self.kv_cache_pool = kv_cache_pool
        self.device = device
        self.mode = mode
    
    def execute(self, input_ids, positions, context):
        return self.model(
            input_ids=input_ids,
            positions=positions,
            context=context,
            kv_cache_pool=self.kv_cache_pool,
            mode=self.mode
        )

            
    def build_block_tables(self, sequences: List[Sequence]) -> torch.Tensor:
        max_num_blocks = max(len(seq.block_ids) for seq in sequences)
        block_tables = torch.full(
            (len(sequences), max_num_blocks),
            fill_value=-1,
            dtype=torch.long,
            device=self.device
        )
        
        for i, seq in enumerate(sequences):
            if seq.block_ids:
                block_tables[i, :len(seq.block_ids)] = torch.tensor(
                    seq.block_ids,
                    dtype=torch.long,
                    device=self.device
                )
                
        return block_tables
    
    def gather_prefill_last_logits(self, logits: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        last_indices = cu_seqlens[1:] - 1
        return logits[last_indices]
    
    
    def sample_next_tokens(self, logits):
        if logits.dim() != 2:
            raise ValueError(f"Expected logits shape [B, V], got {tuple(logits.shape)}")
        return torch.argmax(logits, dim=-1).tolist()
    
    def build_prefill_store_mapping(
        self,
        sequences: List[Sequence],
        cu_seqlens: torch.Tensor,
        prefix_lens: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        total_tokens = int(cu_seqlens[-1].item())
        slot_mapping = torch.empty((total_tokens, ), dtype=torch.long, device=self.device)
        block_offsets = torch.empty((total_tokens, ), dtype=torch.long, device=self.device)
        
        for seq_idx, seq in enumerate(sequences):
            start = int(cu_seqlens[seq_idx].item())
            end = int(cu_seqlens[seq_idx+1].item())
            prefill_start = prefix_lens[seq_idx]
            
            for token_idx_in_chunk in range(end - start):
                token_pos = prefill_start + token_idx_in_chunk
                logical_block_idx = token_pos // self.kv_cache_pool.block_size
                block_offset = token_pos % self.kv_cache_pool.block_size
                physical_block_id = seq.block_ids[logical_block_idx]
                
                slot_mapping[start+token_idx_in_chunk] = physical_block_id
                block_offsets[start+token_idx_in_chunk] = block_offset
                
        return slot_mapping, block_offsets
    
    
    def build_decode_store_mapping(
        self,
        sequences: List[Sequence],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = len(sequences)
        
        cache_lens = torch.empty((batch_size, ), dtype=torch.long, device=self.device)
        slot_mapping = torch.empty((batch_size, ), dtype=torch.long, device=self.device)
        block_offsets = torch.empty((batch_size, ), dtype=torch.long, device=self.device)
        
        for seq_idx, seq in enumerate(sequences):
            cache_len = seq.num_kv_tokens
            logical_block_idx = cache_len // self.kv_cache_pool.block_size
            block_offset = cache_len % self.kv_cache_pool.block_size
            physical_block_id = seq.block_ids[logical_block_idx]
            
            cache_lens[seq_idx] = cache_len
            slot_mapping[seq_idx] = physical_block_id
            block_offsets[seq_idx] = block_offset
                
        return cache_lens, slot_mapping, block_offsets
    
    def run_prefill(self, sequences: List[Sequence]) -> Tuple[torch.Tensor, Context]:
        for seq in sequences:
            self.kv_cache_pool.allocate_prefill_sequence(seq)
        
        input_ids = []
        positions = []
        cu_seqlens = [0]
        
        for seq in sequences:
            if seq.prompt_suffix_len == 0:
                suffix_start = seq.prompt_len - 1
                suffix_token = seq.prompt_token_ids[-1]
                input_ids.append(suffix_token)
                positions.append(suffix_start)
                cu_seqlens.append(cu_seqlens[-1] + 1)
            else:
                suffix_start = seq.shared_prefix_len
                suffix_tokens = seq.prompt_token_ids[suffix_start:]
                input_ids.extend(suffix_tokens)
                positions.extend(range(suffix_start, seq.prompt_len))
                cu_seqlens.append(cu_seqlens[-1] + len(suffix_tokens))
            
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device)
        positions = torch.tensor(positions, dtype=torch.long, device=self.device)
        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.long, device=self.device)
        
        seq_lens = torch.tensor(
            [seq.prompt_len for seq in sequences],
            dtype=torch.long,
            device=self.device
        )
        
        prefix_lens = torch.tensor(
            [
                seq.shared_prefix_len if seq.prompt_suffix_len != 0
                else seq.shared_prefix_len - 1
                for seq in sequences
            ],
            dtype=torch.long,
            device=self.device
        )
        
        block_tables = self.build_block_tables(sequences)
        slot_mapping, block_offsets = self.build_prefill_store_mapping(sequences, cu_seqlens, prefix_lens)
        
        context = Context(
            is_decode=False,
            cu_seqlens=cu_seqlens,
            max_seqlen=max(seq.prompt_suffix_len for seq in sequences),
            prefix_lens=prefix_lens,
            seq_lens=seq_lens,
            block_tables=block_tables,
            block_size=self.kv_cache_pool.block_size,
            slot_mapping=slot_mapping,
            block_offsets=block_offsets
        )
            
        logits = self.execute(
            input_ids=input_ids,
            positions=positions,
            context=context
        )
        
        return logits, context
    
    def release_sequence(self, seq: Sequence):
        self.kv_cache_pool.release_sequence(seq)
        
    def register_shared_blocks(self, sequences: List[Sequence]) -> None:
        self.kv_cache_pool.register_shared_blocks(sequences)
    
    
    def run_decode_step(self, sequences: List[Sequence]):
        input_ids = [seq.last_token_id for seq in sequences]
        positions = [seq.num_computed_tokens for seq in sequences]
        
        for seq in sequences:
            self.kv_cache_pool.allocate_decode_sequence(seq)
            
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device)
        positions = torch.tensor(positions, dtype=torch.long, device=self.device)
        
        cache_lens, slot_mapping, block_offsets = self.build_decode_store_mapping(sequences)
        
        seq_lens = cache_lens + 1
        block_tables = self.build_block_tables(sequences)
        
        context = Context(
            is_decode=True,
            cache_lens=cache_lens,
            seq_lens=seq_lens,
            block_tables=block_tables,
            block_size=self.kv_cache_pool.block_size,
            slot_mapping=slot_mapping,
            block_offsets=block_offsets
        )
                    
        logits = self.execute(
            input_ids=input_ids,
            positions=positions,
            context=context,
        )
                
        return logits
    
    
    
    
    
