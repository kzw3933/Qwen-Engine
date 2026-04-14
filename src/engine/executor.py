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
    
    def forward(self, input_ids, positions, context):
        return self.model(
            input_ids=input_ids,
            positions=positions,
            context=context,
            kv_cache_pool=self.kv_cache_pool,
            mode=self.mode
        )
        
    def ensure_sequence_capacity(self, sequence: Sequence, target_len: int) -> None:
        required_blocks = self.kv_cache_pool.get_num_required_blocks(target_len)
        missing_blocks = required_blocks - len(sequence.block_ids)
        
        if missing_blocks > 0:
            new_block_ids = self.kv_cache_pool.allocate_blocks(missing_blocks)
            sequence.block_ids.extend(new_block_ids)
            
            
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
    
    def build_prefill_store_mapping(
        self,
        sequences: List[Sequence],
        cu_seqlens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        total_tokens = int(cu_seqlens[-1].item())
        slot_mapping = torch.empty((total_tokens, ), dtype=torch.long, device=self.device)
        block_offsets = torch.empty((total_tokens, ), dtype=torch.long, device=self.device)
        
        for seq_idx, seq in enumerate(sequences):
            start = int(cu_seqlens[seq_idx].item())
            end = int(cu_seqlens[seq_idx+1].item())
            
            for token_idx_in_seq in range(end - start):
                logical_block_idx = token_idx_in_seq // self.kv_cache_pool.block_size
                block_offset = token_idx_in_seq % self.kv_cache_pool.block_size
                physical_block_id = seq.block_ids[logical_block_idx]
                
                slot_mapping[start+token_idx_in_seq] = physical_block_id
                block_offsets[start+token_idx_in_seq] = block_offset
                
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
    
    
    def run_prefill(self, sequences: List[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens = [0]
        
        for seq in sequences:
            self.ensure_sequence_capacity(seq, seq.prompt_len)
            input_ids.extend(seq.prompt_token_ids)
            positions.extend(range(seq.prompt_len))
            cu_seqlens.append(cu_seqlens[-1] + seq.prompt_len)
            
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device)
        positions = torch.tensor(positions, dtype=torch.long, device=self.device)
        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.long, device=self.device)
        
        seq_lens = torch.tensor(
            [seq.prompt_len for seq in sequences],
            dtype=torch.long,
            device=self.device
        )
        
        block_tables = self.build_block_tables(sequences)
        slot_mapping, block_offsets = self.build_prefill_store_mapping(sequences, cu_seqlens)
        
        context = Context(
            is_decode=False,
            cu_seqlens=cu_seqlens,
            max_seqlen=max(seq.prompt_len for seq in sequences),
            seq_lens=seq_lens,
            block_tables=block_tables,
            block_size=self.kv_cache_pool.block_size,
            slot_mapping=slot_mapping,
            block_offsets=block_offsets
        )
        
        logits = self.forward(
            input_ids=input_ids,
            positions=positions,
            context=context,
        )
        
        for seq in sequences:
            seq.num_computed_tokens = seq.prompt_len
            seq.num_kv_tokens = seq.prompt_len
            seq.status = SequenceStatus.PREFILLED
                
        return logits, context
    
    def gather_prefill_last_logits(self, logits: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        last_indices = cu_seqlens[1:] - 1
        return logits[last_indices]
    
    
    def run_decode_step(self, sequences: List[Sequence]):
        input_ids = [seq.last_token_id for seq in sequences]
        positions = [seq.num_computed_tokens for seq in sequences]
        
        for seq in sequences:
            self.ensure_sequence_capacity(seq, seq.num_kv_tokens+1)
            
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
                    
        logits = self.forward(
            input_ids=input_ids,
            positions=positions,
            context=context,
        )
        
        for seq in sequences:
            seq.num_computed_tokens += 1
            seq.num_kv_tokens += 1
            seq.status = SequenceStatus.DECODING
        
        return logits, context
    
    
    def sample_next_tokens(self, logits):
        if logits.dim() != 2:
            raise ValueError(f"Expected logits shape [B, V], got {tuple(logits.shape)}")
        return torch.argmax(logits, dim=-1).tolist()
    
    
    
    
    
