import torch

from engine.context import Context
from engine.sequence import Sequence, SequenceStatus
from engine.kvcache import KVCachePool

from typing import List


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
    
    
    def run_prefill(self, sequences: List[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens = [0]
        cache_slots = []
        
        for seq in sequences:
            input_ids.extend(seq.prompt_token_ids)
            positions.extend(range(seq.prompt_len))
            cu_seqlens.append(cu_seqlens[-1] + seq.prompt_len)
            cache_slots.append(seq.cache_slot)
            
        context = Context(
            is_decode=False,
            cu_seqlens=torch.tensor(cu_seqlens, dtype=torch.long, device=self.device),
            max_seqlen=max(seq.prompt_len for seq in sequences),
            cache_slots=torch.tensor(cache_slots, dtype=torch.long, device=self.device),
        )

        input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device)
        positions = torch.tensor(positions, dtype=torch.long, device=self.device)
        
        logits = self.forward(
            input_ids=input_ids,
            positions=positions,
            context=context,
        )
        
        for seq in sequences:
            seq.num_computed_tokens = seq.prompt_len
            seq.status = SequenceStatus.PREFILLED
                
        return logits, context
    
    def gather_prefill_last_logits(self, logits: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        last_indices = cu_seqlens[1:] - 1
        return logits[last_indices]
    
    
    def run_decode_step(self, sequences):
        input_ids = [seq.last_token_id for seq in sequences]
        positions = [seq.num_computed_tokens for seq in sequences]
        cache_lens = [seq.num_computed_tokens for seq in sequences]
        cache_slots = [seq.cache_slot for seq in sequences]
        
        context = Context(
            is_decode=True,
            cache_lens=torch.tensor(cache_lens, dtype=torch.long, device=self.device),
            cache_slots=torch.tensor(cache_slots, dtype=torch.long, device=self.device)
        )

        input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device)
        positions = torch.tensor(positions, dtype=torch.long, device=self.device)
        
        
        logits = self.forward(
            input_ids=input_ids,
            positions=positions,
            context=context,
        )
        
        for seq in sequences:
            seq.num_computed_tokens += 1
            seq.status = SequenceStatus.DECODING
        
        return logits, context
    
    
    def sample_next_tokens(self, logits):
        if logits.dim() != 2:
            raise ValueError(f"Expected logits shape [B, V], got {tuple(logits.shape)}")
        return torch.argmax(logits, dim=-1).tolist()
    
    
    
    
    
