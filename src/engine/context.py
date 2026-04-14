import torch

from dataclasses import dataclass


@dataclass
class Context:
    is_decode: bool = False
    
    # prefill
    cu_seqlens: torch.Tensor | None = None
    max_seqlen: int | None = None
    
    # decode
    cache_lens: torch.Tensor | None = None
    
    # sequence
    seq_lens: torch.Tensor | None = None
    
    # paged cache
    block_tables: torch.Tensor | None = None
    block_size: int | None = None
    slot_mapping: torch.Tensor | None = None
    block_offsets: torch.Tensor | None = None
    
    
    
    
    
    