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
    
    # both
    cache_slots: torch.Tensor | None = None
    
    
    
    
    
    