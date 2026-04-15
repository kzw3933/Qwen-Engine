from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List

class SequenceStatus(Enum):
    WAITING = auto()
    PREFILLED = auto()
    DECODING = auto()
    FINISHED = auto()
    
    
@dataclass
class Sequence:
    seq_id: int
    
    prompt_text: str
    
    prompt_token_ids: List[int]
    output_token_ids: List[int] = field(default_factory=list)
    
    # page kv cache
    block_ids: List[int] = field(default_factory=list)
    
    # prefix cache metadata
    shared_prefix_len: int = 0
    shared_block_count: int = 0
    used_prefix_cache: bool = False
    block_hashes: List[int] = field(default_factory=list)
    
    # runtime states
    num_computed_tokens: int = 0
    num_kv_tokens: int = 0
    
    
    max_new_tokens: int = 128
    status: SequenceStatus = SequenceStatus.WAITING
    finish_reason: str | None = None
    
    
    def get_tokens(self, start: int, total: int) -> List[int]:
        end = start + total
    
        prompt_len = self.prompt_len
        
        if end <= prompt_len:
            return self.prompt_token_ids[start:end]
        
        if start >= prompt_len:
            output_start = start - prompt_len
            output_end = end - prompt_len
            return self.output_token_ids[output_start:output_end]
        
        prompt_part = self.prompt_token_ids[start:prompt_len]
        output_part = self.output_token_ids[: end - prompt_len]
        
        return prompt_part + output_part
    
    
    @property
    def prompt_len(self) -> int:
        return len(self.prompt_token_ids)
    
    @property
    def output_len(self) -> int:
        return len(self.output_token_ids)
    
    @property
    def total_len(self) -> int:
        return self.prompt_len + self.output_len
    
    
    @property
    def last_token_id(self) -> int:
        if self.output_token_ids:
            return self.output_token_ids[-1]
        return self.prompt_token_ids[-1]
    
    @property
    def prompt_suffix_len(self) -> int:
        return self.prompt_len - self.shared_prefix_len