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
    prompt_token_ids: List[int]
    output_token_ids: List[int] = field(default_factory=list)
    cache_slot: int | None = None
    num_computed_tokens: int = 0
    max_new_tokens: int = 128
    status: SequenceStatus = SequenceStatus.WAITING
    finish_reason: str | None = None
    
    
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
    