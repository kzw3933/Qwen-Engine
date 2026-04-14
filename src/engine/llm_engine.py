import torch

from engine.executor import ModelExecutor
from engine.sequence import Sequence, SequenceStatus
from engine.kvcache import KVCachePool

from typing import List


class LLMEngine:
    def __init__(self, model, tokenizer, kv_cache_pool: KVCachePool, device: torch.device, mode: str = "torch"):
        self.model = model
        self.tokenizer = tokenizer
        self.kv_cache_pool = kv_cache_pool
        self.device = device
        self.executor = ModelExecutor(
            model=model,
            kv_cache_pool=kv_cache_pool,
            device=device,
            mode=mode
        )      
        self._next_seq_id = 0
    
    def create_sequence(self, prompt_token_ids, max_new_tokens: int):
        sequence = Sequence(
            seq_id=self._next_seq_id,
            prompt_token_ids=prompt_token_ids,
            max_new_tokens=max_new_tokens
        )
        self._next_seq_id += 1
        return sequence
    
    def should_stop(self, sequence: Sequence):
        if sequence.output_len >= sequence.max_new_tokens:
            sequence.status = SequenceStatus.FINISHED
            sequence.finish_reason = "eos"
            return True
        
        return False
    
    
    def encode_prompts(self, prompt_texts: List[str], max_new_tokens: int = 16) -> List[Sequence]:
        if isinstance(prompt_texts, str):
            prompt_texts = [prompt_texts]
        
        
        sequences: List[Sequence] = []
        for prompt_text in prompt_texts:
            prompt_token_ids = self.tokenizer(
                prompt_text,
                return_tensors="pt"
            ).input_ids[0].tolist()
                
            sequence = self.create_sequence(
                prompt_token_ids=prompt_token_ids,
                max_new_tokens=max_new_tokens
            )
            
            sequences.append(sequence)
            
        return sequences
    
    def decode_sequences(self, sequences: List[Sequence]) -> List[str]:
        outputs = []
        for seq in sequences:
            all_token_ids = seq.prompt_token_ids + seq.output_token_ids
            output_text = self.tokenizer.decode(all_token_ids, skip_special_tokens=True)
            outputs.append(output_text)
        return outputs
        
    
    
    def generate(self, prompt_texts: List[str], max_new_tokens: int = 16) -> List[str]:
        
        sequences = self.encode_prompts(prompt_texts, max_new_tokens)
            
        try:
            prefill_logits, prefill_context = self.executor.run_prefill(sequences)
            
            last_logits = self.executor.gather_prefill_last_logits(
                prefill_logits,
                prefill_context.cu_seqlens
            )
            
            next_token_ids = self.executor.sample_next_tokens(last_logits)
            
            for seq, next_token_id in zip(sequences, next_token_ids):
                seq.output_token_ids.append(next_token_id)
                
            active_sequences = [seq for seq in sequences if not self.should_stop(seq)]
            
            while active_sequences:
                decode_logits, _ = self.executor.run_decode_step(active_sequences)
                next_token_ids = self.executor.sample_next_tokens(decode_logits)
                
                for seq, next_token_id in zip(active_sequences, next_token_ids):
                    seq.output_token_ids.append(next_token_id)
                    
                active_sequences = [seq for seq in active_sequences if not self.should_stop(seq)]

            return self.decode_sequences(sequences)
        finally:
            for seq in sequences:
                if seq.block_ids:
                    self.kv_cache_pool.free_blocks_by_ids(seq.block_ids)
                    seq.block_ids.clear()
                