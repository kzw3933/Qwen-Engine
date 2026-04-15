import torch

from engine.executor import ModelExecutor
from engine.sequence import Sequence, SequenceStatus
from engine.kvcache import KVCachePool

from typing import List, Callable


class LLMEngine:
    def __init__(self, model, tokenizer, kv_cache_pool: KVCachePool, max_num_seqs: int, device: torch.device, mode: str = "torch"):
        self.model = model
        self.tokenizer = tokenizer
        self.max_num_seqs = max_num_seqs
        self.device = device
        self.executor = ModelExecutor(
            model=model,
            kv_cache_pool=kv_cache_pool,
            device=device,
            mode=mode
        )      
        self._next_seq_id = 0
        self.work_queue: List[Sequence] = []
        self.finish_queue: List[Sequence] = []
    
    def create_sequence(self, prompt_text, prompt_token_ids, max_new_tokens: int):
        sequence = Sequence(
            seq_id=self._next_seq_id,
            prompt_text=prompt_text,
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
                prompt_text=prompt_text,
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
    
    def submit(self, prompt_texts: str | List[str], max_new_tokens: int = 16) -> None:
        if isinstance(prompt_texts, str):
            prompt_texts = [prompt_texts]
        self.work_queue.extend(
            self.encode_prompts(prompt_texts=prompt_texts, max_new_tokens=max_new_tokens)
        )
        
        
    def consume_sequences(self, status: SequenceStatus, max_num_seqs: int) -> List[Sequence]:
        seqs = []
        remaining_seqs = []
        
        for seq in self.work_queue:
            if (
                seq.status == status
                and len(seqs) < max_num_seqs
            ):
                seqs.append(seq)
            else:
                remaining_seqs.append(seq)
                
        self.work_queue = remaining_seqs
        
        return seqs
            
    
    def serving(self, print_func: Callable) -> None:
        
        while self.work_queue:
            prefill_seqs = self.consume_sequences(SequenceStatus.WAITING, self.max_num_seqs)
            if prefill_seqs:
                logits, context = self.executor.run_prefill(prefill_seqs)

                last_logits = self.executor.gather_prefill_last_logits(
                    logits,
                    context.cu_seqlens
                )
                    
                next_token_ids = self.executor.sample_next_tokens(last_logits)
                    
                for seq, next_token_id in zip(prefill_seqs, next_token_ids):
                    seq.output_token_ids.append(next_token_id)
                    
                    seq.num_computed_tokens = seq.prompt_len
                    seq.num_kv_tokens = seq.prompt_len
                    if self.should_stop(seq):
                        seq.status = SequenceStatus.FINISHED
                        self.finish_queue.append(seq)
                    else:
                        seq.status = SequenceStatus.PREFILLED
                        self.work_queue.append(seq)
                        
            decode_seqs = self.consume_sequences(SequenceStatus.DECODING, self.max_num_seqs)
            if len(decode_seqs) < self.max_num_seqs:
                decode_seqs.extend(self.consume_sequences(SequenceStatus.PREFILLED, self.max_num_seqs - len(decode_seqs)))
                
            if decode_seqs:
                logits = self.executor.run_decode_step(decode_seqs)
                next_token_ids = self.executor.sample_next_tokens(logits)
                for seq, next_token_id in zip(decode_seqs, next_token_ids):
                    seq.output_token_ids.append(next_token_id)
                    seq.num_computed_tokens += 1
                    seq.num_kv_tokens += 1
                    if self.should_stop(seq):
                        seq.status = SequenceStatus.FINISHED
                        self.finish_queue.append(seq)
                    else:
                        seq.status = SequenceStatus.DECODING
                        self.work_queue.append(seq)
                        
                self.executor.register_shared_blocks(decode_seqs)

        decode_texts = self.decode_sequences(self.finish_queue)   
        for seq, decode_text in zip(self.finish_queue, decode_texts):
            print_func(seq.prompt_text, decode_text)
            if seq.block_ids:
                self.executor.release_sequence(seq)
                seq.block_ids.clear()