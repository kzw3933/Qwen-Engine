import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer


ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models import Qwen3ForCausalLM, Qwen3Config, load_weights_from_checkpoint
from engine import KVCachePool, LLMEngine

def main():
    model_name_or_path = "Qwen/Qwen3-0.6B"
    
    max_new_tokens = 64
    max_num_seqs = 4
    max_model_len = 256
    block_size = 16
    num_blocks = max_num_seqs * ((max_model_len + block_size - 1) // block_size)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    runtime_mode = "triton"

    hf_config = AutoConfig.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )
    config = Qwen3Config.from_hf_config(hf_config)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )

    model = Qwen3ForCausalLM(**config.to_model_kwargs(), dtype=model_dtype)
    load_weights_from_checkpoint(model, model_name_or_path, silent=True)
    model = model.to(device)
    model.eval()
    
    kv_cache_pool = KVCachePool(
        num_layers=config.num_hidden_layers,
        num_blocks=num_blocks,
        block_size=block_size,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        dtype=model_dtype,
        device=torch.device(device),
    )
    
    engine = LLMEngine(
        model=model,
        tokenizer=tokenizer,
        kv_cache_pool=kv_cache_pool,
        max_num_seqs=max_num_seqs,
        device=torch.device(device),
        mode=runtime_mode,
    )
    
    prompt_texts = [
        "介绍一下Qwen模型",
        "How can remote teams improve trust, focus, communication, and delivery speed without adding more meetings? Please include practical examples that small teams can apply immediately.",
        "How can remote teams improve trust, focus, communication, and delivery speed without adding",
    ]
    
    engine.submit(
        prompt_texts=prompt_texts, max_new_tokens=max_new_tokens
    )
    
    def print_info(prompt_text, output_text):
        print("=" * 88)
        print(f"Prompt: {prompt_text}")
        print("=" * 88)
        print(output_text)
        print("=" * 88)
    
    
    engine.serving(print_func=print_info)
    

if __name__ == "__main__":
    main()
