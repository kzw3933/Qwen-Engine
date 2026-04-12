import sys
from pathlib import Path

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoTokenizer


ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.qwen3.config import Qwen3Config
from models.qwen3.model import Qwen3ForCausalLM
from models.qwen3.utils import load_weights_from_checkpoint


def greedy_generate(model: nn.Module, tokenizer, prompt_text: str, device: str, max_new_tokens: int = 16):
    input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids
    generated_ids = input_ids

    print("=" * 88)
    print(f"Prompt: {prompt_text}")
    print("=" * 88)

    model.eval()
    
    model = model.to(device)
    generated_ids = generated_ids.to(device)
    
    with torch.no_grad():
        for step in range(max_new_tokens):
            logits = model(generated_ids, mode="triton")
            next_token_logits = logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)

            decoded_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            print(f"[step {step + 1:02d}] {decoded_text}")

            if (
                tokenizer.eos_token_id is not None
                and next_token_id.item() == tokenizer.eos_token_id
            ):
                break

    print("=" * 88)
    print("Final text:")
    print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
    print("=" * 88)


def main():
    model_name_or_path = "Qwen/Qwen3-0.6B"
    prompt_text = "介绍一下Qwen模型"
    max_new_tokens = 64
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dtype = torch.bfloat16 if device == "cuda" else torch.float32

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
    load_weights_from_checkpoint(model, model_name_or_path)

    greedy_generate(model, tokenizer, prompt_text, device=device, max_new_tokens=max_new_tokens)


if __name__ == "__main__":
    main()
