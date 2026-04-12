from dataclasses import dataclass
from math import sqrt
from typing import Any


@dataclass
class Qwen3Config:
    attention_bias: bool = False
    head_dim: int = 128
    hidden_size: int = 1024
    intermediate_size: int = 3072
    max_position_embeddings: int = 40960
    num_attention_heads: int = 16
    num_hidden_layers: int = 28
    num_key_value_heads: int = 8
    rms_norm_eps: float = 1e-6
    rope_theta: int = 1_000_000
    tie_word_embeddings: bool = True
    vocab_size: int = 151936

    @classmethod
    def from_hf_config(cls, hf_config) -> "Qwen3Config":
        rope_parameters = getattr(hf_config, "rope_parameters", None) or {}

        return cls(
            attention_bias=getattr(hf_config, "attention_bias", False),
            head_dim=getattr(hf_config, "head_dim", 128),
            hidden_size=getattr(hf_config, "hidden_size", 1024),
            intermediate_size=getattr(hf_config, "intermediate_size", 3072),
            max_position_embeddings=getattr(hf_config, "max_position_embeddings", 40960),
            num_attention_heads=getattr(hf_config, "num_attention_heads", 16),
            num_hidden_layers=getattr(hf_config, "num_hidden_layers", 28),
            num_key_value_heads=getattr(hf_config, "num_key_value_heads", 8),
            rms_norm_eps=getattr(hf_config, "rms_norm_eps", 1e-6),
            rope_theta=rope_parameters.get("rope_theta", getattr(hf_config, "rope_theta", 1_000_000)),
            tie_word_embeddings=getattr(hf_config, "tie_word_embeddings", True),
            vocab_size=getattr(hf_config, "vocab_size", 151936),
        )

    def to_model_kwargs(self) -> dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_heads": self.num_attention_heads,
            "head_dim": self.head_dim,
            "scale": 1.0 / sqrt(self.head_dim),
            "num_kv_heads": self.num_key_value_heads,
            "rms_norm_epsilon": self.rms_norm_eps,
            "qkv_bias": self.attention_bias,
            "base": self.rope_theta,
            "max_position": self.max_position_embeddings,
            "intermediate_size": self.intermediate_size,
            "ffn_bias": False,
            "num_layers": self.num_hidden_layers,
            "tie_word_embeddings": self.tie_word_embeddings,
        }
