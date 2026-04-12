from typing import Optional

import sys
from pathlib import Path

import torch
import torch.nn as nn

SRC_DIR = Path(__file__).resolve().parents[2]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from layers import *


class Qwen3Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, dtype=dtype))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, mode: str = "triton") -> torch.Tensor:
        if mode == "triton":
            return triton_embedding(x, self.weight)
        else:
            return torch_embedding(x, self.weight)


class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.eps = eps

    def forward(self, x: torch.Tensor, mode: str = "triton") -> torch.Tensor:
        if mode == "triton":
            return triton_rmsnorm(x, self.weight, eps=self.eps)
        else:
            return torch_rmsnorm(x, self.weight, eps=self.eps)


class Qwen3Linear(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(output_size, input_size, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_size, dtype=dtype))
        else:
            self.register_parameter("bias", None)
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

    def forward(self, x: torch.Tensor, mode: str = "triton") -> torch.Tensor:
        if mode == "triton":
            return triton_linear(x, self.weight, bias=self.bias)
        else:
            return torch_linear(x, self.weight, bias=self.bias)
        

class Qwen3RotaryEmbedding(nn.Module):
    def __init__(self, base: int, rotary_embedding: int, max_positions: int):
        super().__init__()
        self.base = base
        self.rotary_embedding = rotary_embedding
        self.max_position = max_positions
        self.register_buffer("cos_sin_cache", self._build_cache(), persistent=False)

    def _build_cache(self) -> torch.Tensor:
        dim = self.rotary_embedding
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        t = torch.arange(self.max_position, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        return torch.cat([cos, sin], dim=-1)

    def forward(
        self,
        positions: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        mode: str = "triton",
    ) -> torch.Tensor:
        if positions.dim() == 1:
            positions = positions.unsqueeze(0)
            
        if mode == "triton":
            q_out, k_out = triton_rotary_embedding(self.cos_sin_cache, positions, q, k)
        else:
            q_out, k_out = torch_rotary_embedding(self.cos_sin_cache, positions, q, k)
            
        return q_out, k_out


class Qwen3Silu(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, mode: str = "triton") -> torch.Tensor:
        if mode == "triton":
            return triton_silu(x)
        else:
            return torch_silu(x)
        
        
class Qwen3SiluMul(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor, mode: str = "triton") -> torch.Tensor:
        if mode == "triton":
            return triton_silumul(x, y)
        else:
            return torch_silumul(x, y)
        
        
class Qwen3SDPA(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float, mode: str = "triton") -> torch.Tensor:
        if mode == "triton":
            return triton_spda(q, k, v, scale=scale)
        else:
            return torch_spda(q, k, v, scale=scale)


class Qwen3Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        scale: float | None = None,
        num_kv_heads: int | None = None,
        rms_norm_epsilon: float = 1e-5,
        qkv_bias: bool = False,
        base: int = 10000,
        max_position: int = 16384,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.qkv_bias = qkv_bias

        self.q_proj = Qwen3Linear(hidden_size, num_heads * head_dim, bias=qkv_bias, dtype=dtype)
        self.k_proj = Qwen3Linear(hidden_size, self.num_kv_heads * head_dim, bias=qkv_bias, dtype=dtype)
        self.v_proj = Qwen3Linear(hidden_size, self.num_kv_heads * head_dim, bias=qkv_bias, dtype=dtype)

        self.q_norm = Qwen3RMSNorm(head_dim, rms_norm_epsilon, dtype=dtype)
        self.k_norm = Qwen3RMSNorm(head_dim, rms_norm_epsilon, dtype=dtype)
        
        self.spda = Qwen3SDPA()

        self.rotary_embed = Qwen3RotaryEmbedding(
            base=base,
            rotary_embedding=head_dim,
            max_positions=max_position,
        )

        self.o_proj = Qwen3Linear(num_heads * head_dim, hidden_size, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor, positions: torch.Tensor, mode: str = "triton") -> torch.Tensor:

        q = self.q_proj(x, mode=mode).view(x.size(0), x.size(1), self.num_heads, self.head_dim)
        k = self.k_proj(x, mode=mode).view(x.size(0), x.size(1), self.num_kv_heads, self.head_dim)
        v = self.v_proj(x, mode=mode).view(x.size(0), x.size(1), self.num_kv_heads, self.head_dim)

        if not self.qkv_bias:
            q = self.q_norm(q, mode=mode)
            k = self.k_norm(k, mode=mode)

        q, k = self.rotary_embed(positions, q, k, mode=mode)

        attn_out = self.spda(q, k, v, scale=self.scale)
        
        attn_out = attn_out.reshape(x.size(0), x.size(1), self.num_heads * self.head_dim)
        out = self.o_proj(attn_out, mode=mode)

        return out


class Qwen3MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = True,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.gate_proj = Qwen3Linear(hidden_size, intermediate_size, bias=bias, dtype=dtype)
        self.up_proj = Qwen3Linear(hidden_size, intermediate_size, bias=bias, dtype=dtype)
        self.silumul = Qwen3SiluMul()
        self.down_proj = Qwen3Linear(intermediate_size, hidden_size, bias=bias, dtype=dtype)

    def forward(self, x: torch.Tensor, mode: str = "triton") -> torch.Tensor:
        gate = self.gate_proj(x, mode=mode)
        up = self.up_proj(x, mode=mode)
        x = self.silumul(gate, up, mode=mode)
        x = self.down_proj(x, mode=mode)
        return x


class Qwen3DecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        scale: float | None = None,
        num_kv_heads: int | None = None,
        rms_norm_epsilon: float = 1e-5,
        qkv_bias: bool = False,
        base: int = 10000,
        max_position: int = 16384,
        intermediate_size: int = 4 * 1024,
        ffn_bias: bool = True,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.input_layernorm = Qwen3RMSNorm(hidden_size, rms_norm_epsilon, dtype=dtype)
        self.self_attn = Qwen3Attention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            scale=scale,
            num_kv_heads=num_kv_heads,
            rms_norm_epsilon=rms_norm_epsilon,
            qkv_bias=qkv_bias,
            base=base,
            max_position=max_position,
            dtype=dtype,
        )
        self.post_attention_layernorm = Qwen3RMSNorm(hidden_size, rms_norm_epsilon, dtype=dtype)
        self.mlp = Qwen3MLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            bias=ffn_bias,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor, mode: str = "triton") -> torch.Tensor:
        positions = torch.arange(x.size(1), device=x.device)

        residual = x
        x = self.input_layernorm(x, mode=mode)
        x = self.self_attn(x, positions=positions, mode=mode)
        
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x, mode=mode)
        x = self.mlp(x, mode=mode)
        x = residual + x

        return x


class Qwen3Model(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_heads: int,
        head_dim: int | None = None,
        scale: float | None = None,
        num_kv_heads: int | None = None,
        rms_norm_epsilon: float = 1e-5,
        qkv_bias: bool = False,
        base: int = 10000,
        max_position: int = 16384,
        intermediate_size: int = 4 * 1024,
        ffn_bias: bool = True,
        num_layers: int = 12,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        head_dim = head_dim if head_dim is not None else hidden_size // num_heads
        self.embed_tokens = Qwen3Embedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size,
            dtype=dtype,
        )
        self.layers = nn.ModuleList(
            [
                Qwen3DecoderLayer(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    scale=scale,
                    num_kv_heads=num_kv_heads,
                    rms_norm_epsilon=rms_norm_epsilon,
                    qkv_bias=qkv_bias,
                    base=base,
                    max_position=max_position,
                    intermediate_size=intermediate_size,
                    ffn_bias=ffn_bias,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = Qwen3RMSNorm(hidden_size, eps=rms_norm_epsilon, dtype=dtype)

    def forward(self, input_ids: torch.Tensor, mode: str = "triton") -> torch.Tensor:

        x = self.embed_tokens(input_ids, mode=mode)
        for layer in self.layers:
            x = layer(x, mode=mode)
        x = self.norm(x, mode=mode)
        return x


class Qwen3LMHead(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, dtype=dtype))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, mode: str = "triton") -> torch.Tensor:
        if mode == "triton":
            return triton_linear(x, self.weight, bias=None)
        else:
            return torch_linear(x, self.weight, bias=None)
        
        
class Qwen3ForCausalLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_heads: int,
        head_dim: int | None = None,
        scale: float | None = None,
        num_kv_heads: int | None = None,
        rms_norm_epsilon: float = 1e-5,
        qkv_bias: bool = False,
        base: int = 10000,
        max_position: int = 16384,
        intermediate_size: int = 4 * 1024,
        ffn_bias: bool = True,
        num_layers: int = 12,
        tie_word_embeddings: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.model = Qwen3Model(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim if head_dim is not None else hidden_size // num_heads,
            scale=scale,
            num_kv_heads=num_kv_heads,
            rms_norm_epsilon=rms_norm_epsilon,
            qkv_bias=qkv_bias,
            base=base,
            max_position=max_position,
            intermediate_size=intermediate_size,
            ffn_bias=ffn_bias,
            num_layers=num_layers,
            dtype=dtype,
        )
        self.lm_head = Qwen3LMHead(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size,
            dtype=dtype,
        )
        if tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor, mode: str = "triton") -> torch.Tensor:
        hidden_states = self.model(input_ids, mode=mode)
        return self.lm_head(hidden_states, mode=mode)



     
