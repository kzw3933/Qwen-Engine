# Qwen3-0.6B 模型架构（项目配置）

本文档汇总当前项目配置中使用的 `Qwen/Qwen3-0.6B` 模型常量，作为实现与验证时的对照依据。

## 核心规格

- 模型：`Qwen/Qwen3-0.6B`
- 层数：28
- 隐藏维度：1024
- 注意力头数：16
- KV 头数：8（GQA）
- Head Dim：128
- MLP 中间维度：3072
- 词表大小：151936
- 最大位置（RoPE）：40960
- RoPE Theta（base）：1000000
- RMSNorm Epsilon：1e-6
- 激活：SwiGLU
- QKV Bias：False
- MLP Bias：False
- Tie Word Embeddings：True

## Attention Block（每层）

- 输入 RMSNorm
- QKV 投影
- 当 `qkv_bias=False` 时，对 Q / K 做逐头归一化
- 对 Q / K 施加 RoPE
- 注意力计算（GQA，`num_heads=16`，`num_kv_heads=8`）
- 输出投影
- 注意力后 RMSNorm + 残差

## MLP Block（每层）

- Gate/Up 投影
- SwiGLU 激活
- Down 投影

## 执行备注

- Prefill 与 Decode 的 position 生成逻辑不同。
- KV Cache 布局与 Block Table 行为遵循运行时配置中的 `block_size=256`。
- 运行时 `max_model_length=128` 是系统约束，不是模型硬限制。
