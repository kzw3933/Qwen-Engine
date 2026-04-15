# Qwen3-Engine

一个基于 Triton 自定义算子的轻量级 **Qwen3-0.6B** 推理框架原型，聚焦于单卡场景下的算子实现、KV Cache 管理与基础调度机制验证。

## 当前能力

- 支持 **Qwen3-0.6B** 的基础推理流程
- 提供 Triton 版本的核心算子实现，包括：
  - Embedding
  - RMSNorm
  - Linear
  - RoPE
  - SiLU / SiLU-Mul
  - Prefill Flash Attention
  - Decode Flash Decoding
- 支持部分算子融合
- 支持 **Paged Attention** 所需的 KV Cache / Block Table 管理
- 支持 **Prefix Sharing**
- 提供对应的 PyTorch reference 路径，便于正确性对照

## 当前范围

当前实现重点是验证一套最小可运行的推理框架：

- Prefill / Decode 分阶段执行
- Page KV Cache 管理
- 多序列基础调度
- Prefix Cache 复用

## 暂未覆盖

- Chunked Prefill
- Prefill / Decode 混合批调度
- Prefill / Decode 算子进一步统一
- 更完整的连续批处理与吞吐优化

## 模型配置

- 模型：`Qwen/Qwen3-0.6B`
- 层数：28
- 隐藏维度：1024
- 注意力头数：16
- KV 头数：8
- Head Dim：128
- MLP 中间维度：3072
- 词表大小：151936
- 最大位置：32768

## 项目定位

这个项目当前更偏向于一个用于学习、验证和迭代推理系统关键机制的原型实现，重点关注：

- Triton 自定义算子实现
- Flash Attention / Flash Decoding
- Paged Attention
- Prefix Sharing
- 推理调度与 KV Cache 生命周期管理
