# Qwen3-Engine

一个基于 Triton 自定义算子的轻量级 **Qwen3-0.6B** 推理框架项目。项目目标是以 **Qwen3-0.6B** 为参考对象，从模型结构分析出发，逐步实现一个可运行、可验证、可扩展的推理系统，并重点探索 LLM 推理中的核心算子优化与调度机制。本项目聚焦于 **单卡、单机、消费级显卡可运行的推理框架原型**。

---

## 项目背景

大语言模型推理系统的优化重点逐渐从“模型能否跑起来”转向“如何以更高吞吐、更低时延、更低显存占用运行”。

**Qwen3-0.6B 是稠密 Transformer**，推理路径更简洁，但仍包含大量关键算子与内存带宽瓶颈，例如：

- RMSNorm
- RoPE
- QKV 投影与张量布局变换
- Attention Score / Softmax / Value 聚合
- KV Cache 写入与读取
- MLP (含门控激活，如 SwiGLU/GEGLU)

本项目希望通过 **从零拆解 Qwen3-0.6B 并手写关键 Triton Kernel** 的方式，系统理解并实现一个“小型推理框架”。

---

## 项目目标

1. **理解 Qwen3-0.6B 模型的推理执行路径**
   - 熟悉模型结构、模块连接关系以及权重组织方式
   - 理解 Prefill / Decode 阶段的计算差异

2. **实现 LLM 推理中的核心算子**
   - 使用 Triton 手写关键 kernel
   - 对比 PyTorch reference 实现，确保数值正确性
   - 分析算子的访存模式、并行粒度与性能瓶颈

3. **实现一个最小可运行的推理框架**
   - 支持基础的 Prefill / Decode
   - 支持 KV Cache 管理
   - 支持基础的序列管理与调度机制

4. **探索现代推理系统中的关键工程问题**
   - Chunked Prefill
   - FlashAttention
   - PagedAttention / Flash Decoding
   - 序列管理与 Block Table
   - Decode 阶段的内存带宽瓶颈

---

## 模型规格（当前配置）

- 模型：`Qwen/Qwen3-0.6B`
- 层数：28
- 隐藏维度：1024
- 注意力头数：16
- KV 头数：8（GQA）
- Head Dim：128
- MLP 中间维度：3072
- 词表大小：151936
- 最大位置：32768
- RoPE Theta：1000000
- RMSNorm Epsilon：1e-6
- 激活：SwiGLU（`SiluAndMul`）
- QKV Bias：False
- MLP Bias：False
- Tie Word Embeddings：True

---

## 项目实施流程

### 阶段 1：模型结构分析

首先拉取并分析 `Qwen3-0.6B`，重点理解：

- 模型整体结构
- Decoder Layer 的组成
- Attention 模块的输入输出形状
- MLP 的结构（激活函数、投影层、张量布局）
- 权重格式与参数命名规则
- Prefill / Decode 阶段的计算差异

---

### 阶段 2：核心算子分析与拆解

在明确模型结构后，进一步分析推理过程中的关键算子，并确定哪些部分适合用 Triton 自定义实现。

初步重点包括：

- RMSNorm
- RoPE
- QKV 投影后的张量重排 / layout transform
- Attention Score / Softmax / Value 聚合
- KV Cache 写入与读取
- MLP（含门控激活）

---

### 阶段 3：算子实现层

在这一阶段开始逐步实现 Triton Kernel，并与 PyTorch Reference 对齐。

优先顺序计划如下：

1. **基础算子**
   - RMSNorm
   - RoPE
   - 基础张量重排 / layout transform

2. **Attention 相关**
   - Prefill 路径中的 FlashAttention 风格实现
   - Decode 路径中的 PagedAttention / Flash Decoding 风格实现
   - KV Cache 写入与读取

3. **MLP 相关**
   - 门控激活融合（如 SwiGLU / GEGLU）
   - 投影层融合

---

### 阶段 4：模型执行与调度层

在核心算子具备之后，开始搭建推理执行层，使得模型能够从“若干算子”变成“完整推理流程”。

这一层计划包括：

- 模型 forward 执行流程
- Prefill / Decode 分支
- Layer-by-Layer 执行调度
- KV Cache 生命周期管理
- Block Table / Slot Mapping 管理
- Sequence State 管理

---

### 阶段 5：序列管理与系统机制

在具备基本可运行推理之后，进入更偏“推理系统”层的问题，包括：

- 多序列管理
- Continuous Batching
- Chunked Prefill
- 请求状态流转
- Block 分配与回收
- 上下文长度管理
- 请求生命周期管理

---

## 备注

- 项目以 **Qwen3-0.6B** 为主目标，但后续可以扩展到更大规模或同系列变体。
- 若你希望 README 中补充更精确的层数、维度、头数、RoPE 配置等结构信息，请提供对应的模型架构细节，我可以进一步完善。
