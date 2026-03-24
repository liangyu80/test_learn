- [x] 讲一下RNN、Transformer的主要区别，以及Mamba是如何平衡RNN和Transformer的计算量
  → 见 `rnn_transformer_mamba.py`: 实现了 SimpleRNN/LSTM/Attention/MambaSSM 四种架构，附详细注释和对比实验
- [x] 搜集一下其他的Linear Transformer方法，比较一下和Mamba的不同
  → 见 `linear_attention.py`: 实现了 Linear Attention / RetNet / RWKV / GLA / Mamba 五种方法，附对比实验和总结
- [x] 仿造Qwen-Next模型，写一段代码展示Transformer和Mamba是如何结合在一个模型中
  → 见 `qwen_mamba_hybrid.py`: 仿 Qwen3 风格实现了 RoPE+GQA+SwiGLU Attention 层和 Mamba 层的混合模型，支持三种排列策略

