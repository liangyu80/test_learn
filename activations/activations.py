"""
激活函数全面对比 —— 从经典到前沿

本文件实现并对比 20+ 种激活函数，涵盖:
    1. 经典激活函数:  Sigmoid, Tanh, ReLU
    2. ReLU 改进:     LeakyReLU, PReLU, ELU, SELU
    3. Transformer 时代: GELU, SiLU/Swish, Mish
    4. GLU 门控变体:  GLU, ReGLU, GeGLU, SwiGLU
    5. 其他实用函数:  Softplus, Softsign, Hardswish, Hardtanh, Squareplus

重点:
    - 每个函数都有完整的数学公式和中文注释
    - 提供前向/导数可视化
    - 分析各函数的数值性质 (梯度消失/爆炸、零中心、单调性等)
    - 分析在大语言模型 (LLM) 中的使用情况
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional, Dict, Tuple


# ==============================================================================
# 1. 经典激活函数
# ==============================================================================

class Sigmoid(nn.Module):
    """
    Sigmoid 激活函数

    公式: σ(x) = 1 / (1 + e^(-x))
    值域: (0, 1)
    导数: σ'(x) = σ(x) · (1 - σ(x))

    特点:
        ✓ 输出范围有界 (0,1)，适合概率输出
        ✗ 梯度消失: 当 |x| 较大时导数趋近于 0
        ✗ 非零中心: 输出恒正，导致梯度更新 zig-zag
        ✗ 指数运算较慢

    历史: 最早使用的激活函数之一，现在主要用于二分类输出层和门控机制
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x)


class Tanh(nn.Module):
    """
    双曲正切 (Tanh) 激活函数

    公式: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    等价: tanh(x) = 2·σ(2x) - 1
    值域: (-1, 1)
    导数: tanh'(x) = 1 - tanh²(x)

    特点:
        ✓ 零中心输出，优于 Sigmoid
        ✓ 梯度比 Sigmoid 更强 (最大值为 1 vs 0.25)
        ✗ 仍有梯度消失问题
        ✗ 计算成本高

    使用: LSTM/GRU 的门控机制、早期 RNN
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x)


class ReLU(nn.Module):
    """
    修正线性单元 (Rectified Linear Unit)

    公式: ReLU(x) = max(0, x)
    值域: [0, +∞)
    导数: ReLU'(x) = 1 if x > 0, else 0

    特点:
        ✓ 计算极快 (只是 max 操作)
        ✓ x > 0 时梯度恒为 1，无梯度消失
        ✓ 稀疏激活 (大约 50% 的神经元输出为 0)
        ✗ "死亡 ReLU" 问题: x < 0 时梯度永远为 0
        ✗ 非零中心
        ✗ 无界输出，可能导致数值不稳定

    历史: 2012 年 AlexNet 首次大规模使用，至今仍是最常用的激活函数之一
    使用: 原始 Transformer (Vaswani et al., 2017) FFN 层使用 ReLU
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x)


# ==============================================================================
# 2. ReLU 改进家族
# ==============================================================================

class LeakyReLU(nn.Module):
    """
    泄漏 ReLU (Leaky ReLU)

    公式: LeakyReLU(x) = x if x > 0, else α·x   (α 通常为 0.01)
    值域: (-∞, +∞)
    导数: 1 if x > 0, else α

    特点:
        ✓ 解决了 "死亡 ReLU" 问题 (负区间有小梯度)
        ✓ 计算快
        ✗ α 是超参数，需要调优
        ✗ 负区间的梯度可能不够大
    """
    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(x, self.negative_slope)


class PReLU(nn.Module):
    """
    参数化 ReLU (Parametric ReLU)

    公式: PReLU(x) = x if x > 0, else α·x   (α 为可学习参数)
    来源: "Delving Deep into Rectifiers" (He et al., 2015)

    特点:
        ✓ α 可以通过反向传播学习
        ✓ 可以为每个通道学习不同的 α
        ✗ 容易过拟合 (多了可学习参数)
    """
    def __init__(self, num_parameters: int = 1, init: float = 0.25):
        super().__init__()
        self.prelu = nn.PReLU(num_parameters, init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.prelu(x)


class ELU(nn.Module):
    """
    指数线性单元 (Exponential Linear Unit)

    公式: ELU(x) = x if x > 0, else α·(e^x - 1)
    值域: (-α, +∞)
    来源: "Fast and Accurate Deep Network Learning by ELU" (Clevert et al., 2015)

    特点:
        ✓ 负值区有平滑的非零输出，均值接近零
        ✓ 对噪声更鲁棒
        ✗ 负值区需要指数运算，较慢
        ✗ 无界 (正半轴)
    """
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.elu(x, self.alpha)


class SELU(nn.Module):
    """
    缩放指数线性单元 (Scaled Exponential Linear Unit)

    公式: SELU(x) = λ · (x if x > 0, else α·(e^x - 1))
    其中 λ ≈ 1.0507, α ≈ 1.6733 (精确计算的自归一化常数)
    来源: "Self-Normalizing Neural Networks" (Klambauer et al., 2017)

    特点:
        ✓ 自归一化: 均值趋向 0, 方差趋向 1 (无需 BatchNorm)
        ✓ 理论上可避免梯度消失/爆炸
        ✗ 要求使用 LeCun 初始化 + AlphaDropout
        ✗ 对架构有严格要求 (全连接网络)
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.selu(x)


# ==============================================================================
# 3. Transformer 时代的激活函数
# ==============================================================================

class GELU(nn.Module):
    """
    高斯误差线性单元 (Gaussian Error Linear Unit)

    公式: GELU(x) = x · Φ(x) = x · 0.5 · (1 + erf(x/√2))
    近似: GELU(x) ≈ 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))
    来源: "Gaussian Error Linear Units" (Hendrycks & Gimpel, 2016)

    特点:
        ✓ 平滑且非单调 (在 x ≈ -0.17 处有微小负值)
        ✓ 概率解释: 以 x 的 CDF 概率保留输入
        ✓ 比 ReLU 更平滑，梯度更稳定
        ✗ 计算成本较高

    使用:
        - BERT (2018): 首个大规模采用 GELU 的模型
        - GPT 系列: GPT-2, GPT-3 均使用 GELU
        - ViT: Vision Transformer 使用 GELU
        - 现在已被 SwiGLU 逐步取代
    """
    def __init__(self, approximate: bool = False):
        super().__init__()
        self.approximate = approximate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.approximate:
            # tanh 近似版本 (更快但精度略低)
            return 0.5 * x * (1.0 + torch.tanh(
                math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))
            ))
        return F.gelu(x)


class SiLU(nn.Module):
    """
    Sigmoid 线性单元 / Swish 激活函数

    公式: SiLU(x) = x · σ(x) = x / (1 + e^(-x))
    等价: Swish(x) = x · σ(βx)  (β=1 时为 SiLU)
    来源:
        - "Sigmoid-Weighted Linear Units" (Elfwing et al., 2017)
        - "Searching for Activation Functions" (Ramachandran et al., 2017, Google Brain)

    特点:
        ✓ 平滑、非单调
        ✓ 自门控: 用输入自身的 sigmoid 来调制输入
        ✓ 优于 ReLU (通过 NAS 搜索发现)
        ✗ 计算成本稍高

    使用:
        - SwiGLU 的核心组件 (SwiGLU = SiLU + GLU)
        - LLaMA / PaLM / Mistral 等通过 SwiGLU 间接使用
        - EfficientNet 等视觉模型
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x)  # x * sigmoid(x)


class Mish(nn.Module):
    """
    Mish 激活函数

    公式: Mish(x) = x · tanh(softplus(x)) = x · tanh(ln(1 + e^x))
    来源: "Mish: A Self Regularized Non-Monotonic Activation Function" (Misra, 2019)

    特点:
        ✓ 平滑、非单调
        ✓ 无上界，有下界 (≈ -0.31)
        ✓ 在某些视觉任务上优于 ReLU 和 Swish
        ✗ 计算成本高 (tanh + softplus)
        ✗ 在 LLM 中使用较少

    使用: YOLOv4, YOLOv5 等目标检测模型
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.mish(x)  # x * tanh(softplus(x))


# ==============================================================================
# 4. GLU 门控变体 —— 大模型时代的主角
# ==============================================================================

class GLU(nn.Module):
    """
    门控线性单元 (Gated Linear Unit)

    公式: GLU(x) = (xW + b) ⊙ σ(xV + c)
    来源: "Language Modeling with Gated Convolutional Networks" (Dauphin et al., 2017)

    核心思想:
        将输入映射为两个等大小的向量:
        - 一个作为"内容" (xW + b)
        - 另一个通过 sigmoid 作为"门控" σ(xV + c)
        两者逐元素相乘 → 门控控制信息流

    特点:
        ✓ 门控机制可学习选择性地传递信息
        ✓ 梯度路径更优 (类似残差连接的效果)
        ✗ 参数量翻倍 (需要两个线性层)
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        # 两个线性层: 一个产生内容, 一个产生门控信号
        self.linear = nn.Linear(in_features, out_features * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 分割为内容和门控两部分
        content, gate = self.linear(x).chunk(2, dim=-1)
        return content * torch.sigmoid(gate)


class ReGLU(nn.Module):
    """
    ReLU 门控线性单元

    公式: ReGLU(x) = (xW + b) ⊙ ReLU(xV + c)
    来源: "GLU Variants Improve Transformer" (Shazeer, 2020)

    用 ReLU 替代 GLU 中的 Sigmoid 作为门控激活函数。
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        content, gate = self.linear(x).chunk(2, dim=-1)
        return content * F.relu(gate)


class GeGLU(nn.Module):
    """
    GELU 门控线性单元

    公式: GeGLU(x) = (xW + b) ⊙ GELU(xV + c)
    来源: "GLU Variants Improve Transformer" (Shazeer, 2020)

    用 GELU 替代 GLU 中的 Sigmoid 作为门控激活函数。

    特点:
        ✓ GELU 的平滑性 + 门控的选择性
        ✓ 在 Shazeer 的实验中与 SwiGLU 并列最佳
        ✗ 被 SwiGLU 在工业界取代

    使用: 部分研究模型采用
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        content, gate = self.linear(x).chunk(2, dim=-1)
        return content * F.gelu(gate)


class SwiGLU(nn.Module):
    """
    Swish 门控线性单元 —— 当前大模型的标准选择

    公式: SwiGLU(x) = (xW + b) ⊙ SiLU(xV + c)
         其中 SiLU(z) = z · σ(z) (即 Swish)
    来源: "GLU Variants Improve Transformer" (Shazeer, 2020)

    核心思想:
        在 GLU 框架下，用 SiLU/Swish 作为门控激活函数。
        结合了:
        1. GLU 的门控机制 (选择性信息传递)
        2. SiLU/Swish 的平滑非单调性
        3. 自门控特性 (SiLU 本身也是一种门控)

    参数量调整:
        标准 FFN: x → Linear(d, 4d) → ReLU → Linear(4d, d)
        SwiGLU:   x → [Linear(d, 4d/3*2)] → SwiGLU → Linear(4d/3, d)
        为保持参数量相当，隐藏层维度设为 4d * 2/3

    使用 (当前最主流):
        - LLaMA / LLaMA-2 / LLaMA-3 (Meta)
        - PaLM / PaLM-2 (Google)
        - Mistral / Mixtral (Mistral AI)
        - DeepSeek-V2 / V3 (DeepSeek)
        - Qwen / Qwen-2 (阿里)
        - Gemma (Google)
        - Yi (零一万物)
        几乎所有 2023 年后的开源大模型都采用 SwiGLU
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        # 两个独立的线性层 (LLaMA 的实际实现方式)
        self.w1 = nn.Linear(in_features, out_features, bias=False)  # 门控
        self.w2 = nn.Linear(in_features, out_features, bias=False)  # 内容

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.w1(x)) * self.w2(x)


# ==============================================================================
# 5. 其他实用激活函数
# ==============================================================================

class Softplus(nn.Module):
    """
    Softplus 激活函数

    公式: Softplus(x) = ln(1 + e^x)
    值域: (0, +∞)
    导数: Softplus'(x) = σ(x) (即 Sigmoid!)

    特点:
        ✓ ReLU 的平滑近似
        ✓ 输出恒正，适合建模方差、正值参数
        ✗ 计算较慢
        ✗ 梯度仍然会饱和

    使用: VAE 中的方差输出、某些正则化项
    """
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x, self.beta)


class Softsign(nn.Module):
    """
    Softsign 激活函数

    公式: Softsign(x) = x / (1 + |x|)
    值域: (-1, 1)
    导数: Softsign'(x) = 1 / (1 + |x|)²

    特点:
        ✓ 类似 Tanh 但饱和更慢 (多项式 vs 指数)
        ✓ 计算快 (无指数运算)
        ✗ 使用较少
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softsign(x)


class Hardswish(nn.Module):
    """
    Hard Swish 激活函数

    公式: Hardswish(x) = x · ReLU6(x+3) / 6
    来源: "Searching for MobileNetV3" (Howard et al., 2019)

    特点:
        ✓ SiLU/Swish 的高效近似
        ✓ 分段线性，适合移动端部署
        ✗ 近似精度有限
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.hardswish(x)


class Squareplus(nn.Module):
    """
    Squareplus 激活函数

    公式: Squareplus(x) = (x + √(x² + b)) / 2   (b > 0, 通常 b = 4)
    来源: "Squareplus: A Softplus-Like Algebraic Rectifier" (Barron, 2021)

    特点:
        ✓ Softplus 的代数近似 (无指数运算)
        ✓ 计算极快，适合资源受限环境
        ✓ 处处可导
        ✗ 精度略低于 Softplus
    """
    def __init__(self, b: float = 4.0):
        super().__init__()
        self.b = b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * (x + torch.sqrt(x * x + self.b))


# ==============================================================================
# 激活函数注册表 —— 统一管理所有激活函数
# ==============================================================================

# 逐元素激活函数 (不含可学习参数，不改变形状)
ELEMENTWISE_ACTIVATIONS: Dict[str, nn.Module] = {
    # 经典
    "Sigmoid":    Sigmoid(),
    "Tanh":       Tanh(),
    "ReLU":       ReLU(),
    # ReLU 改进
    "LeakyReLU":  LeakyReLU(0.01),
    "PReLU":      PReLU(),
    "ELU":        ELU(),
    "SELU":       SELU(),
    # Transformer 时代
    "GELU":       GELU(),
    "GELU(approx)": GELU(approximate=True),
    "SiLU/Swish": SiLU(),
    "Mish":       Mish(),
    # 其他
    "Softplus":   Softplus(),
    "Softsign":   Softsign(),
    "Hardswish":  Hardswish(),
    "Squareplus": Squareplus(),
}

# GLU 门控变体 (包含线性层，改变维度)
def get_glu_activations(in_features: int = 128, out_features: int = 64) -> Dict[str, nn.Module]:
    """创建所有 GLU 变体的实例。"""
    return {
        "GLU":    GLU(in_features, out_features),
        "ReGLU":  ReGLU(in_features, out_features),
        "GeGLU":  GeGLU(in_features, out_features),
        "SwiGLU": SwiGLU(in_features, out_features),
    }


# ==============================================================================
# 分析工具函数
# ==============================================================================

def compute_properties(act_fn: nn.Module, x: torch.Tensor) -> Dict:
    """
    计算激活函数的关键数值性质。

    返回:
        - output_mean: 输出均值 (越接近 0 越好，说明零中心)
        - output_std:  输出标准差
        - sparsity:    稀疏度 (输出为 0 的比例)
        - grad_mean:   梯度均值 (越接近 1 越好)
        - grad_std:    梯度标准差
        - dead_ratio:  "死亡" 比例 (梯度为 0 的比例)
        - bounded:     是否有界
    """
    x = x.clone().requires_grad_(True)
    y = act_fn(x)
    # 计算梯度
    y.sum().backward()
    grad = x.grad

    properties = {
        "output_mean": y.mean().item(),
        "output_std": y.std().item(),
        "sparsity": (y.abs() < 1e-6).float().mean().item(),
        "grad_mean": grad.mean().item(),
        "grad_std": grad.std().item(),
        "dead_ratio": (grad.abs() < 1e-6).float().mean().item(),
        "bounded": bool(y.min() > -100 and y.max() < 100),
    }
    return properties


def analyze_gradient_flow(act_fn: nn.Module, depth: int = 10, dim: int = 128) -> Dict:
    """
    模拟多层网络中的梯度流。

    构建 depth 层全连接网络，使用给定激活函数，
    检测梯度是否消失或爆炸。
    """
    layers = []
    for _ in range(depth):
        layers.append(nn.Linear(dim, dim))
        layers.append(act_fn)

    model = nn.Sequential(*layers)

    x = torch.randn(32, dim)
    x.requires_grad_(True)

    # 前向
    y = model(x)
    loss = y.sum()
    loss.backward()

    # 收集每层线性层的梯度范数
    grad_norms = []
    for layer in model:
        if isinstance(layer, nn.Linear) and layer.weight.grad is not None:
            grad_norms.append(layer.weight.grad.norm().item())

    if len(grad_norms) == 0:
        return {"grad_norms": [], "vanishing": True, "exploding": False}

    return {
        "grad_norms": grad_norms,
        "vanishing": grad_norms[-1] < 1e-6 if grad_norms else True,
        "exploding": grad_norms[0] > 1e6 if grad_norms else False,
        "ratio_first_last": grad_norms[0] / max(grad_norms[-1], 1e-12),
    }


# ==============================================================================
# FFN 模块 —— 对比不同激活函数在 Transformer FFN 中的效果
# ==============================================================================

class TransformerFFN(nn.Module):
    """
    标准 Transformer FFN (Feed-Forward Network) 模块。

    标准结构:
        FFN(x) = W₂ · Activation(W₁ · x + b₁) + b₂
        维度: d_model → d_ff → d_model

    SwiGLU 结构:
        FFN(x) = W₃ · (SiLU(W₁ · x) ⊙ W₂ · x)
        维度: d_model → d_ff*2/3 → d_model (保持参数量一致)
    """
    def __init__(self, d_model: int, d_ff: int, activation: str = "relu",
                 dropout: float = 0.1):
        super().__init__()
        self.activation_name = activation

        if activation in ("swiglu", "geglu", "reglu", "glu"):
            # GLU 变体: 隐藏层调整为 2/3 以保持参数量
            adjusted_ff = int(d_ff * 2 / 3)
            if activation == "swiglu":
                self.gate = SwiGLU(d_model, adjusted_ff)
            elif activation == "geglu":
                self.gate = GeGLU(d_model, adjusted_ff)
            elif activation == "reglu":
                self.gate = ReGLU(d_model, adjusted_ff)
            else:
                self.gate = GLU(d_model, adjusted_ff)
            self.w_out = nn.Linear(adjusted_ff, d_model)
        else:
            # 标准结构
            self.w_in = nn.Linear(d_model, d_ff)
            self.w_out = nn.Linear(d_ff, d_model)
            act_map = {
                "relu": nn.ReLU(),
                "gelu": nn.GELU(),
                "silu": nn.SiLU(),
                "mish": nn.Mish(),
                "elu": nn.ELU(),
                "leaky_relu": nn.LeakyReLU(0.01),
                "tanh": nn.Tanh(),
            }
            self.act = act_map.get(activation, nn.ReLU())

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation_name in ("swiglu", "geglu", "reglu", "glu"):
            return self.w_out(self.dropout(self.gate(x)))
        else:
            return self.w_out(self.dropout(self.act(self.w_in(x))))


# ==============================================================================
# Demo
# ==============================================================================

def demo_activations():
    """演示所有激活函数的基本性质。"""
    print("=" * 70)
    print("  激活函数全面对比 Demo")
    print("=" * 70)

    # 生成测试数据: 标准正态分布
    torch.manual_seed(42)
    x = torch.randn(10000)

    print("\n▶ 1. 逐元素激活函数基本性质")
    print("-" * 70)
    header = f"{'激活函数':<16} {'输出均值':>8} {'输出标准差':>10} {'稀疏度':>8} {'梯度均值':>8} {'死亡率':>8}"
    print(header)
    print("-" * 70)

    for name, act_fn in ELEMENTWISE_ACTIVATIONS.items():
        props = compute_properties(act_fn, x.clone())
        print(f"{name:<16} {props['output_mean']:>8.4f} {props['output_std']:>10.4f} "
              f"{props['sparsity']:>8.2%} {props['grad_mean']:>8.4f} {props['dead_ratio']:>8.2%}")

    # 梯度流分析
    print("\n▶ 2. 梯度流分析 (10 层全连接网络)")
    print("-" * 70)
    test_acts = {
        "ReLU":      ReLU(),
        "LeakyReLU": LeakyReLU(0.01),
        "GELU":      GELU(),
        "SiLU/Swish": SiLU(),
        "Tanh":      Tanh(),
        "Sigmoid":   Sigmoid(),
        "ELU":       ELU(),
        "Mish":      Mish(),
    }

    for name, act_fn in test_acts.items():
        result = analyze_gradient_flow(act_fn, depth=10, dim=64)
        status = ""
        if result["vanishing"]:
            status = "⚠️ 梯度消失"
        elif result["exploding"]:
            status = "⚠️ 梯度爆炸"
        else:
            status = "✓ 正常"
        norms = result["grad_norms"]
        if norms:
            print(f"  {name:<14}: 第1层梯度={norms[0]:.4f}, "
                  f"最后层梯度={norms[-1]:.6f}, "
                  f"首/末比={result['ratio_first_last']:.2f}  {status}")

    # FFN 对比
    print("\n▶ 3. Transformer FFN 对比")
    print("-" * 70)
    d_model, d_ff, batch, seq = 128, 512, 16, 32

    ffn_types = ["relu", "gelu", "silu", "swiglu", "geglu", "reglu", "mish"]
    x = torch.randn(batch, seq, d_model)

    for name in ffn_types:
        ffn = TransformerFFN(d_model, d_ff, activation=name, dropout=0.0)
        params = sum(p.numel() for p in ffn.parameters())
        out = ffn(x)
        print(f"  {name:<10}: 参数量={params:>7,d}, 输出形状={list(out.shape)}, "
              f"输出均值={out.mean():.4f}, 输出标准差={out.std():.4f}")

    print("\n✅ Demo 完成!")


if __name__ == "__main__":
    demo_activations()
