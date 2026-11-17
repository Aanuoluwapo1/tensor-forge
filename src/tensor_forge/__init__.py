"""tensor-forge: Reverse-Mode Autodiff & Numerical Stability Engine."""

from tensor_forge.gradcheck import numerical_gradient_check
from tensor_forge.nn import (
    MLP,
    LayerNorm,
    Linear,
    Module,
    Parameter,
    RMSNorm,
    TransformerBlock,
)
from tensor_forge.optim import SGD, AdamW, clip_grad_norm_
from tensor_forge.stable_ops import (
    cross_entropy_loss,
    layer_norm,
    log_softmax,
    mse_loss,
    rms_norm,
    scaled_dot_product_attention,
)
from tensor_forge.tensor import Tensor, unbroadcast

__version__ = "0.1.0"

__all__ = [
    "MLP",
    "SGD",
    "AdamW",
    "LayerNorm",
    "Linear",
    "Module",
    "Parameter",
    "RMSNorm",
    "Tensor",
    "TransformerBlock",
    "clip_grad_norm_",
    "cross_entropy_loss",
    "layer_norm",
    "log_softmax",
    "mse_loss",
    "numerical_gradient_check",
    "rms_norm",
    "scaled_dot_product_attention",
    "unbroadcast",
]
