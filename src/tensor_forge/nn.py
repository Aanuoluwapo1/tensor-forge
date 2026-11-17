"""Stateful neural network modules and parameter containers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, DTypeLike

from tensor_forge.stable_ops import (
    layer_norm,
    rms_norm,
    scaled_dot_product_attention,
)
from tensor_forge.tensor import Tensor


class Parameter(Tensor):
    """A Tensor subclass that is automatically registered as a trainable module parameter."""

    def __init__(
        self,
        data: ArrayLike | Tensor,
        dtype: DTypeLike = np.float64,
        requires_grad: bool = True,
    ) -> None:
        super().__init__(data, requires_grad=requires_grad, dtype=dtype)


class Module:
    """Base class for all neural network modules in `tensor-forge`."""

    def __init__(self) -> None:
        self.training: bool = True

    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        """Define the computation performed at every call."""
        raise NotImplementedError

    def __call__(self, *args: Any, **kwargs: Any) -> Tensor:
        return self.forward(*args, **kwargs)

    def named_parameters(self, prefix: str = "") -> list[tuple[str, Parameter]]:
        """Return a list of `(name, Parameter)` pairs for this module and all submodules."""
        params: list[tuple[str, Parameter]] = []
        seen: set[int] = set()

        def _collect(obj: Any, cur_prefix: str) -> None:
            if isinstance(obj, Parameter):
                if id(obj) not in seen:
                    seen.add(id(obj))
                    params.append((cur_prefix, obj))
            elif isinstance(obj, Module):
                for k, v in obj.__dict__.items():
                    if k.startswith("_"):
                        continue
                    sub_prefix = f"{cur_prefix}.{k}" if cur_prefix else k
                    _collect(v, sub_prefix)
            elif isinstance(obj, (list, tuple)):
                for idx, item in enumerate(obj):
                    sub_prefix = f"{cur_prefix}.{idx}" if cur_prefix else str(idx)
                    _collect(item, sub_prefix)
            elif isinstance(obj, dict):
                for k, item in obj.items():
                    sub_prefix = f"{cur_prefix}.{k}" if cur_prefix else str(k)
                    _collect(item, sub_prefix)

        _collect(self, prefix)
        return params

    def parameters(self) -> list[Parameter]:
        """Return a list of all trainable `Parameter` instances in the module hierarchy."""
        return [p for _, p in self.named_parameters()]

    def zero_grad(self) -> None:
        """Reset gradients of all module parameters to `None`."""
        for p in self.parameters():
            p.zero_grad()

    def train(self, mode: bool = True) -> Module:
        """Set the module and all submodules into training mode."""
        self.training = mode
        for v in self.__dict__.values():
            if isinstance(v, Module):
                v.train(mode)
            elif isinstance(v, (list, tuple)):
                for item in v:
                    if isinstance(item, Module):
                        item.train(mode)
        return self

    def eval(self) -> Module:
        """Set the module and all submodules into evaluation mode."""
        return self.train(False)

    def state_dict(self) -> dict[str, np.ndarray]:
        """Return a dictionary mapping parameter names to cloned NumPy arrays."""
        return {name: param.data.copy() for name, param in self.named_parameters()}

    def load_state_dict(self, state_dict: dict[str, np.ndarray], strict: bool = True) -> None:
        """Copy parameters from `state_dict` into this module and its descendants."""
        current = dict(self.named_parameters())
        if strict:
            missing = set(current.keys()) - set(state_dict.keys())
            unexpected = set(state_dict.keys()) - set(current.keys())
            if missing or unexpected:
                raise KeyError(
                    f"Error(s) in loading state_dict: missing={sorted(missing)}, "
                    f"unexpected={sorted(unexpected)}"
                )
        for name, param in current.items():
            if name in state_dict:
                arr = np.asarray(state_dict[name], dtype=param.dtype)
                if arr.shape != param.shape:
                    raise ValueError(
                        f"Shape mismatch for parameter {name!r}: expected {param.shape}, got {arr.shape}"
                    )
                param.data = arr.copy()


class Linear(Module):
    """Applies an affine linear transformation `y = x @ W + b`."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        init: str = "kaiming_normal",
        dtype: DTypeLike = np.float64,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        gen = rng if rng is not None else np.random.default_rng()

        if init == "kaiming_normal":
            std = float(np.sqrt(2.0 / float(in_features)))
            w_data = gen.normal(0.0, std, size=(in_features, out_features))
        elif init == "xavier_normal":
            std = float(np.sqrt(2.0 / float(in_features + out_features)))
            w_data = gen.normal(0.0, std, size=(in_features, out_features))
        elif init == "xavier_uniform":
            bound = float(np.sqrt(6.0 / float(in_features + out_features)))
            w_data = gen.uniform(-bound, bound, size=(in_features, out_features))
        elif init == "kaiming_uniform":
            bound = float(np.sqrt(6.0 / float(in_features)))
            w_data = gen.uniform(-bound, bound, size=(in_features, out_features))
        else:
            raise ValueError(f"Unsupported initialization scheme: {init!r}")

        self.weight = Parameter(w_data, dtype=dtype)
        self.bias: Parameter | None = (
            Parameter(np.zeros(out_features, dtype=dtype), dtype=dtype) if bias else None
        )

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out


class LayerNorm(Module):
    """Applies numerically stabilized Layer Normalization over the last dimension."""

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        dtype: DTypeLike = np.float64,
    ) -> None:
        super().__init__()
        self.normalized_shape = int(normalized_shape)
        self.eps = float(eps)
        self.elementwise_affine = bool(elementwise_affine)
        if self.elementwise_affine:
            self.weight: Parameter | None = Parameter(
                np.ones(self.normalized_shape, dtype=dtype), dtype=dtype
            )
            self.bias: Parameter | None = Parameter(
                np.zeros(self.normalized_shape, dtype=dtype), dtype=dtype
            )
        else:
            self.weight = None
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        return layer_norm(x, self.weight, self.bias, eps=self.eps)


class RMSNorm(Module):
    """Applies numerically stabilized Root Mean Square Layer Normalization."""

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        elementwise_affine: bool = True,
        dtype: DTypeLike = np.float64,
    ) -> None:
        super().__init__()
        self.normalized_shape = int(normalized_shape)
        self.eps = float(eps)
        self.elementwise_affine = bool(elementwise_affine)
        self.weight: Parameter | None = (
            Parameter(np.ones(self.normalized_shape, dtype=dtype), dtype=dtype)
            if self.elementwise_affine
            else None
        )

    def forward(self, x: Tensor) -> Tensor:
        return rms_norm(x, self.weight, eps=self.eps)


class MLP(Module):
    """Multi-Layer Perceptron with configurable hidden dimensions and activation."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | list[int] | tuple[int, ...],
        out_features: int,
        activation: str | Callable[[Tensor], Tensor] = "gelu",
        bias: bool = True,
        dtype: DTypeLike = np.float64,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__()
        gen = rng if rng is not None else np.random.default_rng()
        if isinstance(hidden_features, int):
            hidden_dims = [hidden_features]
        else:
            hidden_dims = list(hidden_features)

        dims = [in_features, *hidden_dims, out_features]
        self.layers: list[Linear] = [
            Linear(dims[i], dims[i + 1], bias=bias, dtype=dtype, rng=gen)
            for i in range(len(dims) - 1)
        ]
        self.activation_name = activation if isinstance(activation, str) else "custom"
        self._act_fn = self._resolve_activation(activation)

    @staticmethod
    def _resolve_activation(act: str | Callable[[Tensor], Tensor]) -> Callable[[Tensor], Tensor]:
        if callable(act):
            return act
        name = act.lower()
        if name == "relu":
            return lambda t: t.relu()
        if name == "gelu":
            return lambda t: t.gelu(approximate="none")
        if name == "gelu_tanh":
            return lambda t: t.gelu(approximate="tanh")
        if name == "silu" or name == "swish":
            return lambda t: t.silu()
        if name == "tanh":
            return lambda t: t.tanh()
        raise ValueError(f"Unsupported activation: {act!r}")

    def forward(self, x: Tensor) -> Tensor:
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = self._act_fn(h)
        return h


class TransformerBlock(Module):
    """Pre-Norm Transformer Block with Multi-Head Self-Attention and Feed-Forward MLP."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int | None = None,
        causal: bool = True,
        norm_type: str = "layernorm",
        dtype: DTypeLike = np.float64,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")

        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.head_dim = d_model // num_heads
        self.causal = bool(causal)
        hidden_ff = d_ff if d_ff is not None else 4 * d_model
        gen = rng if rng is not None else np.random.default_rng()

        if norm_type.lower() == "rmsnorm":
            self.norm1: Module = RMSNorm(d_model, dtype=dtype)
            self.norm2: Module = RMSNorm(d_model, dtype=dtype)
        else:
            self.norm1 = LayerNorm(d_model, dtype=dtype)
            self.norm2 = LayerNorm(d_model, dtype=dtype)

        self.q_proj = Linear(
            d_model, d_model, bias=False, init="xavier_normal", dtype=dtype, rng=gen
        )
        self.k_proj = Linear(
            d_model, d_model, bias=False, init="xavier_normal", dtype=dtype, rng=gen
        )
        self.v_proj = Linear(
            d_model, d_model, bias=False, init="xavier_normal", dtype=dtype, rng=gen
        )
        self.out_proj = Linear(
            d_model, d_model, bias=True, init="xavier_normal", dtype=dtype, rng=gen
        )

        self.mlp = MLP(
            in_features=d_model,
            hidden_features=hidden_ff,
            out_features=d_model,
            activation="gelu",
            dtype=dtype,
            rng=gen,
        )

    def forward(self, x: Tensor) -> Tensor:
        # Support 3D (B, T, D) or 2D (T, D) inputs
        is_2d = x.ndim == 2
        inp = x.reshape(1, x.shape[0], x.shape[1]) if is_2d else x
        bsz, seq_len, d_model = inp.shape

        h = self.norm1(inp)
        q = (
            self.q_proj(h)
            .reshape(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        k = (
            self.k_proj(h)
            .reshape(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        v = (
            self.v_proj(h)
            .reshape(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )

        attn_out = scaled_dot_product_attention(q, k, v, causal_mask=self.causal)
        merged = attn_out.transpose(0, 2, 1, 3).reshape(bsz, seq_len, d_model)
        x_res1 = inp + self.out_proj(merged)
        x_res2 = x_res1 + self.mlp(self.norm2(x_res1))

        return x_res2.reshape(seq_len, d_model) if is_2d else x_res2
