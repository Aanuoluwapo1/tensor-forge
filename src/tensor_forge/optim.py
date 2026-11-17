"""First-order and adaptive optimizers and gradient clipping utilities."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from tensor_forge.tensor import Tensor


def clip_grad_norm_(
    parameters: Iterable[Tensor],
    max_norm: float,
    eps: float = 1e-6,
) -> float:
    """Clip the global L2 norm of parameter gradients in-place.

    Args:
        parameters: Iterable of Tensors whose `.grad` will be scaled in-place.
        max_norm: Maximum allowed global L2 norm.
        eps: Small constant added to denominator for division stability.

    Returns:
        Total global L2 norm of the gradients prior to clipping.
    """
    if max_norm <= 0.0:
        raise ValueError(f"max_norm must be positive, got {max_norm}")

    params = [p for p in parameters if p.grad is not None]
    if not params:
        return 0.0

    total_norm_sq = 0.0
    for p in params:
        assert p.grad is not None
        g64 = p.grad.astype(np.float64, copy=False)
        total_norm_sq += float(np.sum(g64 * g64))

    total_norm = float(np.sqrt(total_norm_sq))
    clip_coef = max_norm / (total_norm + float(eps))
    if clip_coef < 1.0:
        for p in params:
            assert p.grad is not None
            p.grad = (p.grad * clip_coef).astype(p.dtype, copy=False)

    return total_norm


class Optimizer:
    """Base class for all optimizers."""

    def __init__(self, params: Iterable[Tensor], lr: float) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        self.params: list[Tensor] = list(params)
        self.lr: float = float(lr)

    def zero_grad(self) -> None:
        """Reset gradients of all managed parameters to `None`."""
        for p in self.params:
            p.zero_grad()

    def step(self) -> None:
        """Perform a single optimization step."""
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic Gradient Descent with momentum and L2 weight decay."""

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-2,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ) -> None:
        super().__init__(params, lr)
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)
        self.nesterov = bool(nesterov)
        self._velocities: dict[int, np.ndarray] = {}

    def step(self) -> None:
        """Update parameters using SGD with optional momentum and weight decay."""
        for p in self.params:
            if p.grad is None:
                continue
            grad = p.grad.astype(np.float64, copy=True)
            if self.weight_decay != 0.0:
                grad = grad + self.weight_decay * p.data.astype(np.float64, copy=False)

            if self.momentum != 0.0:
                pid = id(p)
                if pid not in self._velocities:
                    buf = grad.copy()
                else:
                    buf = self.momentum * self._velocities[pid] + grad
                self._velocities[pid] = buf
                step_grad = (grad + self.momentum * buf) if self.nesterov else buf
            else:
                step_grad = grad

            p.data = (p.data.astype(np.float64, copy=False) - self.lr * step_grad).astype(
                p.dtype, copy=False
            )


class AdamW(Optimizer):
    """AdamW optimizer with decoupled weight decay and bias correction (Loshchilov & Hutter, 2019).

    Args:
        params: Iterable of trainable `Tensor` or `Parameter` instances.
        lr: Learning rate.
        betas: Coefficients `(beta1, beta2)` for running averages of gradient and its square.
        eps: Term added to the denominator to improve numerical stability.
        weight_decay: Decoupled weight decay coefficient.
        decay_1d: Whether to apply decoupled weight decay to 1D parameters (such as biases
            and LayerNorm/RMSNorm affine weights). Defaults to `True`; set to `False` to
            exempt all parameters with `p.ndim < 2` from weight decay.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        decay_1d: bool = True,
    ) -> None:
        super().__init__(params, lr)
        beta1, beta2 = betas
        if not (0.0 <= beta1 < 1.0 and 0.0 <= beta2 < 1.0):
            raise ValueError(f"Invalid betas: {betas}")
        if eps <= 0.0:
            raise ValueError(f"Invalid eps: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.decay_1d = bool(decay_1d)
        self.t: int = 0
        self._m: dict[int, np.ndarray] = {}
        self._v: dict[int, np.ndarray] = {}

    def step(self) -> None:
        """Perform a single AdamW parameter update step."""
        self.t += 1
        bias_corr1 = 1.0 - (self.beta1**self.t)
        bias_corr2 = 1.0 - (self.beta2**self.t)

        for p in self.params:
            if p.grad is None:
                continue
            pid = id(p)
            g64 = p.grad.astype(np.float64, copy=False)
            w64 = p.data.astype(np.float64, copy=True)

            # 1. Decoupled weight decay directly on parameter weights
            apply_wd = self.weight_decay != 0.0 and (self.decay_1d or p.ndim >= 2)
            if apply_wd:
                w64 = w64 * (1.0 - self.lr * self.weight_decay)

            # 2. First and second moment running averages in float64
            if pid not in self._m:
                self._m[pid] = np.zeros_like(w64, dtype=np.float64)
                self._v[pid] = np.zeros_like(w64, dtype=np.float64)

            m = self.beta1 * self._m[pid] + (1.0 - self.beta1) * g64
            v = self.beta2 * self._v[pid] + (1.0 - self.beta2) * (g64 * g64)
            self._m[pid] = m
            self._v[pid] = v

            m_hat = m / bias_corr1
            v_hat = v / bias_corr2

            w64 = w64 - self.lr * (m_hat / (np.sqrt(v_hat) + self.eps))
            p.data = w64.astype(p.dtype, copy=False)
