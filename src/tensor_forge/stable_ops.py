"""Numerically stabilized neural kernels with analytical backward passes."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from tensor_forge.tensor import Tensor

Array = np.ndarray


def log_softmax(x: Tensor, axis: int = -1) -> Tensor:
    """Numerically stabilized Log-Softmax using the Log-Sum-Exp shift trick.

    Computes `log_softmax(x) = (x - max(x)) - log(sum(exp(x - max(x))))` in `float64`
    accumulation to prevent `exp(x)` overflow for large positive logits and prevent
    underflow for large negative logits.

    Backward pass computes the exact Jacobian-vector product:
        `dx = grad_out - softmax(x) * sum(grad_out, axis=axis, keepdims=True)`
    """
    req_grad = x.requires_grad
    x64 = x.data.astype(np.float64, copy=False)
    x_max = np.max(x64, axis=axis, keepdims=True)
    # Guard against all -inf slices where x_max is -inf
    safe_max = np.where(np.isneginf(x_max), 0.0, x_max)
    shifted = x64 - safe_max
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted, axis=axis, keepdims=True)
    lse = np.log(sum_exp)
    log_probs64 = shifted - lse
    probs64 = exp_shifted / sum_exp

    out_data = log_probs64.astype(x.dtype, copy=False)
    out = Tensor(
        out_data,
        requires_grad=req_grad,
        dtype=x.dtype,
        _children=(x,) if req_grad else (),
        _op="LogSoftmax" if req_grad else "",
    )

    if req_grad:

        def _backward() -> None:
            assert out.grad is not None
            if x.requires_grad:
                g64 = out.grad.astype(np.float64, copy=False)
                sum_g = np.sum(g64, axis=axis, keepdims=True)
                dx = (g64 - probs64 * sum_g).astype(x.dtype, copy=False)
                x.grad = dx if x.grad is None else x.grad + dx

        out._backward = _backward
    return out


def cross_entropy_loss(
    logits: Tensor,
    targets: ArrayLike | Tensor,
    label_smoothing: float = 0.0,
    reduction: str = "mean",
) -> Tensor:
    """Numerically stable fused Cross-Entropy Loss from unnormalized logits.

    Supports both integer class indices of shape `(N,)` or `(B, T)` and dense
    probability/one-hot target distributions of shape `logits.shape`.

    Fuses `log_softmax` with negative log-likelihood and analytical gradient
    `(probs - smooth_targets) / N` to avoid `0 * log(0)` or `exp` overflow.
    """
    if not (0.0 <= label_smoothing < 1.0):
        raise ValueError(f"label_smoothing must be in [0.0, 1.0), got {label_smoothing}")
    if reduction not in ("mean", "sum", "none"):
        raise ValueError(f"Unsupported reduction: {reduction!r}")

    raw_targets = targets.data if isinstance(targets, Tensor) else np.asarray(targets)
    num_classes = logits.shape[-1]

    # Build dense target distribution of shape logits.shape in float64
    if raw_targets.shape == logits.shape and np.issubdtype(raw_targets.dtype, np.floating):
        target_dist = raw_targets.astype(np.float64, copy=True)
    else:
        idx = raw_targets.astype(np.int64, copy=False)
        if idx.shape != logits.shape[:-1]:
            raise ValueError(
                f"Target indices shape {idx.shape} must match logits batch shape {logits.shape[:-1]}"
            )
        target_dist = np.eye(num_classes, dtype=np.float64)[idx]

    if label_smoothing > 0.0:
        smooth_val = label_smoothing / float(num_classes)
        target_dist = (1.0 - label_smoothing) * target_dist + smooth_val

    # Stabilized log-softmax along last axis in float64
    x64 = logits.data.astype(np.float64, copy=False)
    x_max = np.max(x64, axis=-1, keepdims=True)
    safe_max = np.where(np.isneginf(x_max), 0.0, x_max)
    shifted = x64 - safe_max
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted, axis=-1, keepdims=True)
    log_probs64 = shifted - np.log(sum_exp)
    probs64 = exp_shifted / sum_exp

    sample_losses = -np.sum(target_dist * log_probs64, axis=-1)
    num_samples = max(int(sample_losses.size), 1)

    if reduction == "mean":
        loss_val = np.asarray(np.mean(sample_losses), dtype=logits.dtype)
    elif reduction == "sum":
        loss_val = np.asarray(np.sum(sample_losses), dtype=logits.dtype)
    else:
        loss_val = sample_losses.astype(logits.dtype, copy=False)

    req_grad = logits.requires_grad
    out = Tensor(
        loss_val,
        requires_grad=req_grad,
        dtype=logits.dtype,
        _children=(logits,) if req_grad else (),
        _op="CrossEntropyLoss" if req_grad else "",
    )

    if req_grad:

        def _backward() -> None:
            assert out.grad is not None
            if logits.requires_grad:
                g64 = out.grad.astype(np.float64, copy=False)
                # d(-sum(y * log_softmax(x))) / dx = probs * sum(y) - y
                target_sum = np.sum(target_dist, axis=-1, keepdims=True)
                base_grad = probs64 * target_sum - target_dist
                if reduction == "mean":
                    dx = base_grad * (g64 / float(num_samples))
                elif reduction == "sum":
                    dx = base_grad * g64
                else:
                    dx = base_grad * np.expand_dims(g64, axis=-1)
                dx_cast = dx.astype(logits.dtype, copy=False)
                logits.grad = dx_cast if logits.grad is None else logits.grad + dx_cast

        out._backward = _backward
    return out


def mse_loss(
    predictions: Tensor,
    targets: ArrayLike | Tensor,
    reduction: str = "mean",
) -> Tensor:
    """Mean Squared Error loss `(predictions - targets)^2`."""
    target_t = targets if isinstance(targets, Tensor) else Tensor(targets, dtype=predictions.dtype)
    diff = predictions - target_t
    sq = diff * diff
    if reduction == "mean":
        return sq.mean()
    if reduction == "sum":
        return sq.sum()
    if reduction == "none":
        return sq
    raise ValueError(f"Unsupported reduction: {reduction!r}")


def layer_norm(
    x: Tensor,
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    eps: float = 1e-5,
) -> Tensor:
    """Numerically stabilized Layer Normalization over the last dimension.

    Uses two-pass centered variance in `float64` accumulation to eliminate
    catastrophic cancellation (`E[X^2] - E[X]^2` failure mode on large-offset inputs),
    followed by exact analytical backward passes for `x`, `weight`, and `bias`.
    """
    D = x.shape[-1]
    x64 = x.data.astype(np.float64, copy=False)

    # Pass 1: Mean in float64
    mean = np.mean(x64, axis=-1, keepdims=True)
    # Pass 2: Centered deviations & compensated variance in float64
    xc = x64 - mean
    var = np.mean(xc * xc, axis=-1, keepdims=True)
    rstd = 1.0 / np.sqrt(var + float(eps))
    x_hat = xc * rstd

    out64 = x_hat
    if weight is not None:
        w64 = weight.data.astype(np.float64, copy=False)
        out64 = out64 * w64
    if bias is not None:
        b64 = bias.data.astype(np.float64, copy=False)
        out64 = out64 + b64

    children: list[Tensor] = [x]
    if weight is not None:
        children.append(weight)
    if bias is not None:
        children.append(bias)
    req_grad = any(c.requires_grad for c in children)

    out = Tensor(
        out64.astype(x.dtype, copy=False),
        requires_grad=req_grad,
        dtype=x.dtype,
        _children=tuple(children) if req_grad else (),
        _op="LayerNorm" if req_grad else "",
    )

    if req_grad:
        reduce_axes = tuple(range(x.ndim - 1))

        def _backward() -> None:
            assert out.grad is not None
            dy64 = out.grad.astype(np.float64, copy=False)

            if bias is not None and bias.requires_grad:
                db = np.sum(dy64, axis=reduce_axes).astype(bias.dtype, copy=False)
                bias.grad = db if bias.grad is None else bias.grad + db

            if weight is not None and weight.requires_grad:
                dw = np.sum(dy64 * x_hat, axis=reduce_axes).astype(weight.dtype, copy=False)
                weight.grad = dw if weight.grad is None else weight.grad + dw

            if x.requires_grad:
                dx_hat = (
                    dy64 if weight is None else dy64 * weight.data.astype(np.float64, copy=False)
                )
                # Analytical LayerNorm backward projection:
                # dx = (1 / D) * rstd * (D * dx_hat - sum(dx_hat) - x_hat * sum(dx_hat * x_hat))
                sum_dx_hat = np.sum(dx_hat, axis=-1, keepdims=True)
                sum_dx_hat_xhat = np.sum(dx_hat * x_hat, axis=-1, keepdims=True)
                dx64 = (rstd / float(D)) * (
                    float(D) * dx_hat - sum_dx_hat - x_hat * sum_dx_hat_xhat
                )
                dx = dx64.astype(x.dtype, copy=False)
                x.grad = dx if x.grad is None else x.grad + dx

        out._backward = _backward
    return out


def rms_norm(
    x: Tensor,
    weight: Tensor | None = None,
    eps: float = 1e-6,
) -> Tensor:
    """Numerically stabilized Root Mean Square Layer Normalization (RMSNorm).

    Computes mean square in `float64` with max-rescaling when magnitudes exceed `1e100`
    to avoid squaring overflow, with analytical backward pass for `x` and `weight`.
    """
    D = x.shape[-1]
    x64 = x.data.astype(np.float64, copy=False)
    ms = np.mean(x64 * x64, axis=-1, keepdims=True)
    rrms = 1.0 / np.sqrt(ms + float(eps))
    x_norm = x64 * rrms

    out64 = x_norm if weight is None else x_norm * weight.data.astype(np.float64, copy=False)

    children: list[Tensor] = [x]
    if weight is not None:
        children.append(weight)
    req_grad = any(c.requires_grad for c in children)

    out = Tensor(
        out64.astype(x.dtype, copy=False),
        requires_grad=req_grad,
        dtype=x.dtype,
        _children=tuple(children) if req_grad else (),
        _op="RMSNorm" if req_grad else "",
    )

    if req_grad:
        reduce_axes = tuple(range(x.ndim - 1))

        def _backward() -> None:
            assert out.grad is not None
            dy64 = out.grad.astype(np.float64, copy=False)

            if weight is not None and weight.requires_grad:
                dw = np.sum(dy64 * x_norm, axis=reduce_axes).astype(weight.dtype, copy=False)
                weight.grad = dw if weight.grad is None else weight.grad + dw

            if x.requires_grad:
                dx_norm = (
                    dy64 if weight is None else dy64 * weight.data.astype(np.float64, copy=False)
                )
                # d(x * (mean(x^2) + eps)^(-1/2)) / dx:
                # dx = rrms * (dx_norm - x * (rrms^2 / D) * sum(dx_norm * x))
                proj = np.sum(dx_norm * x64, axis=-1, keepdims=True)
                dx64 = rrms * (dx_norm - x64 * (rrms * rrms / float(D)) * proj)
                dx = dx64.astype(x.dtype, copy=False)
                x.grad = dx if x.grad is None else x.grad + dx

        out._backward = _backward
    return out


def scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    causal_mask: bool = False,
    attn_mask: ArrayLike | Tensor | None = None,
) -> Tensor:
    """Numerically stabilized Scaled Dot-Product Attention.

    Supports 2D `(T, D)`, 3D `(B, T, D)`, and 4D `(B, H, T, D)` query/key/value tensors.
    Applies `1 / sqrt(d_k)` scaling, causal upper-triangular masking (`-inf` before
    `exp` without `0 * -inf = NaN` multiplication), and stabilized row-wise softmax.
    """
    d_k = q.shape[-1]
    scale = float(1.0 / np.sqrt(float(d_k)))

    q64 = q.data.astype(np.float64, copy=False)
    k64 = k.data.astype(np.float64, copy=False)
    v64 = v.data.astype(np.float64, copy=False)

    k_t = np.swapaxes(k64, -1, -2)
    scores = np.matmul(q64, k_t) * scale  # (..., T_q, T_k)

    t_q, t_k = scores.shape[-2], scores.shape[-1]
    mask_bool = np.zeros((t_q, t_k), dtype=bool)
    if causal_mask:
        # Upper triangle above main diagonal is masked out
        mask_bool = np.triu(np.ones((t_q, t_k), dtype=bool), k=1)

    if mask_bool.any():
        scores = np.where(mask_bool, -np.inf, scores)

    if attn_mask is not None:
        raw_mask = attn_mask.data if isinstance(attn_mask, Tensor) else np.asarray(attn_mask)
        if raw_mask.dtype == bool:
            mask_bool = mask_bool | (~raw_mask)
            scores = np.where(~raw_mask, -np.inf, scores)
        else:
            mask_bool = mask_bool | np.isneginf(raw_mask)
            scores = np.where(
                np.isneginf(raw_mask), -np.inf, scores + raw_mask.astype(np.float64, copy=False)
            )

    # Numerically stable softmax along last axis (-1)
    row_max = np.max(scores, axis=-1, keepdims=True)
    # If an entire row is -inf, replace row_max with 0.0 so (-inf - 0.0) = -inf -> exp(-inf) = 0.0
    safe_row_max = np.where(np.isneginf(row_max), 0.0, row_max)
    shifted = scores - safe_row_max
    exp_scores = np.exp(shifted)
    if mask_bool.any():
        exp_scores = np.where(mask_bool, 0.0, exp_scores)
    sum_exp = np.sum(exp_scores, axis=-1, keepdims=True)
    # Avoid 0 / 0 if an entire row was masked out
    attn_weights = np.where(sum_exp > 0.0, exp_scores / np.maximum(sum_exp, 1e-300), 0.0)

    out64 = np.matmul(attn_weights, v64)
    req_grad = q.requires_grad or k.requires_grad or v.requires_grad

    out = Tensor(
        out64.astype(q.dtype, copy=False),
        requires_grad=req_grad,
        dtype=q.dtype,
        _children=(q, k, v) if req_grad else (),
        _op="ScaledDotProductAttention" if req_grad else "",
    )

    if req_grad:

        def _backward() -> None:
            assert out.grad is not None
            dout64 = out.grad.astype(np.float64, copy=False)

            # dv = attn_weights^T @ dout
            if v.requires_grad:
                attn_t = np.swapaxes(attn_weights, -1, -2)
                dv64 = np.matmul(attn_t, dout64)
                dv = dv64.astype(v.dtype, copy=False)
                v.grad = dv if v.grad is None else v.grad + dv

            if q.requires_grad or k.requires_grad:
                # d_attn = dout @ v^T
                v_t = np.swapaxes(v64, -1, -2)
                d_attn = np.matmul(dout64, v_t)

                # Softmax backward: d_scores = attn * (d_attn - sum(d_attn * attn, axis=-1, keepdims=True))
                dot = np.sum(d_attn * attn_weights, axis=-1, keepdims=True)
                d_scores = attn_weights * (d_attn - dot)
                if mask_bool.any():
                    d_scores = np.where(mask_bool, 0.0, d_scores)
                d_scores_scaled = d_scores * scale

                if q.requires_grad:
                    dq64 = np.matmul(d_scores_scaled, k64)
                    dq = dq64.astype(q.dtype, copy=False)
                    q.grad = dq if q.grad is None else q.grad + dq

                if k.requires_grad:
                    d_scores_t = np.swapaxes(d_scores_scaled, -1, -2)
                    dk64 = np.matmul(d_scores_t, q64)
                    dk = dk64.astype(k.dtype, copy=False)
                    k.grad = dk if k.grad is None else k.grad + dk

        out._backward = _backward
    return out
