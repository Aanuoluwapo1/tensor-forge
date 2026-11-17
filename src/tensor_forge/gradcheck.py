"""Central Finite-Difference Gradient Verifier for automatic differentiation."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from tensor_forge.tensor import Tensor


def numerical_gradient_check(
    fn: Callable[..., Tensor],
    inputs: Sequence[Tensor],
    eps: float = 1e-6,
    rtol: float = 1e-5,
    atol: float = 1e-6,
    raise_error: bool = True,
) -> bool:
    """Verify analytical reverse-mode gradients against central finite differences.

    Perturbs every scalar element of every input tensor in `float64` precision:
        `grad_num[idx] = (f(x + eps * e_idx) - f(x - eps * e_idx)) / (2 * eps)`
    and compares elementwise against `x.grad` computed via `backward()`.

    Because `inputs` are mutated in-place (and restored afterwards), `fn` may
    either accept `*inputs` positionally or reference stateful `Module` parameters
    included in `inputs`.

    If `fn(*inputs)` returns a non-scalar tensor, its `.sum()` is taken implicitly
    so `backward()` computes the exact Jacobian sum.

    Args:
        fn: Differentiable callable mapping `*inputs` to a `Tensor`.
        inputs: Sequence of input `Tensor` objects to verify.
        eps: Perturbation step size for central difference.
        rtol: Relative tolerance for `np.allclose`.
        atol: Absolute tolerance for `np.allclose`.
        raise_error: If True, raises an `AssertionError` with detailed diagnostics on mismatch.

    Returns:
        `True` if all input gradients agree within `(rtol, atol)`.
    """
    saved_data = [inp.data.copy() for inp in inputs]
    saved_dtypes = [inp.dtype for inp in inputs]
    saved_grads = [None if inp.grad is None else inp.grad.copy() for inp in inputs]

    try:
        # 1. Promote all inputs to float64 and clear grads for analytical pass
        for inp, orig in zip(inputs, saved_data, strict=True):
            inp.data = orig.astype(np.float64, copy=True)
            inp.grad = None

        out = fn(*inputs)
        loss = out if out.size == 1 else out.sum()
        loss.backward()

        ana_grads: list[np.ndarray | None] = [
            None if inp.grad is None else inp.grad.astype(np.float64, copy=True) for inp in inputs
        ]

        # 2. Central finite-difference perturbation for each input tensor
        for arg_idx, inp in enumerate(inputs):
            if not inp.requires_grad:
                continue
            ana_grad = ana_grads[arg_idx]
            if ana_grad is None:
                if raise_error:
                    raise AssertionError(
                        f"Input #{arg_idx} requires_grad=True but received grad=None"
                    )
                return False

            num_grad = np.zeros_like(ana_grad, dtype=np.float64)
            target_arr = inp.data

            it = np.nditer(target_arr, flags=["multi_index"], op_flags=["readwrite"])
            while not it.finished:
                multi_idx = it.multi_index
                orig_val = float(target_arr[multi_idx])

                # f(x + eps)
                target_arr[multi_idx] = orig_val + eps
                out_pos = fn(*inputs)
                val_pos = float(out_pos.data.sum())

                # f(x - eps)
                target_arr[multi_idx] = orig_val - eps
                out_neg = fn(*inputs)
                val_neg = float(out_neg.data.sum())

                # Restore original value
                target_arr[multi_idx] = orig_val

                num_grad[multi_idx] = (val_pos - val_neg) / (2.0 * eps)
                it.iternext()

            if not np.allclose(ana_grad, num_grad, rtol=rtol, atol=atol):
                max_abs = float(np.max(np.abs(ana_grad - num_grad)))
                denom = np.maximum(np.abs(num_grad), np.abs(ana_grad)) + 1e-12
                max_rel = float(np.max(np.abs(ana_grad - num_grad) / denom))
                if raise_error:
                    raise AssertionError(
                        f"Gradient check failed for input #{arg_idx} (shape={inp.shape}): "
                        f"max_abs_err={max_abs:.3e}, max_rel_err={max_rel:.3e} "
                        f"(atol={atol:.1e}, rtol={rtol:.1e}).\n"
                        f"Analytical:\n{ana_grad}\nNumerical:\n{num_grad}"
                    )
                return False

        return True
    finally:
        for inp, orig, dt, g in zip(inputs, saved_data, saved_dtypes, saved_grads, strict=True):
            inp.data = orig.astype(dt, copy=True)
            inp.grad = g
