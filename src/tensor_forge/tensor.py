"""Core Tensor class and reverse-mode automatic differentiation engine."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, DTypeLike

Array = np.ndarray


def unbroadcast(grad: Array, target_shape: tuple[int, ...]) -> Array:
    """Reduce a broadcasted gradient array back to `target_shape`.

    Handles both prepended dimensions (where `grad.ndim > len(target_shape)`)
    and singleton broadcast axes (where `target_shape[i] == 1` and `grad.shape[i] > 1`).

    Args:
        grad: Upstream gradient array of shape `S_out`.
        target_shape: Original tensor shape `S_in` prior to broadcasting.

    Returns:
        Gradient array reduced to `target_shape` with identical dtype to `grad`.
    """
    if grad.shape == target_shape:
        return grad

    if len(target_shape) == 0:
        return np.asarray(grad.sum(), dtype=grad.dtype)

    ndim_diff = grad.ndim - len(target_shape)
    if ndim_diff < 0:
        raise ValueError(
            f"Cannot unbroadcast gradient of shape {grad.shape} to higher-rank shape {target_shape}"
        )

    # 1. Sum over leading dimensions added by broadcasting
    if ndim_diff > 0:
        lead_axes = tuple(range(ndim_diff))
        grad = grad.sum(axis=lead_axes)

    # 2. Sum over axes that were length-1 in target_shape but broadcasted to >1
    keep_axes = tuple(
        i
        for i, (g_dim, t_dim) in enumerate(zip(grad.shape, target_shape, strict=True))
        if t_dim == 1 and g_dim > 1
    )
    if keep_axes:
        grad = grad.sum(axis=keep_axes, keepdims=True)

    return grad.reshape(target_shape)


def _erf_approx(x: Array) -> Array:
    """High-precision Abramowitz & Stegun 7.1.26 rational approximation of erf(x).

    Maximum absolute error <= 1.5e-7 across all real numbers, vectorized in NumPy
    and overflow-safe for large magnitudes.
    """
    sign = np.sign(x)
    abs_x = np.abs(x)
    p = 0.3275911
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429

    t = 1.0 / (1.0 + p * abs_x)
    poly = (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t
    # Clamp abs_x squared at 100.0 so exp(-x^2) gracefully underflows to 0.0 without overflow
    exp_term = np.exp(-np.minimum(abs_x * abs_x, 100.0))
    y = 1.0 - poly * exp_term
    return (sign * y).astype(x.dtype, copy=False)


class Tensor:
    """Multi-dimensional array supporting reverse-mode automatic differentiation.

    Wraps a `numpy.ndarray` (`float32` or `float64`) and dynamically builds a
    directed acyclic computation graph (DAG) during forward evaluation.
    """

    __slots__ = ("data", "grad", "requires_grad", "_prev", "_op", "_backward")
    __array_priority__ = 1000.0

    def __init__(
        self,
        data: ArrayLike | Tensor,
        requires_grad: bool = False,
        dtype: DTypeLike | None = None,
        _children: tuple[Tensor, ...] = (),
        _op: str = "",
    ) -> None:
        if isinstance(data, Tensor):
            raw = data.data
            raw = raw.astype(dtype, copy=True) if dtype is not None else raw.copy()
        elif isinstance(data, np.ndarray):
            target_dtype = (
                dtype
                if dtype is not None
                else (data.dtype if np.issubdtype(data.dtype, np.floating) else np.float64)
            )
            raw = np.asarray(data, dtype=target_dtype)
        else:
            target_dtype = dtype if dtype is not None else np.float64
            raw = np.asarray(data, dtype=target_dtype)

        if not np.issubdtype(raw.dtype, np.floating):
            raw = raw.astype(np.float64)

        self.data: Array = raw
        self.grad: Array | None = None
        self.requires_grad: bool = bool(requires_grad)
        self._prev: tuple[Tensor, ...] = _children
        self._op: str = _op
        self._backward: Callable[[], None] = lambda: None

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape tuple of the underlying NumPy array."""
        return self.data.shape

    @property
    def ndim(self) -> int:
        """Number of array dimensions."""
        return self.data.ndim

    @property
    def size(self) -> int:
        """Total number of elements in the tensor."""
        return self.data.size

    @property
    def dtype(self) -> np.dtype[Any]:
        """Data type of the underlying NumPy array."""
        return self.data.dtype

    @property
    def T(self) -> Tensor:
        """Reverse or permute the axes of the tensor."""
        return self.transpose()

    def item(self) -> float:
        """Return the scalar value of a 1-element tensor as a Python float."""
        return float(self.data.item())

    def numpy(self) -> Array:
        """Return a view or copy of the underlying NumPy array."""
        return self.data

    def detach(self) -> Tensor:
        """Return a new Tensor detached from the computation graph."""
        return Tensor(self.data.copy(), requires_grad=False, dtype=self.dtype)

    def zero_grad(self) -> None:
        """Reset `.grad` to None."""
        self.grad = None

    def __repr__(self) -> str:
        grad_str = f", requires_grad={self.requires_grad}" if self.requires_grad else ""
        op_str = f", grad_fn=<{self._op}>" if self._op and self.requires_grad else ""
        return f"Tensor({np.array2string(self.data, separator=', ')}, dtype={self.dtype}{grad_str}{op_str})"

    def _ensure_tensor(self, other: ArrayLike | Tensor) -> Tensor:
        if isinstance(other, Tensor):
            return other
        return Tensor(other, requires_grad=False, dtype=self.dtype)

    def backward(self, grad: ArrayLike | Tensor | None = None) -> None:
        """Compute gradients of this tensor with respect to all graph leaves.

        Uses an explicit iterative Depth-First Search (DFS) topological sort to
        avoid Python recursion depth limits on deep computational graphs.

        Args:
            grad: Upstream gradient with same shape as `self`. Defaults to ones
                for scalar outputs (`self.size == 1`).
        """
        if not self.requires_grad:
            raise RuntimeError(
                "Cannot call backward() on a Tensor that does not require gradients."
            )

        if grad is None:
            if self.size != 1:
                raise RuntimeError(
                    f"grad can be implicitly created only for scalar outputs, got shape {self.shape}"
                )
            self.grad = np.ones_like(self.data, dtype=self.dtype)
        elif isinstance(grad, Tensor):
            if grad.shape != self.shape:
                raise ValueError(
                    f"grad shape {grad.shape} does not match tensor shape {self.shape}"
                )
            self.grad = grad.data.astype(self.dtype, copy=True)
        else:
            grad_arr = np.asarray(grad, dtype=self.dtype)
            if grad_arr.shape != self.shape:
                raise ValueError(
                    f"grad shape {grad_arr.shape} does not match tensor shape {self.shape}"
                )
            self.grad = grad_arr.copy()

        # Iterative post-order DFS topological sort
        topo: list[Tensor] = []
        visited: set[int] = set()
        # Stack entries: (node, children_processed)
        stack: list[tuple[Tensor, bool]] = [(self, False)]

        while stack:
            node, processed = stack.pop()
            node_id = id(node)
            if processed:
                topo.append(node)
                continue
            if node_id in visited:
                continue
            visited.add(node_id)
            stack.append((node, True))
            for child in node._prev:
                if id(child) not in visited and child.requires_grad:
                    stack.append((child, False))

        for node in reversed(topo):
            node._backward()

    # -------------------------------------------------------------------------
    # Arithmetic & Broadcasting Primitives
    # -------------------------------------------------------------------------

    def __add__(self, other: ArrayLike | Tensor) -> Tensor:
        other_t = self._ensure_tensor(other)
        req_grad = self.requires_grad or other_t.requires_grad
        out_data = self.data + other_t.data
        out = Tensor(
            out_data,
            requires_grad=req_grad,
            dtype=out_data.dtype,
            _children=(self, other_t) if req_grad else (),
            _op="Add" if req_grad else "",
        )

        if req_grad:

            def _backward() -> None:
                assert out.grad is not None
                if self.requires_grad:
                    g_self = unbroadcast(out.grad, self.shape).astype(self.dtype, copy=False)
                    self.grad = g_self if self.grad is None else self.grad + g_self
                if other_t.requires_grad:
                    g_other = unbroadcast(out.grad, other_t.shape).astype(other_t.dtype, copy=False)
                    other_t.grad = g_other if other_t.grad is None else other_t.grad + g_other

            out._backward = _backward
        return out

    def __radd__(self, other: ArrayLike | Tensor) -> Tensor:
        return self.__add__(other)

    def __neg__(self) -> Tensor:
        req_grad = self.requires_grad
        out = Tensor(
            -self.data,
            requires_grad=req_grad,
            dtype=self.dtype,
            _children=(self,) if req_grad else (),
            _op="Neg" if req_grad else "",
        )
        if req_grad:

            def _backward() -> None:
                assert out.grad is not None
                if self.requires_grad:
                    g = -out.grad
                    self.grad = g if self.grad is None else self.grad + g

            out._backward = _backward
        return out

    def __sub__(self, other: ArrayLike | Tensor) -> Tensor:
        other_t = self._ensure_tensor(other)
        req_grad = self.requires_grad or other_t.requires_grad
        out_data = self.data - other_t.data
        out = Tensor(
            out_data,
            requires_grad=req_grad,
            dtype=out_data.dtype,
            _children=(self, other_t) if req_grad else (),
            _op="Sub" if req_grad else "",
        )

        if req_grad:

            def _backward() -> None:
                assert out.grad is not None
                if self.requires_grad:
                    g_self = unbroadcast(out.grad, self.shape).astype(self.dtype, copy=False)
                    self.grad = g_self if self.grad is None else self.grad + g_self
                if other_t.requires_grad:
                    g_other = unbroadcast(-out.grad, other_t.shape).astype(
                        other_t.dtype, copy=False
                    )
                    other_t.grad = g_other if other_t.grad is None else other_t.grad + g_other

            out._backward = _backward
        return out

    def __rsub__(self, other: ArrayLike | Tensor) -> Tensor:
        other_t = self._ensure_tensor(other)
        return other_t.__sub__(self)

    def __mul__(self, other: ArrayLike | Tensor) -> Tensor:
        other_t = self._ensure_tensor(other)
        req_grad = self.requires_grad or other_t.requires_grad
        out_data = self.data * other_t.data
        out = Tensor(
            out_data,
            requires_grad=req_grad,
            dtype=out_data.dtype,
            _children=(self, other_t) if req_grad else (),
            _op="Mul" if req_grad else "",
        )

        if req_grad:

            def _backward() -> None:
                assert out.grad is not None
                if self.requires_grad:
                    g_self = unbroadcast(out.grad * other_t.data, self.shape).astype(
                        self.dtype, copy=False
                    )
                    self.grad = g_self if self.grad is None else self.grad + g_self
                if other_t.requires_grad:
                    g_other = unbroadcast(out.grad * self.data, other_t.shape).astype(
                        other_t.dtype, copy=False
                    )
                    other_t.grad = g_other if other_t.grad is None else other_t.grad + g_other

            out._backward = _backward
        return out

    def __rmul__(self, other: ArrayLike | Tensor) -> Tensor:
        return self.__mul__(other)

    def __truediv__(self, other: ArrayLike | Tensor) -> Tensor:
        other_t = self._ensure_tensor(other)
        req_grad = self.requires_grad or other_t.requires_grad
        out_data = self.data / other_t.data
        out = Tensor(
            out_data,
            requires_grad=req_grad,
            dtype=out_data.dtype,
            _children=(self, other_t) if req_grad else (),
            _op="Div" if req_grad else "",
        )

        if req_grad:

            def _backward() -> None:
                assert out.grad is not None
                if self.requires_grad:
                    g_self = unbroadcast(out.grad / other_t.data, self.shape).astype(
                        self.dtype, copy=False
                    )
                    self.grad = g_self if self.grad is None else self.grad + g_self
                if other_t.requires_grad:
                    raw_g = -out.grad * self.data / (other_t.data * other_t.data)
                    g_other = unbroadcast(raw_g, other_t.shape).astype(other_t.dtype, copy=False)
                    other_t.grad = g_other if other_t.grad is None else other_t.grad + g_other

            out._backward = _backward
        return out

    def __rtruediv__(self, other: ArrayLike | Tensor) -> Tensor:
        other_t = self._ensure_tensor(other)
        return other_t.__truediv__(self)

    def __pow__(self, other: float | int | ArrayLike | Tensor) -> Tensor:
        if isinstance(other, (int, float)):
            req_grad = self.requires_grad
            out_data = (self.data**other).astype(self.dtype, copy=False)
            out = Tensor(
                out_data,
                requires_grad=req_grad,
                dtype=self.dtype,
                _children=(self,) if req_grad else (),
                _op="PowScalar" if req_grad else "",
            )
            if req_grad:

                def _backward() -> None:
                    assert out.grad is not None
                    if self.requires_grad:
                        g_self = (out.grad * (other * (self.data ** (other - 1)))).astype(
                            self.dtype, copy=False
                        )
                        self.grad = g_self if self.grad is None else self.grad + g_self

                out._backward = _backward
            return out

        other_t = self._ensure_tensor(other)
        req_grad = self.requires_grad or other_t.requires_grad
        out_data = self.data**other_t.data
        out = Tensor(
            out_data,
            requires_grad=req_grad,
            dtype=out_data.dtype,
            _children=(self, other_t) if req_grad else (),
            _op="Pow" if req_grad else "",
        )

        if req_grad:

            def _backward() -> None:
                assert out.grad is not None
                if self.requires_grad:
                    raw_self = out.grad * other_t.data * (self.data ** (other_t.data - 1.0))
                    g_self = unbroadcast(raw_self, self.shape).astype(self.dtype, copy=False)
                    self.grad = g_self if self.grad is None else self.grad + g_self
                if other_t.requires_grad:
                    safe_base = np.maximum(self.data, np.finfo(self.dtype).tiny)
                    raw_other = out.grad * out_data * np.log(safe_base)
                    g_other = unbroadcast(raw_other, other_t.shape).astype(
                        other_t.dtype, copy=False
                    )
                    other_t.grad = g_other if other_t.grad is None else other_t.grad + g_other

            out._backward = _backward
        return out

    # -------------------------------------------------------------------------
    # Matrix Multiplication (1D, 2D, and Batched 3D/4D+)
    # -------------------------------------------------------------------------

    def __matmul__(self, other: ArrayLike | Tensor) -> Tensor:
        return self.matmul(other)

    def __rmatmul__(self, other: ArrayLike | Tensor) -> Tensor:
        other_t = self._ensure_tensor(other)
        return other_t.matmul(self)

    def matmul(self, other: ArrayLike | Tensor) -> Tensor:
        """Matrix product of two tensors supporting 1D, 2D, and batched 3D/4D+ tensors."""
        other_t = self._ensure_tensor(other)
        req_grad = self.requires_grad or other_t.requires_grad
        out_data = np.matmul(self.data, other_t.data)
        out = Tensor(
            out_data,
            requires_grad=req_grad,
            dtype=out_data.dtype,
            _children=(self, other_t) if req_grad else (),
            _op="MatMul" if req_grad else "",
        )

        if req_grad:

            def _backward() -> None:
                assert out.grad is not None
                a = self.data
                b = other_t.data
                g = out.grad

                if a.ndim == 1 and b.ndim == 1:
                    if self.requires_grad:
                        g_a = (g * b).astype(self.dtype, copy=False)
                        self.grad = g_a if self.grad is None else self.grad + g_a
                    if other_t.requires_grad:
                        g_b = (g * a).astype(other_t.dtype, copy=False)
                        other_t.grad = g_b if other_t.grad is None else other_t.grad + g_b
                elif a.ndim == 1 and b.ndim >= 2:
                    # Promote a to (..., 1, K) and g to (..., 1, N)
                    a_exp = a[np.newaxis, :]
                    g_exp = np.expand_dims(g, axis=-2)
                    if self.requires_grad:
                        b_t = np.swapaxes(b, -1, -2)
                        raw_a = np.matmul(g_exp, b_t).squeeze(axis=-2)
                        g_a = unbroadcast(raw_a, self.shape).astype(self.dtype, copy=False)
                        self.grad = g_a if self.grad is None else self.grad + g_a
                    if other_t.requires_grad:
                        a_t = np.swapaxes(a_exp, -1, -2)
                        raw_b = np.matmul(a_t, g_exp)
                        g_b = unbroadcast(raw_b, other_t.shape).astype(other_t.dtype, copy=False)
                        other_t.grad = g_b if other_t.grad is None else other_t.grad + g_b
                elif a.ndim >= 2 and b.ndim == 1:
                    # Promote b to (K, 1) and g to (..., M, 1)
                    b_exp = b[:, np.newaxis]
                    g_exp = np.expand_dims(g, axis=-1)
                    if self.requires_grad:
                        b_t = np.swapaxes(b_exp, -1, -2)
                        raw_a = np.matmul(g_exp, b_t)
                        g_a = unbroadcast(raw_a, self.shape).astype(self.dtype, copy=False)
                        self.grad = g_a if self.grad is None else self.grad + g_a
                    if other_t.requires_grad:
                        a_t = np.swapaxes(a, -1, -2)
                        raw_b = np.matmul(a_t, g_exp).squeeze(axis=-1)
                        g_b = unbroadcast(raw_b, other_t.shape).astype(other_t.dtype, copy=False)
                        other_t.grad = g_b if other_t.grad is None else other_t.grad + g_b
                else:
                    # Both >= 2D (2D, 3D, 4D batched matmul with possible batch broadcasting)
                    if self.requires_grad:
                        b_t = np.swapaxes(b, -1, -2)
                        raw_a = np.matmul(g, b_t)
                        g_a = unbroadcast(raw_a, self.shape).astype(self.dtype, copy=False)
                        self.grad = g_a if self.grad is None else self.grad + g_a
                    if other_t.requires_grad:
                        a_t = np.swapaxes(a, -1, -2)
                        raw_b = np.matmul(a_t, g)
                        g_b = unbroadcast(raw_b, other_t.shape).astype(other_t.dtype, copy=False)
                        other_t.grad = g_b if other_t.grad is None else other_t.grad + g_b

            out._backward = _backward
        return out

    # -------------------------------------------------------------------------
    # Shape & Indexing Operations
    # -------------------------------------------------------------------------

    def reshape(self, *shape: int | tuple[int, ...]) -> Tensor:
        """Return a tensor with the same data and a new shape."""
        target_shape: tuple[int, ...]
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            target_shape = tuple(shape[0])
        else:
            target_shape = tuple(int(s) for s in shape)  # type: ignore[arg-type]

        req_grad = self.requires_grad
        out_data = self.data.reshape(target_shape)
        out = Tensor(
            out_data,
            requires_grad=req_grad,
            dtype=self.dtype,
            _children=(self,) if req_grad else (),
            _op="Reshape" if req_grad else "",
        )
        if req_grad:
            orig_shape = self.shape

            def _backward() -> None:
                assert out.grad is not None
                if self.requires_grad:
                    g = out.grad.reshape(orig_shape)
                    self.grad = g if self.grad is None else self.grad + g

            out._backward = _backward
        return out

    def transpose(self, *axes: int | tuple[int, ...] | None) -> Tensor:
        """Permute tensor dimensions.

        Supports:
        - `t.transpose()` -> reverses all axes
        - `t.transpose(ax0, ax1)` when `ndim > 2` and 2 ints given -> swaps `ax0` and `ax1`
          if not a full permutation, or full permutation when `len(axes) == ndim`.
        - `t.transpose((p0, p1, ...))` -> full axis permutation
        """
        perm: tuple[int, ...] | None
        if len(axes) == 0 or (len(axes) == 1 and axes[0] is None):
            perm = None
        elif len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            perm = tuple(axes[0])
        elif len(axes) == 2 and self.ndim > 2 and all(isinstance(a, int) for a in axes):
            # Swap two specific axes (PyTorch-style transpose(dim0, dim1))
            ax0, ax1 = int(axes[0]), int(axes[1])  # type: ignore[arg-type]
            perm_list = list(range(self.ndim))
            perm_list[ax0], perm_list[ax1] = perm_list[ax1], perm_list[ax0]
            perm = tuple(perm_list)
        else:
            perm = tuple(int(a) for a in axes)  # type: ignore[arg-type]

        req_grad = self.requires_grad
        out_data = np.transpose(self.data, axes=perm)
        out = Tensor(
            out_data,
            requires_grad=req_grad,
            dtype=self.dtype,
            _children=(self,) if req_grad else (),
            _op="Transpose" if req_grad else "",
        )
        if req_grad:
            inv_perm = None if perm is None else tuple(np.argsort(perm))

            def _backward() -> None:
                assert out.grad is not None
                if self.requires_grad:
                    g = np.transpose(out.grad, axes=inv_perm)
                    self.grad = g if self.grad is None else self.grad + g

            out._backward = _backward
        return out

    def __getitem__(self, idx: Any) -> Tensor:
        """Slice or index the tensor while tracking gradients via `np.add.at`."""
        req_grad = self.requires_grad
        out_data = self.data[idx]
        out = Tensor(
            out_data,
            requires_grad=req_grad,
            dtype=self.dtype,
            _children=(self,) if req_grad else (),
            _op="Slice" if req_grad else "",
        )
        if req_grad:

            def _backward() -> None:
                assert out.grad is not None
                if self.requires_grad:
                    g = np.zeros_like(self.data, dtype=self.dtype)
                    np.add.at(g, idx, out.grad)
                    self.grad = g if self.grad is None else self.grad + g

            out._backward = _backward
        return out

    # -------------------------------------------------------------------------
    # Reduction Operations
    # -------------------------------------------------------------------------

    def sum(
        self,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Tensor:
        """Sum of tensor elements over a given axis or axes."""
        req_grad = self.requires_grad
        out_data = self.data.sum(axis=axis, keepdims=keepdims)
        out = Tensor(
            out_data,
            requires_grad=req_grad,
            dtype=self.dtype,
            _children=(self,) if req_grad else (),
            _op="Sum" if req_grad else "",
        )

        if req_grad:
            orig_shape = self.shape
            norm_axes: tuple[int, ...] | None
            if axis is None:
                norm_axes = None
            elif isinstance(axis, int):
                norm_axes = (axis % self.ndim,) if self.ndim > 0 else ()
            else:
                norm_axes = tuple(a % self.ndim for a in axis)

            def _backward() -> None:
                assert out.grad is not None
                if self.requires_grad:
                    g = out.grad
                    if not keepdims and norm_axes is not None:
                        expanded_shape = [
                            1 if i in norm_axes else orig_shape[i] for i in range(len(orig_shape))
                        ]
                        g = g.reshape(expanded_shape)
                    g_broad = np.broadcast_to(g, orig_shape).astype(self.dtype, copy=True)
                    self.grad = g_broad if self.grad is None else self.grad + g_broad

            out._backward = _backward
        return out

    def mean(
        self,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Tensor:
        """Arithmetic mean over a given axis or axes."""
        req_grad = self.requires_grad
        out_data = self.data.mean(axis=axis, keepdims=keepdims)
        out = Tensor(
            out_data,
            requires_grad=req_grad,
            dtype=self.dtype,
            _children=(self,) if req_grad else (),
            _op="Mean" if req_grad else "",
        )

        if req_grad:
            orig_shape = self.shape
            norm_axes: tuple[int, ...] | None
            if axis is None:
                norm_axes = None
                count = max(self.size, 1)
            elif isinstance(axis, int):
                ax = axis % self.ndim if self.ndim > 0 else 0
                norm_axes = (ax,)
                count = orig_shape[ax] if self.ndim > 0 else 1
            else:
                norm_axes = tuple(a % self.ndim for a in axis)
                count = int(np.prod([orig_shape[a] for a in norm_axes]))

            def _backward() -> None:
                assert out.grad is not None
                if self.requires_grad:
                    g = out.grad
                    if not keepdims and norm_axes is not None:
                        expanded_shape = [
                            1 if i in norm_axes else orig_shape[i] for i in range(len(orig_shape))
                        ]
                        g = g.reshape(expanded_shape)
                    g_broad = (np.broadcast_to(g, orig_shape) / float(count)).astype(
                        self.dtype, copy=False
                    )
                    self.grad = g_broad if self.grad is None else self.grad + g_broad

            out._backward = _backward
        return out

    def max(
        self,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Tensor:
        """Maximum of tensor elements over a given axis or axes.

        Subgradient splits upstream gradient evenly among tied maximum entries.
        """
        req_grad = self.requires_grad
        out_data = self.data.max(axis=axis, keepdims=keepdims)
        out = Tensor(
            out_data,
            requires_grad=req_grad,
            dtype=self.dtype,
            _children=(self,) if req_grad else (),
            _op="Max" if req_grad else "",
        )

        if req_grad:
            orig_shape = self.shape
            norm_axes: tuple[int, ...] | None
            if axis is None:
                norm_axes = None
            elif isinstance(axis, int):
                norm_axes = (axis % self.ndim,) if self.ndim > 0 else ()
            else:
                norm_axes = tuple(a % self.ndim for a in axis)

            def _backward() -> None:
                assert out.grad is not None
                if self.requires_grad:
                    g = out.grad
                    max_keep = out_data
                    if not keepdims and norm_axes is not None:
                        expanded_shape = [
                            1 if i in norm_axes else orig_shape[i] for i in range(len(orig_shape))
                        ]
                        g = g.reshape(expanded_shape)
                        max_keep = max_keep.reshape(expanded_shape)

                    mask = (self.data == max_keep).astype(self.dtype)
                    tie_counts = mask.sum(axis=norm_axes, keepdims=True)
                    g_self = (mask / np.maximum(tie_counts, 1.0)) * g
                    self.grad = g_self if self.grad is None else self.grad + g_self

            out._backward = _backward
        return out

    # -------------------------------------------------------------------------
    # Nonlinear & Activation Primitives
    # -------------------------------------------------------------------------

    def exp(self) -> Tensor:
        """Elementwise exponential e^x."""
        req_grad = self.requires_grad
        out_data = np.exp(self.data)
        out = Tensor(
            out_data,
            requires_grad=req_grad,
            dtype=self.dtype,
            _children=(self,) if req_grad else (),
            _op="Exp" if req_grad else "",
        )
        if req_grad:

            def _backward() -> None:
                assert out.grad is not None
                if self.requires_grad:
                    g = (out.grad * out_data).astype(self.dtype, copy=False)
                    self.grad = g if self.grad is None else self.grad + g

            out._backward = _backward
        return out

    def log(self) -> Tensor:
        """Elementwise natural logarithm ln(x)."""
        req_grad = self.requires_grad
        out_data = np.log(self.data)
        out = Tensor(
            out_data,
            requires_grad=req_grad,
            dtype=self.dtype,
            _children=(self,) if req_grad else (),
            _op="Log" if req_grad else "",
        )
        if req_grad:

            def _backward() -> None:
                assert out.grad is not None
                if self.requires_grad:
                    g = (out.grad / self.data).astype(self.dtype, copy=False)
                    self.grad = g if self.grad is None else self.grad + g

            out._backward = _backward
        return out

    def tanh(self) -> Tensor:
        """Elementwise hyperbolic tangent tanh(x)."""
        req_grad = self.requires_grad
        out_data = np.tanh(self.data)
        out = Tensor(
            out_data,
            requires_grad=req_grad,
            dtype=self.dtype,
            _children=(self,) if req_grad else (),
            _op="Tanh" if req_grad else "",
        )
        if req_grad:

            def _backward() -> None:
                assert out.grad is not None
                if self.requires_grad:
                    g = (out.grad * (1.0 - out_data * out_data)).astype(self.dtype, copy=False)
                    self.grad = g if self.grad is None else self.grad + g

            out._backward = _backward
        return out

    def relu(self) -> Tensor:
        """Rectified Linear Unit max(0, x)."""
        req_grad = self.requires_grad
        out_data = np.maximum(self.data, 0.0)
        out = Tensor(
            out_data,
            requires_grad=req_grad,
            dtype=self.dtype,
            _children=(self,) if req_grad else (),
            _op="ReLU" if req_grad else "",
        )
        if req_grad:

            def _backward() -> None:
                assert out.grad is not None
                if self.requires_grad:
                    g = (out.grad * (self.data > 0.0)).astype(self.dtype, copy=False)
                    self.grad = g if self.grad is None else self.grad + g

            out._backward = _backward
        return out

    def gelu(self, approximate: str = "none") -> Tensor:
        """Gaussian Error Linear Unit activation.

        Args:
            approximate: Either `'none'` (exact Gaussian CDF via `erf`) or `'tanh'`
                (Hendrycks & Gimpel tanh approximation).
        """
        req_grad = self.requires_grad
        x = self.data.astype(np.float64, copy=False)

        if approximate == "none":
            inv_sqrt2 = float(1.0 / np.sqrt(2.0))
            inv_sqrt2pi = float(1.0 / np.sqrt(2.0 * np.pi))
            cdf = 0.5 * (1.0 + _erf_approx(x * inv_sqrt2))
            out_data = (x * cdf).astype(self.dtype, copy=False)
            out = Tensor(
                out_data,
                requires_grad=req_grad,
                dtype=self.dtype,
                _children=(self,) if req_grad else (),
                _op="GELU" if req_grad else "",
            )
            if req_grad:

                def _backward() -> None:
                    assert out.grad is not None
                    if self.requires_grad:
                        pdf = inv_sqrt2pi * np.exp(-0.5 * np.minimum(x * x, 200.0))
                        grad_gelu = cdf + x * pdf
                        g = (out.grad * grad_gelu).astype(self.dtype, copy=False)
                        self.grad = g if self.grad is None else self.grad + g

                out._backward = _backward
            return out

        if approximate == "tanh":
            k0 = float(np.sqrt(2.0 / np.pi))
            k1 = 0.044715
            inner = k0 * (x + k1 * (x**3))
            tanh_inner = np.tanh(inner)
            out_data = (0.5 * x * (1.0 + tanh_inner)).astype(self.dtype, copy=False)
            out = Tensor(
                out_data,
                requires_grad=req_grad,
                dtype=self.dtype,
                _children=(self,) if req_grad else (),
                _op="GELUTanh" if req_grad else "",
            )
            if req_grad:

                def _backward() -> None:
                    assert out.grad is not None
                    if self.requires_grad:
                        sech2 = 1.0 - tanh_inner * tanh_inner
                        d_inner = k0 * (1.0 + 3.0 * k1 * x * x)
                        grad_gelu = 0.5 * (1.0 + tanh_inner) + 0.5 * x * sech2 * d_inner
                        g = (out.grad * grad_gelu).astype(self.dtype, copy=False)
                        self.grad = g if self.grad is None else self.grad + g

                out._backward = _backward
            return out

        raise ValueError(
            f"Unsupported gelu approximate mode: {approximate!r}. Expected 'none' or 'tanh'."
        )

    def silu(self) -> Tensor:
        """Sigmoid Linear Unit (Swish): x * sigmoid(x), overflow-stabilized."""
        req_grad = self.requires_grad
        x = self.data.astype(np.float64, copy=False)
        # Overflow-safe sigmoid: for x >= 0, 1 / (1 + exp(-x)); for x < 0, exp(x) / (1 + exp(x))
        pos_mask = x >= 0.0
        z = np.exp(-np.abs(x))
        sig = np.where(pos_mask, 1.0 / (1.0 + z), z / (1.0 + z))
        out_data = (x * sig).astype(self.dtype, copy=False)

        out = Tensor(
            out_data,
            requires_grad=req_grad,
            dtype=self.dtype,
            _children=(self,) if req_grad else (),
            _op="SiLU" if req_grad else "",
        )
        if req_grad:

            def _backward() -> None:
                assert out.grad is not None
                if self.requires_grad:
                    grad_silu = sig * (1.0 + x * (1.0 - sig))
                    g = (out.grad * grad_silu).astype(self.dtype, copy=False)
                    self.grad = g if self.grad is None else self.grad + g

            out._backward = _backward
        return out
