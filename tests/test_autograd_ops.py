"""Finite-difference gradient verification tests for all differentiable primitives."""

from __future__ import annotations

import numpy as np
import pytest

from tensor_forge import (
    MLP,
    LayerNorm,
    Linear,
    RMSNorm,
    Tensor,
    TransformerBlock,
    cross_entropy_loss,
    layer_norm,
    log_softmax,
    mse_loss,
    numerical_gradient_check,
    rms_norm,
    scaled_dot_product_attention,
    unbroadcast,
)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(2026)


class TestArithmeticAndBroadcastingGradients:
    """Verify binary/unary arithmetic and arbitrary NumPy broadcasting reductions."""

    @pytest.mark.parametrize(
        ("shape_a", "shape_b"),
        [
            ((3, 4), (3, 4)),
            ((3, 4), (4,)),
            ((3, 4), (3, 1)),
            ((1, 4), (3, 1)),
            ((2, 3, 4), (4,)),
            ((2, 1, 4), (1, 3, 1)),
            ((), (2, 3)),
        ],
    )
    def test_add_sub_mul_broadcast_gradcheck(
        self,
        rng: np.random.Generator,
        shape_a: tuple[int, ...],
        shape_b: tuple[int, ...],
    ) -> None:
        a = Tensor(rng.normal(size=shape_a), requires_grad=True)
        b = Tensor(rng.normal(size=shape_b), requires_grad=True)

        assert numerical_gradient_check(lambda x, y: x + y, [a, b])
        assert numerical_gradient_check(lambda x, y: x - y, [a, b])
        assert numerical_gradient_check(lambda x, y: x * y, [a, b])

    def test_div_and_pow_gradcheck(self, rng: np.random.Generator) -> None:
        a = Tensor(rng.uniform(0.5, 2.5, size=(3, 4)), requires_grad=True)
        b = Tensor(rng.uniform(0.8, 2.0, size=(1, 4)), requires_grad=True)

        assert numerical_gradient_check(lambda x, y: x / y, [a, b])
        assert numerical_gradient_check(lambda x: x**2.5, [a])
        assert numerical_gradient_check(lambda x, y: x**y, [a, b])
        assert numerical_gradient_check(lambda x: -x + (3.0 - x) * (2.0 / x), [a])

    def test_unbroadcast_direct_unit(self, rng: np.random.Generator) -> None:
        grad = rng.normal(size=(2, 3, 4, 5))
        reduced = unbroadcast(grad, (1, 4, 1))
        assert reduced.shape == (1, 4, 1)
        expected = grad.sum(axis=(0, 1, 3)).reshape(1, 4, 1)
        np.testing.assert_allclose(reduced, expected)

    def test_diamond_dag_shared_node_accumulation(self, rng: np.random.Generator) -> None:
        x = Tensor(rng.normal(size=(3, 3)), requires_grad=True)

        def diamond_fn(t: Tensor) -> Tensor:
            branch1 = (t * t).tanh()
            branch2 = (t + 1.5).silu()
            return (branch1 * branch2) + t

        assert numerical_gradient_check(diamond_fn, [x])

    def test_float32_dtype_preservation_and_0d_scalar_semantics(self) -> None:
        # 1. Verify float32 inputs never upcast to float64 in forward or backward
        a = Tensor(np.array([[1.5, 2.0], [0.5, 1.0]], dtype=np.float32), requires_grad=True)
        b = Tensor(np.array([[0.5, 1.5], [2.0, 0.5]], dtype=np.float32), requires_grad=True)
        w = Tensor(np.ones(2, dtype=np.float32), requires_grad=True)
        bias = Tensor(np.zeros(2, dtype=np.float32), requires_grad=True)

        y = ((a + 1.0) * (b - 0.25) / 1.5) ** 2.0
        y = (y @ b).gelu(approximate="none").gelu(approximate="tanh").silu().relu().tanh()
        y_ln = layer_norm(y, w, bias)
        y_rn = rms_norm(y_ln, w)
        y_lsm = log_softmax(y_rn)
        q = a.reshape(1, 1, 2, 2)
        attn = scaled_dot_product_attention(q, q, q, causal_mask=True)
        loss = cross_entropy_loss(y_lsm, np.array([0, 1], dtype=np.int64)) + attn.sum()

        assert y.dtype == np.float32
        assert y_ln.dtype == np.float32
        assert y_rn.dtype == np.float32
        assert y_lsm.dtype == np.float32
        assert attn.dtype == np.float32
        assert loss.dtype == np.float32

        loss.backward()
        for t in (a, b, w, bias):
            assert t.grad is not None
            assert t.grad.dtype == np.float32

        # 2. Verify 0-D scalar Tensor (shape == ()) forward, item(), unbroadcast(), and backward()
        s = Tensor(3.0, requires_grad=True, dtype=np.float32)
        assert s.shape == ()
        v = s * s + s
        assert v.shape == ()
        np.testing.assert_allclose(v.item(), 12.0)
        v.backward()
        assert s.grad is not None
        assert s.grad.shape == ()
        assert s.grad.dtype == np.float32
        np.testing.assert_allclose(s.grad, 7.0)


class TestMatmulShapeAndReductionGradients:
    """Verify 1D, 2D, 3D, and 4D batched matrix multiplication, reshape, transpose, and reductions."""

    @pytest.mark.parametrize(
        ("shape_a", "shape_b"),
        [
            ((5,), (5,)),
            ((4, 5), (5, 3)),
            ((5,), (5, 3)),
            ((4, 5), (5,)),
            ((2, 4, 5), (2, 5, 3)),
            ((2, 4, 5), (5, 3)),
            ((1, 4, 5), (2, 5, 3)),
            ((2, 3, 4, 6), (2, 1, 6, 5)),
        ],
    )
    def test_matmul_variants_gradcheck(
        self,
        rng: np.random.Generator,
        shape_a: tuple[int, ...],
        shape_b: tuple[int, ...],
    ) -> None:
        a = Tensor(rng.normal(size=shape_a), requires_grad=True)
        b = Tensor(rng.normal(size=shape_b), requires_grad=True)
        assert numerical_gradient_check(lambda x, y: x @ y, [a, b])

    def test_reshape_transpose_slice_gradcheck(self, rng: np.random.Generator) -> None:
        x = Tensor(rng.normal(size=(2, 3, 4)), requires_grad=True)
        assert numerical_gradient_check(lambda t: t.reshape(6, 4).transpose() * 1.7, [x])
        assert numerical_gradient_check(lambda t: t.transpose(0, 2, 1)[:, 1:3, ::2], [x])
        assert numerical_gradient_check(lambda t: t.transpose(1, 2), [x])

    @pytest.mark.parametrize("axis", [None, 0, 1, -1, (0, 2)])
    @pytest.mark.parametrize("keepdims", [False, True])
    def test_sum_and_mean_gradcheck(
        self,
        rng: np.random.Generator,
        axis: int | tuple[int, ...] | None,
        keepdims: bool,
    ) -> None:
        x = Tensor(rng.normal(size=(2, 3, 4)), requires_grad=True)
        w = Tensor(rng.normal(size=(2, 3, 4)), requires_grad=False)
        assert numerical_gradient_check(lambda t: (t * w).sum(axis=axis, keepdims=keepdims), [x])
        assert numerical_gradient_check(lambda t: (t * w).mean(axis=axis, keepdims=keepdims), [x])

    def test_max_reduction_gradcheck(self, rng: np.random.Generator) -> None:
        # Distinct values so max is uniquely differentiable around eps
        data = np.arange(12, dtype=np.float64).reshape(3, 4) * 0.37 + rng.uniform(
            -0.05, 0.05, size=(3, 4)
        )
        x = Tensor(data, requires_grad=True)
        assert numerical_gradient_check(lambda t: (t * 1.3).max(axis=-1, keepdims=False), [x])
        assert numerical_gradient_check(lambda t: (t * 1.3).max(axis=0, keepdims=True), [x])
        assert numerical_gradient_check(lambda t: (t * 1.3).max(), [x])


class TestActivationsAndStabilizedKernels:
    """Verify activations and numerically stabilized neural primitives via finite differences."""

    def test_nonlinear_activations_gradcheck(self, rng: np.random.Generator) -> None:
        x = Tensor(rng.uniform(-2.0, 2.0, size=(3, 4)), requires_grad=True)
        pos_x = Tensor(rng.uniform(0.2, 3.0, size=(3, 4)), requires_grad=True)
        # Keep away from 0.0 kink for ReLU finite-difference check
        relu_x = Tensor(np.array([[-1.2, -0.4, 0.5], [1.1, -2.3, 0.8]]), requires_grad=True)

        assert numerical_gradient_check(lambda t: t.exp(), [x])
        assert numerical_gradient_check(lambda t: t.log(), [pos_x])
        assert numerical_gradient_check(lambda t: t.tanh(), [x])
        assert numerical_gradient_check(lambda t: t.relu(), [relu_x])
        assert numerical_gradient_check(lambda t: t.gelu(approximate="none"), [x])
        assert numerical_gradient_check(lambda t: t.gelu(approximate="tanh"), [x])
        assert numerical_gradient_check(lambda t: t.silu(), [x])

    def test_log_softmax_and_cross_entropy_gradcheck(self, rng: np.random.Generator) -> None:
        logits = Tensor(rng.normal(size=(4, 6)), requires_grad=True)
        targets_idx = np.array([1, 4, 0, 5], dtype=np.int64)
        w = Tensor(rng.normal(size=(4, 6)), requires_grad=False)

        assert numerical_gradient_check(lambda z: (log_softmax(z, axis=-1) * w).sum(), [logits])
        assert numerical_gradient_check(
            lambda z: cross_entropy_loss(z, targets_idx, label_smoothing=0.0, reduction="mean"),
            [logits],
        )
        assert numerical_gradient_check(
            lambda z: cross_entropy_loss(z, targets_idx, label_smoothing=0.15, reduction="sum"),
            [logits],
        )

    def test_mse_loss_gradcheck(self, rng: np.random.Generator) -> None:
        preds = Tensor(rng.normal(size=(4, 5)), requires_grad=True)
        targets = Tensor(rng.normal(size=(4, 5)), requires_grad=False)
        assert numerical_gradient_check(lambda p: mse_loss(p, targets, reduction="mean"), [preds])

    def test_layer_norm_and_rms_norm_gradcheck(self, rng: np.random.Generator) -> None:
        x = Tensor(rng.normal(size=(2, 3, 6)), requires_grad=True)
        gamma = Tensor(rng.normal(1.0, 0.2, size=(6,)), requires_grad=True)
        beta = Tensor(rng.normal(0.0, 0.2, size=(6,)), requires_grad=True)
        upstream = Tensor(rng.normal(size=(2, 3, 6)), requires_grad=False)

        assert numerical_gradient_check(
            lambda inp, w, b: (layer_norm(inp, w, b, eps=1e-5) * upstream).sum(),
            [x, gamma, beta],
        )
        assert numerical_gradient_check(
            lambda inp, w: (rms_norm(inp, w, eps=1e-6) * upstream).sum(),
            [x, gamma],
        )

    @pytest.mark.parametrize("causal", [False, True])
    def test_scaled_dot_product_attention_gradcheck(
        self,
        rng: np.random.Generator,
        causal: bool,
    ) -> None:
        q = Tensor(rng.normal(size=(2, 2, 4, 6)), requires_grad=True)
        k = Tensor(rng.normal(size=(2, 2, 4, 6)), requires_grad=True)
        v = Tensor(rng.normal(size=(2, 2, 4, 6)), requires_grad=True)
        upstream = Tensor(rng.normal(size=(2, 2, 4, 6)), requires_grad=False)

        assert numerical_gradient_check(
            lambda q_, k_, v_: (
                scaled_dot_product_attention(q_, k_, v_, causal_mask=causal) * upstream
            ).sum(),
            [q, k, v],
        )

    def test_nn_modules_end_to_end_gradcheck(self, rng: np.random.Generator) -> None:
        x = Tensor(rng.normal(size=(2, 3, 8)), requires_grad=True)
        block = TransformerBlock(d_model=8, num_heads=2, d_ff=16, causal=True, rng=rng)
        head = Linear(8, 4, rng=rng)
        targets = np.array([[0, 2, 1], [3, 1, 0]], dtype=np.int64)

        params = [x, *block.parameters(), *head.parameters()]

        def forward_loss(*tensors: Tensor) -> Tensor:
            inp = tensors[0]
            h = block(inp)
            logits = head(h)
            return cross_entropy_loss(logits, targets, label_smoothing=0.1)

        assert numerical_gradient_check(forward_loss, params)

    def test_mlp_and_norms_module_gradcheck(self, rng: np.random.Generator) -> None:
        x = Tensor(rng.normal(size=(3, 6)), requires_grad=True)
        mlp = MLP(in_features=6, hidden_features=10, out_features=4, activation="silu", rng=rng)
        ln = LayerNorm(4)
        rn = RMSNorm(4)

        assert numerical_gradient_check(
            lambda inp, *_params: rn(ln(mlp(inp))).sum(),
            [x, *mlp.parameters(), *ln.parameters(), *rn.parameters()],
        )
