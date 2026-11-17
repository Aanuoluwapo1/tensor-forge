"""Adversarial numerical stability tests under extreme magnitudes and ill-conditioned inputs."""

from __future__ import annotations

import numpy as np
import pytest

from tensor_forge import (
    Tensor,
    cross_entropy_loss,
    layer_norm,
    log_softmax,
    rms_norm,
    scaled_dot_product_attention,
)


class TestNumericalStabilityHardening:
    """Verify zero NaN/Inf forward outputs and finite gradients under extreme inputs."""

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_extreme_logits_log_softmax_and_cross_entropy(self, dtype: type[np.floating]) -> None:
        # In float32, exp(89) overflows to +inf and exp(-105) underflows to 0.
        # Here we test ±1e4 which overflows even float64 exp(710).
        raw = np.array(
            [
                [1e4, -1e4, 0.0, 5e3],
                [-1e4, -1e4, -1e4, -1e4],
                [1e4, 1e4, 1e4, 1e4],
            ],
            dtype=dtype,
        )
        logits = Tensor(raw, requires_grad=True, dtype=dtype)
        lsm = log_softmax(logits, axis=-1)

        assert np.all(np.isfinite(lsm.data)), f"Non-finite log_softmax output: {lsm.data}"
        # Row 0: argmax is index 0 -> log_softmax[0, 0] == 0.0, log_softmax[0, 1] == -2e4
        np.testing.assert_allclose(lsm.data[0, 0], 0.0, atol=1e-5)
        np.testing.assert_allclose(lsm.data[0, 1], -2e4, rtol=1e-5)
        # Uniform rows (rows 1 and 2) must equal -log(4)
        np.testing.assert_allclose(lsm.data[1], -np.log(4.0), rtol=1e-5)
        np.testing.assert_allclose(lsm.data[2], -np.log(4.0), rtol=1e-5)

        targets = np.array([1, 2, 0], dtype=np.int64)
        loss = cross_entropy_loss(logits, targets, label_smoothing=0.05)
        assert np.isfinite(loss.item())

        loss.backward()
        assert logits.grad is not None
        assert np.all(np.isfinite(logits.grad)), f"Non-finite gradients: {logits.grad}"
        # Row-wise gradient of cross-entropy with respect to logits must sum to 0
        np.testing.assert_allclose(logits.grad.sum(axis=-1), np.zeros(3), atol=1e-5)

    def test_subnormal_and_tiny_magnitude_stability(self) -> None:
        tiny = np.array(
            [[1e-30, 2e-30, -1e-30, 0.0], [1e-35, 1e-35, 1e-35, 1e-35]], dtype=np.float64
        )
        x = Tensor(tiny, requires_grad=True)
        out = log_softmax(x) + x.silu() + x.gelu(approximate="none")
        loss = out.sum()
        loss.backward()

        assert np.all(np.isfinite(out.data))
        assert x.grad is not None
        assert np.all(np.isfinite(x.grad))

    def test_catastrophic_cancellation_and_zero_variance_layernorm(self) -> None:
        # Classic catastrophic cancellation test: large constant offset 1e7 + small signal in float32.
        # Naive E[X^2] - E[X]^2 in float32 yields 0 or negative variance -> NaN!
        signal = np.array([[-1.5, -0.5, 0.5, 1.5], [0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        shifted = signal + np.float32(1e7)

        x = Tensor(shifted, requires_grad=True, dtype=np.float32)
        w = Tensor(np.ones(4, dtype=np.float32), requires_grad=True, dtype=np.float32)
        b = Tensor(np.zeros(4, dtype=np.float32), requires_grad=True, dtype=np.float32)

        y = layer_norm(x, w, b, eps=1e-5)
        assert np.all(np.isfinite(y.data))

        # Row 0 must have zero mean and unit variance despite the 1e7 offset
        np.testing.assert_allclose(y.data[0].mean(), 0.0, atol=1e-5)
        np.testing.assert_allclose(y.data[0].var(), 1.0, atol=1e-4)
        # Row 1 has exact zero variance -> normalized output must be identically 0.0
        np.testing.assert_allclose(y.data[1], np.zeros(4), atol=1e-6)

        y.sum().backward()
        assert x.grad is not None and w.grad is not None and b.grad is not None
        assert np.all(np.isfinite(x.grad))
        assert np.all(np.isfinite(w.grad))
        assert np.all(np.isfinite(b.grad))

    def test_rmsnorm_zero_and_large_inputs(self) -> None:
        x = Tensor(
            np.array([[0.0, 0.0, 0.0, 0.0], [1e4, -1e4, 1e4, -1e4]], dtype=np.float32),
            requires_grad=True,
            dtype=np.float32,
        )
        w = Tensor(np.ones(4, dtype=np.float32), requires_grad=True, dtype=np.float32)
        y = rms_norm(x, w, eps=1e-6)
        assert np.all(np.isfinite(y.data))
        np.testing.assert_allclose(np.abs(y.data[1]), np.ones(4), rtol=1e-5)

        (
            y * Tensor(np.array([[1.0, 2.0, 3.0, 4.0], [1.0, -2.0, 0.5, -1.0]], dtype=np.float32))
        ).sum().backward()
        assert x.grad is not None and np.all(np.isfinite(x.grad))

    def test_causal_attention_extreme_scores_and_mask_stability(self) -> None:
        # Extreme Q/K magnitudes (1e3) + causal masking: verifies no 0 * (-inf) = NaN
        rng = np.random.default_rng(99)
        q = Tensor(rng.normal(scale=100.0, size=(2, 2, 6, 8)), requires_grad=True)
        k = Tensor(rng.normal(scale=100.0, size=(2, 2, 6, 8)), requires_grad=True)
        v = Tensor(rng.normal(scale=1.0, size=(2, 2, 6, 8)), requires_grad=True)

        out = scaled_dot_product_attention(q, k, v, causal_mask=True)
        assert np.all(np.isfinite(out.data))

        # First query token (t=0) in causal attention can only attend to k[t=0], so out[..., 0, :] == v[..., 0, :]
        np.testing.assert_allclose(out.data[:, :, 0, :], v.data[:, :, 0, :], atol=1e-12)

        loss = (out * Tensor(rng.normal(size=out.shape))).sum()
        loss.backward()
        assert q.grad is not None and k.grad is not None and v.grad is not None
        assert np.all(np.isfinite(q.grad))
        assert np.all(np.isfinite(k.grad))
        assert np.all(np.isfinite(v.grad))

    def test_deep_dag_iterative_backward_no_recursion_limit(self) -> None:
        # Depth 2500 exceeds Python's default sys.getrecursionlimit() (1000)
        x = Tensor([1.0001, -0.9999], requires_grad=True)
        h = x
        for _ in range(2500):
            h = h * 0.999 + 0.0001
        loss = h.sum()
        loss.backward()
        assert x.grad is not None
        assert np.all(np.isfinite(x.grad))
        expected_grad = 0.999**2500
        np.testing.assert_allclose(x.grad, [expected_grad, expected_grad], rtol=1e-6)

    def test_relu_kink_and_max_tie_splitting_subgradients(self) -> None:
        # 1. ReLU at x == 0.0 must evaluate to 0.0 with subgradient 0.0 (no NaN/warning)
        x_relu = Tensor(np.array([-2.0, 0.0, 3.0], dtype=np.float64), requires_grad=True)
        out_relu = x_relu.relu()
        np.testing.assert_allclose(out_relu.data, [0.0, 0.0, 3.0])
        out_relu.sum().backward()
        assert x_relu.grad is not None
        np.testing.assert_allclose(x_relu.grad, [0.0, 0.0, 1.0])

        # 2. Max with tied maximum entries splits upstream gradient evenly (1/k)
        x_max = Tensor(np.array([[5.0, 5.0, 1.0, 5.0], [2.0, 4.0, 4.0, -1.0]]), requires_grad=True)
        out_max = x_max.max(axis=-1)
        out_max.sum().backward()
        assert x_max.grad is not None
        np.testing.assert_allclose(x_max.grad[0], [1.0 / 3.0, 1.0 / 3.0, 0.0, 1.0 / 3.0])
        np.testing.assert_allclose(x_max.grad[1], [0.0, 0.5, 0.5, 0.0])

        # 3. Custom boolean and additive -inf attn_mask in scaled_dot_product_attention
        q = Tensor(np.ones((2, 4)), requires_grad=True)
        k = Tensor(np.ones((2, 4)), requires_grad=True)
        v = Tensor(np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]), requires_grad=True)
        bool_mask = np.array([[True, False], [True, True]], dtype=bool)
        out_attn = scaled_dot_product_attention(q, k, v, attn_mask=bool_mask)
        out_attn.sum().backward()
        assert q.grad is not None and np.all(np.isfinite(q.grad))
        np.testing.assert_allclose(out_attn.data[0], [1.0, 2.0, 3.0, 4.0])
