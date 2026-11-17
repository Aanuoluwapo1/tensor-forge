"""Reproducible end-to-end training convergence, optimizer, and serialization tests."""

from __future__ import annotations

import numpy as np

from tensor_forge import (
    MLP,
    SGD,
    AdamW,
    Linear,
    Module,
    Parameter,
    Tensor,
    TransformerBlock,
    clip_grad_norm_,
    cross_entropy_loss,
    mse_loss,
)


class TinyTransformerClassifier(Module):
    """2-layer Transformer sequence classifier for end-to-end convergence testing."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_classes: int,
        rng: np.random.Generator,
    ) -> None:
        super().__init__()
        self.block1 = TransformerBlock(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=2 * d_model,
            causal=True,
            norm_type="rmsnorm",
            rng=rng,
        )
        self.block2 = TransformerBlock(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=2 * d_model,
            causal=True,
            norm_type="layernorm",
            rng=rng,
        )
        self.head = Linear(d_model, num_classes, rng=rng)

    def forward(self, x: Tensor) -> Tensor:
        h = self.block1(x)
        h = self.block2(h)
        # Mean-pool over sequence dimension (axis=1)
        pooled = h.mean(axis=1)
        return self.head(pooled)


class TestTrainingAndOptimizers:
    """Verify training convergence, gradient clipping, decoupled weight decay, and state_dict."""

    def test_mlp_classification_convergence_with_adamw(self) -> None:
        rng = np.random.default_rng(42)
        n_samples, in_dim, num_classes = 48, 12, 4
        x_np = rng.normal(size=(n_samples, in_dim))
        true_w = rng.normal(size=(in_dim, num_classes))
        y_np = np.argmax(x_np @ true_w, axis=-1)

        model = MLP(
            in_features=in_dim,
            hidden_features=[24, 16],
            out_features=num_classes,
            activation="gelu",
            rng=rng,
        )
        optimizer = AdamW(model.parameters(), lr=1.5e-2, weight_decay=1e-3)

        x_t = Tensor(x_np, requires_grad=False)
        losses: list[float] = []
        for _ in range(60):
            optimizer.zero_grad()
            logits = model(x_t)
            loss = cross_entropy_loss(logits, y_np, label_smoothing=0.02)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0] * 0.15, (
            f"Loss did not converge sufficiently: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )
        preds = np.argmax(model(x_t).data, axis=-1)
        accuracy = float(np.mean(preds == y_np))
        assert accuracy >= 0.95, f"Expected >=95% accuracy on synthetic task, got {accuracy:.2%}"

    def test_transformer_sequence_classification_convergence(self) -> None:
        rng = np.random.default_rng(1337)
        bsz, seq_len, d_model, num_classes = 16, 6, 12, 3
        x_np = rng.normal(size=(bsz, seq_len, d_model))
        y_np = np.arange(bsz) % num_classes
        # Inject class prototype into input sequences so task is learnable
        prototypes = rng.normal(scale=1.5, size=(num_classes, 1, d_model))
        x_np = x_np + prototypes[y_np]

        model = TinyTransformerClassifier(
            d_model=d_model, num_heads=3, num_classes=num_classes, rng=rng
        )
        optimizer = AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)

        x_t = Tensor(x_np, requires_grad=False)
        initial_loss = cross_entropy_loss(model(x_t), y_np).item()
        final_loss = initial_loss

        for _ in range(40):
            optimizer.zero_grad()
            logits = model(x_t)
            loss = cross_entropy_loss(logits, y_np)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            final_loss = loss.item()

        assert final_loss < initial_loss * 0.1, (
            f"Transformer loss failed to decrease >90%: {initial_loss:.4f} -> {final_loss:.4f}"
        )

    def test_sgd_momentum_and_nesterov_regression_convergence(self) -> None:
        rng = np.random.default_rng(7)
        x_np = rng.normal(size=(32, 8))
        target_w = rng.normal(size=(8, 2))
        y_np = x_np @ target_w + 0.5

        model = Linear(8, 2, rng=rng)
        optimizer = SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4, nesterov=True)

        x_t = Tensor(x_np)
        start_loss = mse_loss(model(x_t), y_np).item()
        for _ in range(80):
            optimizer.zero_grad()
            loss = mse_loss(model(x_t), y_np)
            loss.backward()
            optimizer.step()
        end_loss = mse_loss(model(x_t), y_np).item()
        assert end_loss < 1e-4 and end_loss < start_loss * 1e-3

    def test_clip_grad_norm_enforces_exact_l2_ceiling(self) -> None:
        p1 = Parameter(np.ones((4, 4)), dtype=np.float64)
        p2 = Parameter(np.ones((8,)), dtype=np.float64)
        p1.grad = np.full((4, 4), 3.0, dtype=np.float64)  # sum_sq = 16 * 9 = 144
        p2.grad = np.full((8,), 4.0, dtype=np.float64)  # sum_sq = 8 * 16 = 128 -> total = sqrt(272)

        max_norm = 2.5
        orig_norm = clip_grad_norm_([p1, p2], max_norm=max_norm, eps=1e-12)
        np.testing.assert_allclose(orig_norm, np.sqrt(272.0), rtol=1e-7)

        clipped_norm = np.sqrt(float(np.sum(p1.grad**2) + np.sum(p2.grad**2)))
        assert clipped_norm <= max_norm + 1e-6
        np.testing.assert_allclose(clipped_norm, max_norm, rtol=1e-5)

    def test_adamw_decoupled_weight_decay_on_zero_gradient(self) -> None:
        # When gradient is identically zero, AdamW should still decay weights by (1 - lr * wd)
        w_init = np.array([10.0, -4.0, 2.5], dtype=np.float64)
        p = Parameter(w_init.copy())
        lr = 0.05
        wd = 0.2
        opt = AdamW([p], lr=lr, weight_decay=wd)

        p.grad = np.zeros_like(w_init)
        opt.step()
        np.testing.assert_allclose(p.data, w_init * (1.0 - lr * wd), rtol=1e-12)

        # Verify decay_1d=False exempts 1D bias/norm parameters while decaying >=2D weights
        w2d = Parameter(np.full((2, 2), 4.0, dtype=np.float64))
        b1d = Parameter(np.full((2,), 4.0, dtype=np.float64))
        w2d.grad = np.zeros_like(w2d.data)
        b1d.grad = np.zeros_like(b1d.data)
        opt_no_1d = AdamW([w2d, b1d], lr=lr, weight_decay=wd, decay_1d=False)
        opt_no_1d.step()
        np.testing.assert_allclose(w2d.data, np.full((2, 2), 4.0 * (1.0 - lr * wd)), rtol=1e-12)
        np.testing.assert_allclose(b1d.data, np.full((2,), 4.0), rtol=1e-12)

    def test_clip_grad_norm_empty_and_zero_norm_safety(self) -> None:
        # Empty parameter list returns 0.0 without error
        assert clip_grad_norm_([], max_norm=1.0) == 0.0

        # Zero gradient norm returns 0.0 and leaves gradients finite zeros
        p = Parameter(np.ones((3, 3)))
        p.grad = np.zeros((3, 3))
        norm = clip_grad_norm_([p], max_norm=1.0)
        assert norm == 0.0
        assert np.all(p.grad == 0.0)

    def test_state_dict_serialization_and_deterministic_reload_parity(self) -> None:
        rng = np.random.default_rng(101)
        model_a = TinyTransformerClassifier(d_model=8, num_heads=2, num_classes=4, rng=rng)
        model_b = TinyTransformerClassifier(d_model=8, num_heads=2, num_classes=4, rng=rng)

        x = Tensor(rng.normal(size=(4, 5, 8)), requires_grad=True)
        # Before loading state_dict, outputs differ
        assert not np.allclose(model_a(x).data, model_b(x).data)

        state = model_a.state_dict()
        model_b.load_state_dict(state, strict=True)

        # Verify deep-copy isolation: mutating model_b in-place must not mutate saved state_dict
        first_key = next(iter(state.keys()))
        saved_snapshot = state[first_key].copy()
        dict(model_b.named_parameters())[first_key].data += 999.0
        np.testing.assert_allclose(state[first_key], saved_snapshot, rtol=1e-12)

        # Reload cleanly and verify forward + backward parity
        model_b.load_state_dict(state, strict=True)
        x_a = Tensor(x.data.copy(), requires_grad=True)
        x_b = Tensor(x.data.copy(), requires_grad=True)
        out_a = model_a(x_a)
        out_b = model_b(x_b)
        np.testing.assert_allclose(out_a.data, out_b.data, rtol=1e-12, atol=1e-12)

        out_a.sum().backward()
        out_b.sum().backward()
        assert x_a.grad is not None and x_b.grad is not None
        np.testing.assert_allclose(x_a.grad, x_b.grad, rtol=1e-12, atol=1e-12)
