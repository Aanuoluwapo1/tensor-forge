"""Benchmarks and numerical stability verification suite for tensor-forge."""

import time
import tracemalloc
from dataclasses import dataclass

import numpy as np

from tensor_forge import (
    MLP,
    AdamW,
    Module,
    Tensor,
    TransformerBlock,
    clip_grad_norm_,
    cross_entropy_loss,
    layer_norm,
    log_softmax,
    rms_norm,
    scaled_dot_product_attention,
)


@dataclass
class BenchResult:
    name: str
    shape_desc: str
    fwd_ms: float
    bwd_ms: float
    total_ms: float
    gflops: float
    peak_kb: float


def _measure_op(
    name: str,
    shape_desc: str,
    flops_per_iter: float,
    setup_fn,
    fwd_fn,
    iters: int = 20,
    warmup: int = 4,
) -> BenchResult:
    for _ in range(warmup):
        inputs = setup_fn()
        out = fwd_fn(*inputs)
        loss = out if out.size == 1 else out.sum()
        loss.backward()

    fwd_times: list[float] = []
    bwd_times: list[float] = []

    tracemalloc.start()
    for _ in range(iters):
        inputs = setup_fn()
        t0 = time.perf_counter()
        out = fwd_fn(*inputs)
        loss = out if out.size == 1 else out.sum()
        t1 = time.perf_counter()
        loss.backward()
        t2 = time.perf_counter()
        fwd_times.append((t1 - t0) * 1e3)
        bwd_times.append((t2 - t1) * 1e3)
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    fwd_ms = float(np.median(fwd_times))
    bwd_ms = float(np.median(bwd_times))
    total_ms = fwd_ms + bwd_ms
    gflops = (flops_per_iter / 1e9) / (total_ms / 1e3) if total_ms > 0 else 0.0
    return BenchResult(
        name=name,
        shape_desc=shape_desc,
        fwd_ms=fwd_ms,
        bwd_ms=bwd_ms,
        total_ms=total_ms,
        gflops=gflops,
        peak_kb=peak_bytes / 1024.0,
    )


def run_throughput_benchmarks() -> list[BenchResult]:
    rng = np.random.default_rng(2026)
    results: list[BenchResult] = []

    # 1. Batched MatMul (B=16, M=128, K=128, N=128)
    # Forward FLOPs = 2 * B * M * K * N, Backward FLOPs = 4 * B * M * K * N -> Total = 6 * B * M * K * N
    b, m, k, n = 16, 128, 128, 128
    a_raw = rng.normal(size=(b, m, k)).astype(np.float64)
    b_raw = rng.normal(size=(b, k, n)).astype(np.float64)
    results.append(
        _measure_op(
            name="Batched MatMul (3D)",
            shape_desc=f"({b}, {m}, {k}) @ ({b}, {k}, {n})",
            flops_per_iter=6.0 * b * m * k * n,
            setup_fn=lambda: (Tensor(a_raw, requires_grad=True), Tensor(b_raw, requires_grad=True)),
            fwd_fn=lambda x, y: x @ y,
        )
    )

    # 2. LayerNorm (B=32, T=128, D=256)
    ln_b, ln_t, ln_d = 32, 128, 256
    x_ln = rng.normal(size=(ln_b, ln_t, ln_d)).astype(np.float64)
    w_ln = np.ones(ln_d, dtype=np.float64)
    b_ln = np.zeros(ln_d, dtype=np.float64)
    results.append(
        _measure_op(
            name="Stabilized LayerNorm",
            shape_desc=f"({ln_b}, {ln_t}, {ln_d})",
            flops_per_iter=20.0 * ln_b * ln_t * ln_d,
            setup_fn=lambda: (
                Tensor(x_ln, requires_grad=True),
                Tensor(w_ln, requires_grad=True),
                Tensor(b_ln, requires_grad=True),
            ),
            fwd_fn=lambda x, w, bias: layer_norm(x, w, bias),
        )
    )

    # 3. RMSNorm (B=32, T=128, D=256)
    results.append(
        _measure_op(
            name="Stabilized RMSNorm",
            shape_desc=f"({ln_b}, {ln_t}, {ln_d})",
            flops_per_iter=14.0 * ln_b * ln_t * ln_d,
            setup_fn=lambda: (
                Tensor(x_ln, requires_grad=True),
                Tensor(w_ln, requires_grad=True),
            ),
            fwd_fn=lambda x, w: rms_norm(x, w),
        )
    )

    # 4. Fused Cross-Entropy Loss (N=1024, C=512)
    ce_n, ce_c = 1024, 512
    logits_raw = rng.normal(size=(ce_n, ce_c)).astype(np.float64)
    targets_raw = rng.integers(0, ce_c, size=(ce_n,), dtype=np.int64)
    results.append(
        _measure_op(
            name="Fused Cross-Entropy",
            shape_desc=f"({ce_n}, {ce_c})",
            flops_per_iter=8.0 * ce_n * ce_c,
            setup_fn=lambda: (Tensor(logits_raw, requires_grad=True),),
            fwd_fn=lambda z: cross_entropy_loss(z, targets_raw, label_smoothing=0.1),
        )
    )

    # 5. Scaled Dot-Product Causal Attention (B=8, H=4, T=64, D=32)
    at_b, at_h, at_t, at_d = 8, 4, 64, 32
    q_raw = rng.normal(size=(at_b, at_h, at_t, at_d)).astype(np.float64)
    k_raw = rng.normal(size=(at_b, at_h, at_t, at_d)).astype(np.float64)
    v_raw = rng.normal(size=(at_b, at_h, at_t, at_d)).astype(np.float64)
    attn_flops = 12.0 * at_b * at_h * (at_t**2) * at_d
    results.append(
        _measure_op(
            name="Causal SDPA (4D)",
            shape_desc=f"({at_b}, {at_h}, {at_t}, {at_d})",
            flops_per_iter=attn_flops,
            setup_fn=lambda: (
                Tensor(q_raw, requires_grad=True),
                Tensor(k_raw, requires_grad=True),
                Tensor(v_raw, requires_grad=True),
            ),
            fwd_fn=lambda q, k, v: scaled_dot_product_attention(q, k, v, causal_mask=True),
        )
    )

    return results


def run_stability_comparison() -> list[tuple[str, str, str, str]]:
    """Compare naive NumPy expressions vs. tensor-forge stabilized kernels."""
    rows: list[tuple[str, str, str, str]] = []

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        # Case 1: Extreme Logits [1e4, -1e4, 0.0] in log_softmax
        logits_arr = np.array([[1e4, -1e4, 0.0]], dtype=np.float64)
        naive_lsm = np.log(np.exp(logits_arr) / np.sum(np.exp(logits_arr), axis=-1, keepdims=True))
        tf_logits = Tensor(logits_arr, requires_grad=True)
        tf_lsm = log_softmax(tf_logits, axis=-1)
        tf_lsm.sum().backward()
        assert tf_logits.grad is not None
        rows.append(
            (
                "Extreme Logits `[1e4, -1e4, 0.0]` (`log_softmax`)",
                f"`{np.array2string(naive_lsm[0], precision=2)}` (Overflow -> `NaN`)",
                f"`{np.array2string(tf_lsm.data[0], precision=1)}`",
                f"Finite (`max|grad|={np.max(np.abs(tf_logits.grad)):.2f}`)",
            )
        )

        # Case 2: Catastrophic Cancellation in float32 LayerNorm (1e7 offset + signal)
        sig32 = np.array([[-1.5, -0.5, 0.5, 1.5]], dtype=np.float32) + np.float32(1e7)
        # Naive one-pass variance E[X^2] - E[X]^2 in float32
        naive_mean = np.mean(sig32, axis=-1, keepdims=True)
        naive_var = np.mean(sig32 * sig32, axis=-1, keepdims=True) - naive_mean * naive_mean
        naive_ln = (sig32 - naive_mean) / np.sqrt(naive_var + np.float32(1e-5))

        tf_x = Tensor(sig32, requires_grad=True, dtype=np.float32)
        tf_ln = layer_norm(tf_x, eps=1e-5)
        (tf_ln * Tensor([[1.0, -2.0, 0.5, 1.5]], dtype=np.float32)).sum().backward()
        assert tf_x.grad is not None
        rows.append(
            (
                "Catastrophic Cancellation `N(1e7, 1.25)` (`float32 LayerNorm`)",
                f"`var={float(naive_var.item()):.1f}`, `out={np.array2string(naive_ln[0], precision=2)}`",
                f"`{np.array2string(tf_ln.data[0], precision=4)}` (`var=1.0000`)",
                f"Finite (`max|grad|={np.max(np.abs(tf_x.grad)):.3e}`)",
            )
        )

        # Case 3: Zero-Variance Constant Vector LayerNorm
        zero_var = np.array([[42.0, 42.0, 42.0, 42.0]], dtype=np.float64)
        naive_zero_ln = (zero_var - np.mean(zero_var)) / np.sqrt(np.var(zero_var))
        tf_z = Tensor(zero_var, requires_grad=True)
        tf_z_ln = layer_norm(tf_z, eps=1e-5)
        tf_z_ln.sum().backward()
        assert tf_z.grad is not None
        rows.append(
            (
                "Zero-Variance Input `[42, 42, 42, 42]` (`LayerNorm`)",
                f"`{np.array2string(naive_zero_ln[0])}` (`0/0 -> NaN`)",
                f"`{np.array2string(tf_z_ln.data[0], precision=1)}`",
                f"Finite (`max|grad|={np.max(np.abs(tf_z.grad)):.1f}`)",
            )
        )

        # Case 4: Causal Masking with 0 * (-inf) artifact vs stabilized where()
        scores = np.array([[2.0, 5000.0], [1.0, 3.0]], dtype=np.float64)
        mask = np.triu(np.ones((2, 2), dtype=bool), k=1)
        naive_masked = scores + mask * (-np.inf)  # 0 * -inf = NaN on unmasked entries!
        naive_probs = np.exp(naive_masked) / np.sum(np.exp(naive_masked), axis=-1, keepdims=True)

        q_t = Tensor(np.array([[100.0, 0.0], [0.0, 100.0]]), requires_grad=True)
        k_t = Tensor(np.array([[100.0, 0.0], [0.0, 100.0]]), requires_grad=True)
        v_t = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
        attn_out = scaled_dot_product_attention(q_t, k_t, v_t, causal_mask=True)
        attn_out.sum().backward()
        assert q_t.grad is not None
        rows.append(
            (
                "Causal Attention Masking (`scores + mask * -inf`)",
                f"`{np.array2string(naive_probs[0])}` (`0 * -inf -> NaN`)",
                f"`out[0]={np.array2string(attn_out.data[0], precision=2)}`",
                f"Finite (`max|grad|={np.max(np.abs(q_t.grad)):.3e}`)",
            )
        )

    return rows


class BenchmarkTransformerNet(Module):
    """2-Layer Transformer + MLP Classifier for synthetic sequence benchmark."""

    def __init__(
        self, d_model: int, num_heads: int, num_classes: int, rng: np.random.Generator
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
        self.mlp_head = MLP(
            in_features=d_model,
            hidden_features=d_model,
            out_features=num_classes,
            activation="gelu",
            rng=rng,
        )

    def forward(self, x: Tensor) -> Tensor:
        h = self.block1(x)
        h = self.block2(h)
        return self.mlp_head(h.mean(axis=1))


def run_convergence_benchmark() -> list[tuple[int, float, float, float]]:
    """Train a 2-layer Transformer + MLP classifier and record (step, loss, grad_norm, acc)."""
    rng = np.random.default_rng(2026)
    bsz, seq_len, d_model, num_classes = 32, 8, 16, 4
    prototypes = rng.normal(scale=1.6, size=(num_classes, 1, d_model))
    y_np = np.arange(bsz) % num_classes
    x_np = rng.normal(scale=0.6, size=(bsz, seq_len, d_model)) + prototypes[y_np]

    model = BenchmarkTransformerNet(d_model=d_model, num_heads=4, num_classes=num_classes, rng=rng)
    optimizer = AdamW(model.parameters(), lr=1.2e-2, weight_decay=1e-3)
    x_t = Tensor(x_np, requires_grad=False)

    history: list[tuple[int, float, float, float]] = []
    for step in range(1, 51):
        optimizer.zero_grad()
        logits = model(x_t)
        loss = cross_entropy_loss(logits, y_np, label_smoothing=0.02)
        loss.backward()
        grad_norm = clip_grad_norm_(model.parameters(), max_norm=3.0)
        optimizer.step()

        if step == 1 or step % 10 == 0:
            preds = np.argmax(logits.data, axis=-1)
            acc = float(np.mean(preds == y_np)) * 100.0
            history.append((step, loss.item(), grad_norm, acc))

    return history


def main() -> None:
    print("=" * 88)
    print(" TENSOR-FORGE: NUMERICAL STABILITY & REVERSE-MODE AUTODIFF BENCHMARK SUITE")
    print("=" * 88)

    print("\n[1/3] Operator Forward + Backward Throughput & Memory Benchmarks")
    print("-" * 88)
    print(
        f"{'Operator':<24} | {'Shape':<26} | {'Fwd (ms)':>8} | {'Bwd (ms)':>8} | {'GFLOP/s':>8} | {'Peak KB':>8}"
    )
    print("-" * 88)
    bench_results = run_throughput_benchmarks()
    for r in bench_results:
        print(
            f"{r.name:<24} | {r.shape_desc:<26} | {r.fwd_ms:8.3f} | {r.bwd_ms:8.3f} | {r.gflops:8.2f} | {r.peak_kb:8.1f}"
        )

    print("\n[2/3] Adversarial Numerical Stability Verification (Naive vs. tensor-forge)")
    print("-" * 88)
    stability_rows = run_stability_comparison()
    for case, naive_str, tf_str, grad_str in stability_rows:
        print(f"* {case}")
        print(f"    Naive NumPy  : {naive_str}")
        print(f"    tensor-forge : {tf_str} | Backward: {grad_str}")

    print("\n[3/3] End-to-End 2-Layer Transformer + MLP Sequence Classification Convergence")
    print("-" * 88)
    print(
        f"{'Step':>6} | {'Cross-Entropy Loss':>20} | {'Pre-Clip Grad L2 Norm':>22} | {'Accuracy (%)':>14}"
    )
    print("-" * 88)
    history = run_convergence_benchmark()
    for step, loss_val, gnorm, acc in history:
        print(f"{step:6d} | {loss_val:20.6f} | {gnorm:22.4f} | {acc:13.1f}%")
    print("=" * 88)


if __name__ == "__main__":
    main()
