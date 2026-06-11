"""Speed benchmark + cProfile for Tversky vs LinearBaseline.

Run with:  python scripts/profile_speed.py
"""
import cProfile
import io
import pstats
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_moons

sys.path.insert(0, "src")
from tversky import nn as tnn

# ── Device ────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f"Device: {DEVICE}\n")

# ── Data ──────────────────────────────────────────────────────────────────────
N = 500
X_np, y_np = make_moons(n_samples=N, noise=0.10, random_state=0)
X_np = X_np.astype(np.float32)
y_np = y_np.astype(np.int64)
lo, hi = X_np.min(0), X_np.max(0)
X_np = (X_np - lo) / np.maximum(hi - lo, 1e-8)
X = torch.tensor(X_np, device=DEVICE)
y = torch.tensor(y_np, device=DEVICE)

LR = 0.008

# ── Models ────────────────────────────────────────────────────────────────────

def _make_block(emb_dim, class_count, fbank_size):
    p = tnn.TverskyProjection(
        embedding_dim=emb_dim, class_count=class_count, fbank_size=fbank_size,
        similarity_model="contrast", normalize=False,
        intersection_reduction="product", difference_reduction="substractmatch",
    )
    nn.init.uniform_(p.feature_bank.weight)
    nn.init.uniform_(p.prototypes.weight)
    return p


class OneLayerTversky(nn.Module):
    def __init__(self, fbank_size=64):
        super().__init__()
        self.l1 = _make_block(2, 2, fbank_size)
    def forward(self, x): return self.l1(x)


class TwoLayerTversky(nn.Module):
    def __init__(self, hidden=8, fbank_size=64):
        super().__init__()
        self.l1 = _make_block(2, hidden, fbank_size)
        self.l2 = _make_block(hidden, 2, fbank_size)
    def forward(self, x): return self.l2(self.l1(x))


class LinearBaseline(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, hidden), nn.ReLU(), nn.Linear(hidden, 2))
    def forward(self, x): return self.net(x)


def _compiled(make):
    def factory():
        return torch.compile(make())
    return factory

MODELS = {
    "Linear (h=64)":          lambda: LinearBaseline(64),
    "Linear (h=64) compiled": _compiled(lambda: LinearBaseline(64)),
    "1L   (f=64)":            lambda: OneLayerTversky(64),
    "1L   (f=64)  compiled":  _compiled(lambda: OneLayerTversky(64)),
    "1L   (f=256)":           lambda: OneLayerTversky(256),
    "1L   (f=256) compiled":  _compiled(lambda: OneLayerTversky(256)),
    "2L   (h=8,f=64)":        lambda: TwoLayerTversky(8, 64),
    "2L   (h=8,f=64) comp":   _compiled(lambda: TwoLayerTversky(8, 64)),
}


# ── Training loop (no recording overhead) ─────────────────────────────────────

def train_loop(model, n_iter):
    model = model.to(DEVICE)
    opt  = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()
    for _ in range(n_iter):
        loss = crit(model(X), y)
        opt.zero_grad(); loss.backward(); opt.step()


# ── 1. Timing vs n_iter ───────────────────────────────────────────────────────

ITER_COUNTS = [100, 500, 1000, 5000, 10_000, 50_000]
# ITER_COUNTS = [100, 500]

print("=" * 72)
print("Timing (seconds) vs n_iter")
print("=" * 72)
header = f"{'Model':<22}" + "".join(f"{n:>10}" for n in ITER_COUNTS)
print(header)
print("-" * len(header))

for label, make in MODELS.items():
    times = []
    for n_iter in ITER_COUNTS:
        torch.manual_seed(0)
        m = make()
        # warmup
        train_loop(m, min(50, n_iter))
        torch.manual_seed(0)
        m = make()
        t0 = time.perf_counter()
        train_loop(m, n_iter)
        times.append(time.perf_counter() - t0)
    row = f"{label:<22}" + "".join(f"{t:>10.2f}" for t in times)
    print(row)

print()


# ── 2. Per-iteration cost (slope) ─────────────────────────────────────────────

ITER_A, ITER_B = 1_000, 10_000

print("=" * 72)
print(f"Per-iteration cost  (measured as (t[{ITER_B}] - t[{ITER_A}]) / {ITER_B - ITER_A})")
print("=" * 72)

for label, make in MODELS.items():
    results = {}
    for n_iter in (ITER_A, ITER_B):
        torch.manual_seed(0)
        m = make()
        train_loop(m, 50)                     # warmup
        torch.manual_seed(0)
        m = make()
        t0 = time.perf_counter()
        train_loop(m, n_iter)
        results[n_iter] = time.perf_counter() - t0
    per_iter_ms = (results[ITER_B] - results[ITER_A]) / (ITER_B - ITER_A) * 1000
    n_params = sum(p.numel() for p in make().parameters())
    print(f"  {label:<22}  {per_iter_ms:.4f} ms/iter   ({n_params} params)")

print()


# ── 3. cProfile: top hot spots per model ──────────────────────────────────────

PROFILE_ITERS = 2_000

print("=" * 72)
print(f"cProfile  (n_iter={PROFILE_ITERS}, top 15 cumulative, eager only)")
print("=" * 72)

for label, make in MODELS.items():
    if "compiled" in label or "comp" in label:
        continue
    torch.manual_seed(0)
    m = make()
    train_loop(m, 50)                         # warmup outside profiler

    torch.manual_seed(0)
    m = make()
    pr = cProfile.Profile()
    pr.enable()
    train_loop(m, PROFILE_ITERS)
    pr.disable()

    buf = io.StringIO()
    ps  = pstats.Stats(pr, stream=buf).sort_stats("cumulative")
    ps.print_stats(15)
    raw = buf.getvalue()

    # strip the preamble lines before the table
    lines = raw.splitlines()
    start = next((i for i, l in enumerate(lines) if "ncalls" in l), 0)
    table = "\n".join(lines[start:start + 17])

    print(f"\n── {label} ──")
    print(table)
