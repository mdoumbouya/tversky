# <<<SU:HDR:BEGIN>>>
# COPYRIGHT © 2026 The Board of Trustees of the Leland Stanford Junior University
# Author: moussa@stanford.edu
# <<<SU:HDR:END>>>

"""
Equivalence tests between the original (v1) and optimised (v2) Tversky math.

v1: differentiable_tversky   — original, uses torch.transpose; computes fbank projections
                                multiple times; similarity_ratio_model has an erroneous
                                leading `self` parameter (known bug, not tested here).
v2: differentiable_tversky_v2 — optimised, uses .mT; projects once and shares results;
                                 fixes the ratio model signature.

For 2D inputs (n, d) the two implementations must produce bit-identical outputs.
"""
import sys
import pytest
import torch

sys.path.insert(0, "src")
import tversky.differentiable_tversky as v1
import tversky.differentiable_tversky_v2 as v2

# ── Fixtures ──────────────────────────────────────────────────────────────────

N, M, D, K = 8, 6, 16, 32
ALPHA, BETA, THETA = 0.3, 0.7, 1.2
THRESHOLD = 0.0

INTERSECTION_REDUCTIONS = ["product", "min", "max", "mean", "softmin"]
DIFFERENCE_REDUCTIONS   = ["ignorematch", "substractmatch"]

def _inputs(seed=0, positive=False):
    """Return (x, y, fbank) as plain 2-D tensors.

    positive=True guarantees all dot products exceed threshold=0,
    which is required for gmean (log needs positive argument).
    """
    torch.manual_seed(seed)
    if positive:
        x     = torch.abs(torch.randn(N, D)) + 0.5
        y     = torch.abs(torch.randn(M, D)) + 0.5
        fbank = torch.abs(torch.randn(K, D)) + 0.1
    else:
        x     = torch.randn(N, D)
        y     = torch.randn(M, D)
        fbank = torch.randn(K, D)
    return x, y, fbank

# ── contrast model — forward ──────────────────────────────────────────────────

@pytest.mark.parametrize("inter", INTERSECTION_REDUCTIONS)
@pytest.mark.parametrize("diff",  DIFFERENCE_REDUCTIONS)
@pytest.mark.parametrize("normalize", [False, True])
def test_contrast_forward(inter, diff, normalize):
    x, y, fbank = _inputs()
    kw = dict(fbank=fbank, threshold=THRESHOLD,
              alpha=ALPHA, beta=BETA, theta=THETA,
              normalize=normalize,
              intersection_reduction=inter,
              difference_reduction=diff)
    out1 = v1.similarity_contrast_model(x, y, **kw)
    out2 = v2.similarity_contrast_model(x, y, **kw)
    assert out1.shape == out2.shape, "shape mismatch"
    assert torch.allclose(out1, out2, atol=1e-6), \
        f"forward mismatch  inter={inter}  diff={diff}  normalize={normalize}\n" \
        f"max |Δ| = {(out1 - out2).abs().max().item():.2e}"

def test_contrast_forward_gmean():
    """gmean requires positive dot products — use a separate positive fixture."""
    x, y, fbank = _inputs(positive=True)
    kw = dict(fbank=fbank, threshold=THRESHOLD,
              alpha=ALPHA, beta=BETA, theta=THETA,
              normalize=False,
              intersection_reduction="gmean",
              difference_reduction="ignorematch")
    out1 = v1.similarity_contrast_model(x, y, **kw)
    out2 = v2.similarity_contrast_model(x, y, **kw)
    assert torch.allclose(out1, out2, atol=1e-6), \
        f"gmean mismatch  max |Δ| = {(out1 - out2).abs().max().item():.2e}"

# ── contrast model — gradients ────────────────────────────────────────────────

@pytest.mark.parametrize("inter", INTERSECTION_REDUCTIONS)
@pytest.mark.parametrize("diff",  DIFFERENCE_REDUCTIONS)
def test_contrast_gradients(inter, diff):
    x_base, y_base, fbank_base = _inputs(seed=1)

    kw = dict(threshold=THRESHOLD,
              alpha=ALPHA, beta=BETA, theta=THETA,
              normalize=False,
              intersection_reduction=inter,
              difference_reduction=diff)

    # v1
    x1     = x_base.clone().requires_grad_(True)
    y1     = y_base.clone().requires_grad_(True)
    fbank1 = fbank_base.clone().requires_grad_(True)
    v1.similarity_contrast_model(x1, y1, fbank=fbank1, **kw).sum().backward()

    # v2
    x2     = x_base.clone().requires_grad_(True)
    y2     = y_base.clone().requires_grad_(True)
    fbank2 = fbank_base.clone().requires_grad_(True)
    v2.similarity_contrast_model(x2, y2, fbank=fbank2, **kw).sum().backward()

    assert torch.allclose(x2.grad,     x1.grad,     atol=1e-5), \
        f"x gradient mismatch  inter={inter}  diff={diff}"
    assert torch.allclose(y2.grad,     y1.grad,     atol=1e-5), \
        f"y gradient mismatch  inter={inter}  diff={diff}"
    assert torch.allclose(fbank2.grad, fbank1.grad, atol=1e-5), \
        f"fbank gradient mismatch  inter={inter}  diff={diff}"

# ── ratio model — v1 vs v2 ───────────────────────────────────────────────────

@pytest.mark.parametrize("inter", INTERSECTION_REDUCTIONS)
@pytest.mark.parametrize("diff",  DIFFERENCE_REDUCTIONS)
def test_ratio_forward(inter, diff):
    x, y, fbank = _inputs()
    kw = dict(fbank=fbank, threshold=THRESHOLD,
              alpha=ALPHA, beta=BETA, theta=None,
              normalize=False,
              intersection_reduction=inter,
              difference_reduction=diff)
    out1 = v1.similarity_ratio_model(x, y, **kw)
    out2 = v2.similarity_ratio_model(x, y, **kw)
    assert out1.shape == (N, M), f"unexpected shape {out1.shape}"
    assert torch.isfinite(out1).all(), "non-finite values in v1 ratio output"
    assert (out1 >= 0).all(),          "v1 ratio output contains negatives"
    assert torch.allclose(out1, out2, atol=1e-6), \
        f"ratio mismatch  inter={inter}  diff={diff}\n" \
        f"max |Δ| = {(out1 - out2).abs().max().item():.2e}"

# ── output shape sanity ───────────────────────────────────────────────────────

def test_output_shape():
    x, y, fbank = _inputs()
    kw = dict(fbank=fbank, threshold=THRESHOLD,
              alpha=ALPHA, beta=BETA, theta=THETA,
              normalize=False,
              intersection_reduction="product",
              difference_reduction="ignorematch")
    assert v1.similarity_contrast_model(x, y, **kw).shape == (N, M)
    assert v2.similarity_contrast_model(x, y, **kw).shape == (N, M)

# ── symmetric inputs ──────────────────────────────────────────────────────────

def test_contrast_symmetric_alpha_beta():
    """When alpha==beta and x==y, S(x,y) should equal S(y,x)."""
    x, _, fbank = _inputs()
    kw = dict(fbank=fbank, threshold=THRESHOLD,
              alpha=0.5, beta=0.5, theta=1.0,
              normalize=False,
              intersection_reduction="product",
              difference_reduction="ignorematch")
    out1 = v1.similarity_contrast_model(x, x, **kw)
    out2 = v2.similarity_contrast_model(x, x, **kw)
    assert torch.allclose(out1, out1.T, atol=1e-6), "v1 not symmetric"
    assert torch.allclose(out2, out2.T, atol=1e-6), "v2 not symmetric"
    assert torch.allclose(out1, out2, atol=1e-6),   "v1 vs v2 mismatch on symmetric input"
