# <<<SU:HDR:BEGIN>>>
# COPYRIGHT © 2026 The Board of Trustees of the Leland Stanford Junior University
# Author: moussa@stanford.edu
# <<<SU:HDR:END>>>

import torch
import torch.nn.functional as F

def intersection_measure(dot_a_m, dot_b_m, ind_a, ind_b, reduction):
    """All inputs are precomputed projections onto the feature bank.

    dot_a_m: (..., n, k) — dot products of a against fbank, zeroed where inactive
    dot_b_m: (..., m, k) — same for b
    ind_a:   (..., n, k) bool — which features are active for a
    ind_b:   (..., m, k) bool — which features are active for b

    .mT transposes the last two dims, so this is correct for both 2D and batched 3D+ inputs.
    """
    if reduction == 'product':
        return dot_a_m @ dot_b_m.mT

    inter_a = dot_a_m @ ind_b.float().mT
    inter_b = ind_a.float() @ dot_b_m.mT

    if reduction == 'min':
        return torch.minimum(inter_a, inter_b)
    if reduction == 'max':
        return torch.maximum(inter_a, inter_b)
    if reduction == 'mean':
        return (inter_a + inter_b) / 2
    if reduction == 'gmean':
        return torch.exp((torch.log(inter_a) + torch.log(inter_b)) / 2.0)
    if reduction == 'softmin':
        return -torch.logsumexp(-torch.stack([inter_a, inter_b], dim=0), dim=0)

    known = {"product", "min", "max", "mean", "gmean", "softmin"}
    raise ValueError(f"Unknown intersection reduction: {reduction!r}. Pick one of: {known}")

def difference_measure(dot_a_m, dot_b_m, ind_a, ind_b, reduction):
    """Features active in a but not in b.

    dot_a_m: (..., n, k) — masked dot products for a
    dot_b_m: (..., m, k) — masked dot products for b
    ind_a:   (..., n, k) bool
    ind_b:   (..., m, k) bool
    """
    if reduction == 'ignorematch':
        return dot_a_m @ (~ind_b).float().mT

    if reduction == 'substractmatch':
        dot_a_ind_b = dot_a_m @ ind_b.float().mT
        dot_b_ind_a = ind_a.float() @ dot_b_m.mT
        ind_gt = dot_a_ind_b > dot_b_ind_a
        return (dot_a_m @ (~ind_b).float().mT
                + dot_a_ind_b * ind_gt
                - dot_b_ind_a * ind_gt)

    known = {"ignorematch", "substractmatch"}
    raise ValueError(f"Unknown difference reduction: {reduction!r}. Pick one of: {known}")

def _project(x, fbank, threshold):
    """Compute projections onto fbank once; return (dot, dot_masked, ind)."""
    dot = x @ fbank.T
    ind = dot > threshold
    return dot, dot * ind, ind

def similarity_contrast_model(x, y, fbank, threshold, alpha, beta, theta,
                               normalize, intersection_reduction, difference_reduction):
    if normalize:
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)

    _, dot_x_m, ind_x = _project(x, fbank, threshold)   # (n, k)
    _, dot_y_m, ind_y = _project(y, fbank, threshold)   # (m, k)

    inter  = intersection_measure(dot_x_m, dot_y_m, ind_x, ind_y, intersection_reduction)
    diff_xy = difference_measure(dot_x_m, dot_y_m, ind_x, ind_y, difference_reduction)
    diff_yx = difference_measure(dot_y_m, dot_x_m, ind_y, ind_x, difference_reduction)

    return theta * inter - alpha * diff_xy - beta * diff_yx.mT

def similarity_ratio_model(x, y, fbank, threshold, alpha, beta, theta,
                            normalize, intersection_reduction, difference_reduction):
    if normalize:
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)

    _, dot_x_m, ind_x = _project(x, fbank, threshold)
    _, dot_y_m, ind_y = _project(y, fbank, threshold)

    inter = intersection_measure(dot_x_m, dot_y_m, ind_x, ind_y, intersection_reduction)
    return torch.divide(
        inter,
        inter
        + alpha * difference_measure(dot_x_m, dot_y_m, ind_x, ind_y, difference_reduction)
        + beta  * difference_measure(dot_y_m, dot_x_m, ind_y, ind_x, difference_reduction).mT
        + 1
    )
