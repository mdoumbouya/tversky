# <<<SU:HDR:BEGIN>>>
# COPYRIGHT © 2026 The Board of Trustees of the Leland Stanford Junior University
# Author: moussa@stanford.edu
# <<<SU:HDR:END>>>

import torch
import torch.nn.functional as F

def intersection_measure(a, b, fbank, threshold, reduction):
    known_reductions = {"product", "min", "max", "mean", "gmean", "softmin"}
    if reduction not in known_reductions:
        raise ValueError(f"Unknown reduction for tversky intersection measure:{reduction}. Pick one of: {known_reductions}")

    dot_a = a @ torch.transpose(fbank, 0, 1)
    # (batch_size_a, d) (bank_size, d).T =  (batch_size_a, bank_size)
    ind_a_features = (dot_a > threshold).float()
    dot_a = torch.multiply(dot_a, ind_a_features)

    dot_b = b @ torch.transpose(fbank, 0, 1)
    # (batch_size_b, d) (bank_size, d).T =  (batch_size_b, bank_size)
    ind_b_features = (dot_b > threshold).float()
    dot_b = torch.multiply(dot_b, ind_b_features)

    if reduction == 'product':
        return dot_a @ torch.transpose(dot_b, 0, 1)
    else:
        intersection_measure_a = dot_a @ torch.transpose(ind_b_features, 0, 1)
        # (batch_size_a, bank_size) (batch_size_b, bank_size).T =  (batch_size_a, batch_size_b)

        intersection_measure_b = ind_a_features @ torch.transpose(dot_b, 0, 1)
        # (batch_size_a, bank_size) (batch_size_b, bank_size).T = (batch_size_a, batch_size_b)
        if reduction == 'min':
            return torch.minimum(intersection_measure_a, intersection_measure_b)
        if reduction == 'max':
            return torch.maximum(intersection_measure_a, intersection_measure_b)
        if reduction == 'mean':
            return (intersection_measure_a + intersection_measure_b) / 2
        if reduction == 'gmean':
            return torch.exp(
                (torch.log(intersection_measure_a) + torch.log(intersection_measure_b))/2.0
            )
        if reduction == 'softmin':
            stacked_intersection_measures = torch.stack([intersection_measure_a, intersection_measure_b], dim=0)
            return -torch.logsumexp(-stacked_intersection_measures, dim=0)

def difference_measure(a, b, fbank, threshold, reduction):
    known_reductions = {"ignorematch", "substractmatch"}
    if reduction not in known_reductions:
        raise ValueError(f"Unknown reduction for tversky difference measure. Pick one of: {known_reductions}")

    if reduction == "ignorematch":
        dot_a = a @ torch.transpose(fbank, 0, 1)
        ind_a_features = dot_a > threshold
        dot_a = torch.multiply(dot_a, ind_a_features)

        dot_b = b @ torch.transpose(fbank, 0, 1)
        # (batch_size_b, d) (bank_size, d).T =  (batch_size_b, bank_size)
        ind_notb_features = dot_b <= threshold

        return dot_a @ torch.transpose(ind_notb_features, 0, 1).float()
        # (batch_size_a, bank_size) (batch_size_b, bank_size).T = (batch_size_a, batch_size_b)
    elif reduction == "substractmatch":
        dot_a = a @ torch.transpose(fbank, 0, 1)
        # (batch_size_a, d) (bank_size, d).T =  (batch_size_a, bank_size)
        ind_a_features = (dot_a > threshold)
        dot_a = torch.multiply(dot_a, ind_a_features)

        dot_b = b @ torch.transpose(fbank, 0, 1)
        # (batch_size_b, d) (bank_size, d).T =  (batch_size_b, bank_size)
        ind_b_features = (dot_b > threshold)
        dot_b = torch.multiply(dot_b, ind_b_features)

        dot_a_ind_b = dot_a @ torch.transpose(ind_b_features, 0, 1).float()
        # (batch_size_a, bank_size) (batch_size_b, bank_size).T = (batch_size_a, batch_size_b)

        dot_b_ind_a = (ind_a_features.float() @ torch.transpose(dot_b, 0, 1))
        # (batch_size_a, bank_size) (batch_size_b, bank_size).T = (batch_size_a, batch_size_b)

        ind_gt_b = dot_a_ind_b>dot_b_ind_a

        return dot_a @ torch.transpose(~ind_b_features, 0, 1).float() \
            + torch.multiply(dot_a_ind_b, ind_gt_b) \
            - torch.multiply(dot_b_ind_a, ind_gt_b)

def similarity_contrast_model(x, y, fbank, threshold, alpha, beta, theta, normalize, intersection_reduction, difference_reduction):
    """Evaluates the degree to which each element in x is similar to each element in y

    Args:
        x (torch.Tensor): (n, d) matrix (n d-simensional vectors)
        y (torch.Tensor): (m, d) matrix (m d-simensional vectors)
        fbank (torch.Tensor): (k, d) matrix (k d-dimentional vectors)
        threshold (float): conventional element-feature dot product threshold for set membership
        alpha (float): Tversky's contrast model alpha parameter
        beta (float): Tversky's contrast model beta parameter
        theta (float): Tversky's contrast model theta parameter
        normalize (bool): whether to normalize element vectors

    Returns:
        torch.Tensor: matrix of size (n, m) where entry (i, j) is the measure of similarity of x[i] to y[j]
    """
    if normalize:
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)

    similarity_matrix =  theta     *   intersection_measure(x, y, fbank, threshold, intersection_reduction) \
                        - alpha    *   difference_measure(x, y, fbank, threshold, difference_reduction) \
                        - beta     *   torch.transpose(difference_measure(y, x, fbank, threshold, difference_reduction), 0, 1) # noqa:  E221,E222

    return similarity_matrix

def similarity_ratio_model(x, y, fbank, threshold, alpha, beta, theta,  normalize, intersection_reduction, difference_reduction):
    if normalize:
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)

    inter_m = intersection_measure(x, y, fbank, threshold, intersection_reduction)
    similarity_matrix = torch.divide(
        inter_m,
        inter_m + alpha    * difference_measure(x, y, fbank, threshold, difference_reduction)  # noqa:  E221,E222
                + beta     * torch.transpose(difference_measure(y, x, fbank, threshold, difference_reduction), 0, 1)  # noqa:  E221,E222
                + 1
    )
    return similarity_matrix
