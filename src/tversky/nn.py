# <<<SU:HDR:BEGIN>>>
# COPYRIGHT © 2026 The Board of Trustees of the Leland Stanford Junior University
# Author: moussa@stanford.edu
# <<<SU:HDR:END>>>

import torch
from torch import nn
import torch.nn.functional as F
from .differentiable_tversky_v2 import similarity_ratio_model, similarity_contrast_model

class TverskyProjection(nn.Module):
    def __init__(self, embedding_dim, class_count, fbank_size, similarity_model, normalize, intersection_reduction="product", difference_reduction="ignorematch"):
        super().__init__()
        self.feature_bank = nn.Embedding(num_embeddings=fbank_size, embedding_dim=embedding_dim)
        self.prototypes = nn.Embedding(num_embeddings=class_count, embedding_dim=embedding_dim)
        torch.nn.init.uniform_(self.feature_bank.weight)
        torch.nn.init.uniform_(self.prototypes.weight)
        self.theta = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.beta = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.threshold = 0.0
        self.similarity_model = similarity_model
        self.normalize = normalize
        self.intersection_reduction = intersection_reduction
        self.difference_reduction = difference_reduction

    def similarity(self, x, y, alpha=None, beta=None, theta=None, fbank=None):
        alpha = self.alpha if alpha is None else alpha
        beta = self.beta if beta is None else beta
        theta = self.theta if theta is None else theta
        fbank = self.feature_bank.weight if fbank is None else fbank

        if self.similarity_model == 'contrast':
            return similarity_contrast_model(
                x, y,
                fbank=fbank,
                threshold=self.threshold,
                alpha=alpha,
                beta=beta,
                theta=theta,
                normalize=self.normalize,
                intersection_reduction=self.intersection_reduction,
                difference_reduction=self.difference_reduction
            )
        elif self.similarity_model == 'ratio':
            return similarity_ratio_model(
                x, y,
                fbank=fbank,
                threshold=self.threshold,
                alpha=alpha,
                beta=beta,
                theta=theta,
                normalize=self.normalize,
                intersection_reduction=self.intersection_reduction,
                difference_reduction=self.difference_reduction
            )
        else:
            raise ValueError(
                f"Unknown similarity_model: {self.similarity_model}")

    def forward(self, x, prototypes=None, fbank=None):
        fbank = self.feature_bank.weight if fbank is None else fbank
        prototypes = self.prototypes.weight if prototypes is None else prototypes

        original_shape = None
        if len(x.shape) == 3:
            original_shape = x.shape
            x = x.reshape(x.shape[0]*x.shape[1], x.shape[2])
        x = self.similarity(x, prototypes, fbank=fbank)
        if original_shape:
            x = x.reshape(original_shape[0], original_shape[1], -1)
        return x

    def get_diagnostics(self):
        diag_dict = {}

        for ix in range(self.feature_bank.weight.grad.shape[0]):
            diag_dict[f"diagnostics/gnorms/feature_{ix:03}"] = \
                self.feature_bank.weight.grad[ix].detach().norm()
        
        for ix in range(self.prototypes.weight.grad.shape[0]):
            diag_dict[f"diagnostics/gnorms/prototype_{ix:03}"] = \
                self.prototypes.weight.grad[ix].detach().norm()

        diag_dict["diagnostics/theta"] = self.theta.item()
        diag_dict["diagnostics/alpha"] = self.alpha.item()
        diag_dict["diagnostics/beta"] = self.beta.item()

        return diag_dict

class TverskySimilarity(nn.Module):
    def __init__(self, embedding_dim, fbank_size, similarity_model, normalize, intersection_reduction="product", difference_reduction="ignorematch"):
        super().__init__()
        self.feature_bank = nn.Embedding(num_embeddings=fbank_size, embedding_dim=embedding_dim)
        # self.prototypes = nn.Embedding(num_embeddings=class_count, embedding_dim=embedding_dim)
        torch.nn.init.uniform_(self.feature_bank.weight)
        self.theta = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.beta = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.threshold = 0.0
        self.similarity_model = similarity_model
        self.normalize = normalize
        self.intersection_reduction = intersection_reduction
        self.difference_reduction = difference_reduction

    def forward(self, x, y, alpha=None, beta=None, theta=None):
        alpha = self.alpha if alpha is None else alpha
        beta = self.beta if beta is None else beta
        theta = self.theta if theta is None else theta

        if self.similarity_model == 'contrast':
            return similarity_contrast_model(
                x, y,
                fbank=self.feature_bank.weight,
                threshold=self.threshold,
                alpha=alpha,
                beta=beta,
                theta=theta,
                normalize=self.normalize,
                intersection_reduction=self.intersection_reduction,
                difference_reduction=self.difference_reduction
            )
        elif self.similarity_model == 'ratio':
            return similarity_ratio_model(
                x, y,
                fbank=self.feature_bank.weight,
                threshold=self.threshold,
                alpha=alpha,
                beta=beta,
                theta=theta,
                normalize=self.normalize,
                intersection_reduction=self.intersection_reduction,
                difference_reduction=self.difference_reduction
            )
        else:
            raise ValueError(
                f"Unknown similarity_model: {self.similarity_model}")

    def get_diagnostics(self):
        diag_dict = {}

        for ix in range(self.feature_bank.weight.grad.shape[0]):
            diag_dict[f"diagnostics/gnorms/feature_{ix:03}"] = \
                self.feature_bank.weight.grad[ix].detach().norm()

        diag_dict["diagnostics/theta"] = self.theta.item()
        diag_dict["diagnostics/alpha"] = self.alpha.item()
        diag_dict["diagnostics/beta"] = self.beta.item()

        return diag_dict
