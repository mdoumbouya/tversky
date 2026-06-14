# <<<SU:HDR:BEGIN>>>
# COPYRIGHT © 2026 The Board of Trustees of the Leland Stanford Junior University
# Author: moussa@stanford.edu
# <<<SU:HDR:END>>>

def test_import():
    import tversky
    assert tversky is not None

    from tversky import nn as tnn
    assert tnn is not None

def test_similarity():
    import torch
    from tversky import nn as tnn
    sim_layer = tnn.TverskySimilarity(
        embedding_dim=64,
        fbank_size=128,
        similarity_model='contrast',
        normalize=False
    )
    x = torch.randn(4, 64)
    y = torch.randn(6, 64)
    out = sim_layer(x, y)
    assert out.shape == (4, 6)

def test_projection():
    from tversky import nn as tnn
    proj_layer = tnn.TverskyProjection(
        embedding_dim=64,
        class_count=10,
        fbank_size=128,
        similarity_model='contrast',
        normalize=False
    )
