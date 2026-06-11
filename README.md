# Official pytorch library for Tversky Neural Networks
<img src="img/poster-thumbnail.jpeg" />

## Links
- [pytorch library](https://github.com/mdoumbouya/tversky) (this repository)
- [ICLR 2026 Experiments](https://github.com/mdoumbouya/tversky-networks-iclr2026) 
- [ICLR 2026 Paper](https://openreview.net/pdf?id=koKWoKaMrE)
- [ICLR 2026 Page](https://iclr.cc/virtual/2026/poster/10007748) 
- [ICLR 2026 High Resolution Poster](https://iclr.cc/media/PosterPDFs/ICLR%202026/10007748.png)
- [web site](https://mdoumbouya.github.io/article_0007_tversky_neural_networks.html)


 [![Tests](https://github.com/mdoumbouya/tversky/actions/workflows/tests.yml/badge.svg)](https://github.com/mdoumbouya/tversky/actions/workflows/tests.yml) [![Publish to PyPI](https://github.com/mdoumbouya/tversky/actions/workflows/pypi.yml/badge.svg)](https://github.com/mdoumbouya/tversky/actions/workflows/pypi.yml) ![PyPI - Version](https://img.shields.io/pypi/v/tversky)


## Installation
Note: tversky requires PyTorch ≥ 2.0. Install it first following the instructions at https://pytorch.org/get-started. Then run:
```
pip install tversky
```

## Notes
- The code used to reproduce the experiments presented in our [ICLR 2026 paper](https://openreview.net/pdf?id=koKWoKaMrE) is located in the [tversky-networks-iclr2026](https://github.com/mdoumbouya/tversky-networks-iclr2026) repository. This library was forked that repository. That repository does not use this library.

## Component: Tversky Similarity Layer
```python
from tversky import nn as tnn

sim_layer = tnn.TverskySimilarity(
    embedding_dim=64,
    fbank_size=128,
    similarity_model='contrast',
    normalize=False
)
```

## Component: Tversky Projection Layer

```python
from tversky import nn as tnn

proj_layer = tnn.TverskyProjection(
    embedding_dim=64,
    class_count=10,
    fbank_size=128,
    similarity_model='contrast',
    normalize=False
)
```


## MNIST Example
```bash
make test-mnist
```

Results (plots, training curves, salience analysis): [tests/outputs/mnist/report.md](tests/outputs/mnist/report.md)

## License
[LICENSE.txt](https://github.com/mdoumbouya/tversky/blob/main/LICENSE.txt)


## Citation
If you use this work, please cite the following paper:

```
@inproceedings{doumbouya2026tversky,
    title={Tversky Neural Networks: Psychologically Plausible Deep Learning with Differentiable Tversky Similarity},
    author={Moussa Koulako Bala Doumbouya and Dan Jurafsky and Christopher D Manning},
    booktitle={The Fourteenth International Conference on Learning Representations},
    year={2026},
    url={https://openreview.net/forum?id=koKWoKaMrE}
}
```
