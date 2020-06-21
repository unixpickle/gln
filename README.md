# gln

A PyTorch implementation of Gated Linear Networks from [Veness et al](https://arxiv.org/abs/1910.01526).

# What's included

 * [gln/model.py](gln/model.py) - The core GLN implementation. The `Layer` class implements the gating mechanism and the local gradient computation.
 * [gln/one_vs_all.py](gln/one_vs_all.py) - A helper for non-binary classification.
 * [train_mnist.py](train_mnist.py) - A script that trains an MNIST classifier using one pass over the training data. Since it trains with batch size 1, it takes a while to train, but very little time to evaluate test performance.
