# LSTM-Numpy
## Description
The implementation of the LSTM network from scratch using Numpy in Python, and the training of a simple word level language model.

This code is an implementation of the LSTM variant called CIFG (Coupled Input and Forget Gates) from the paper `LSTM : A Search Space Odyssey` [link](https://arxiv.org/pdf/1503.04069.pdf)

The model is a two stacked layers of LSTMs followed by a linear classification layer for decoding the next character. It is trained on a shakespeare corpus.

The code also contains a simple implementation of the SGD with momentum algorithm for the network parameters optimization.
## My Changes in This Fork

This is a fork of [younesslanda/LSTM-Numpy](https://github.com/younesslanda/LSTM-Numpy) by **Youness Landa** (MIT License, preserved in this repo). The original code is his; my contribution is below.

**Fixed a gradient bug in the input gate** (`lstm.py`, `CifgLSTM.backward`): this LSTM couples its gates — the forget gate is `f = 1 - i` — so nudging the input gate changes the cell state two ways: more new content in, and less old content kept. The original code only counted the first effect (`dc/di = z`). The fix counts both (`dc/di = z - c_prev`, with the previous cell state taken as 0 at the first timestep):

```python
c_prev = c[t - 1] if t > 0 else 0  # cell state before t (0 at t=0)
di[t] = dc[t] * (z[t] - c_prev) * dsigma(i_[t])  # CIFG: f = 1 - i, so dc/di = z - c_prev
```

**Added the repo's first test** (`test_lstm_gradient.py`): a finite-difference gradient check that compares the backward pass against numerical gradients for all 11 parameters. Before the fix all 11 parameters fail the check; after the fix all 11 pass (max relative error under 6e-08). Run it with `python -m unittest test_lstm_gradient -v`.

Honest note: the bug never crashed anything and the network still trained — the gradient check is the proof, not the training curve. Also, despite the description above, the language model is character-level, not word-level.
