"""Gradient check for CifgLSTM.backward (lstm.py).

In the CIFG variant the forget gate is coupled to the input gate
(f = 1 - i), so the input gate changes the cell state two ways:
dc/di = z - c_prev, not just z. This test proves the backward pass
is correct by comparing its analytic gradients against central
finite differences of the same loss, for all 11 parameters.

Loss: L = sum_t dot(y[t], w[t]) with a fixed probe matrix w, so the
upstream deltas handed to backward() are exactly w.

Run from the repo root:  python -m unittest test_lstm_gradient -v
"""
import unittest

import numpy as np

from lstm import CifgLSTM

PARAM_NAMES = ["Wz", "Wi", "Wo", "Rz", "Ri", "Ro", "pi", "po", "bz", "bi", "bo"]


class TestLSTMGradient(unittest.TestCase):
    def test_backward_matches_finite_differences(self):
        rng = np.random.RandomState(42)
        M, N, T = 3, 2, 4
        layer = CifgLSTM(M, N)
        x = rng.randn(T, M) * 0.5
        w = rng.randn(T, N)  # probe: L = sum(y * w)

        y, fwd_state = layer.forward(x)
        self.assertTrue(np.isfinite(np.sum(y * w)))
        _, grads, _ = layer.backward(w, fwd_state)

        eps = 1e-6
        params = layer.get_params()
        self.assertEqual(len(params), len(PARAM_NAMES))
        for name, param, grad in zip(PARAM_NAMES, params, grads):
            num = np.zeros_like(param, dtype=float)
            it = np.nditer(param, flags=["multi_index"], op_flags=["readwrite"])
            while not it.finished:
                idx = it.multi_index
                old = param[idx]
                param[idx] = old + eps
                lp, _ = layer.forward(x)
                param[idx] = old - eps
                lm, _ = layer.forward(x)
                param[idx] = old
                num[idx] = (float(np.sum(lp * w)) - float(np.sum(lm * w))) / (2 * eps)
                it.iternext()
            denom = np.maximum(np.abs(num), np.abs(grad))
            rel = np.max(np.abs(num - grad) / np.maximum(denom, 1e-8))
            with self.subTest(param=name):
                self.assertLess(rel, 1e-5, f"{name}: max rel error {rel:.2e}")


if __name__ == "__main__":
    unittest.main()
