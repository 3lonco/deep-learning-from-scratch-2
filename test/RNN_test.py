import unittest
import numpy as np


class TestRNN(unittest.TestCase):
    def setUp(self):
        # Create an RNN instance
        Wx = np.random.randn(3, 4)
        Wh = np.random.randn(4, 4)
        b = np.random.randn(4)
        self.rnn = RNN(Wx, Wh, b)

    def test_forward(self):
        x = np.random.randn(2, 3)
        h_prev = np.random.randn(2, 4)
        h_next = self.rnn.forward(x, h_prev)
        self.assertEqual(h_next.shape, (2, 4))

    def test_backward(self):
        x = np.random.randn(2, 3)
        h_prev = np.random.randn(2, 4)
        h_next = self.rnn.forward(x, h_prev)
        dh_next = np.random.randn(*h_next.shape)
        dx, dh_prev = self.rnn.backward(dh_next)
        self.assertEqual(dx.shape, x.shape)
        self.assertEqual(dh_prev.shape, h_prev.shape)

    def test_gradients(self):
        x = np.random.randn(2, 3)
        h_prev = np.random.randn(2, 4)
        h_next = self.rnn.forward(x, h_prev)
        dh_next = np.random.randn(*h_next.shape)
        dx, dh_prev = self.rnn.backward(dh_next)
        epsilon = 1e-6
        for param, grad in zip(self.rnn.params, self.rnn.grad):
            self.assertEqual(param.shape, grad.shape)
            for i in range(param.size):
                # Compute the numerical gradient
                param_flat = param.flat[i]
                param.flat[i] = param_flat + epsilon
                h_next1 = self.rnn.forward(x, h_prev)
                loss1 = np.sum(h_next1 * dh_next)
                param.flat[i] = param_flat - epsilon
                h_next2 = self.rnn.forward(x, h_prev)
                loss2 = np.sum(h_next2 * dh_next)
                param.flat[i] = param_flat
                numerical_grad = (loss1 - loss2) / (2 * epsilon)

                # Check that the numerical gradient is close to the computed gradient
                self.assertLess(np.abs(numerical_grad - grad.flat[i]), epsilon)


if __name__ == "__main__":
    unittest.main()
