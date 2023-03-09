# Add path for CI/CD tool
from locale import normalize
import sys
import os

sys.path.append(
    os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + "/../myself/")
)
import numpy as np
import unittest
import rnn1


class TestTimeRNN(unittest.TestCase):
    def setUp(self):
        # Set up some example parameters and input data for testing
        self.Wx = np.random.randn(10, 20)
        self.Wh = np.random.randn(20, 20)
        self.b = np.random.randn(20)
        self.xs = np.random.randn(5, 8, 10)
        self.dhs = np.random.randn(5, 8, 20)

    def test_forward(self):
        # Test the forward method of TimeRNN
        rnn = TimeRNN(self.Wx, self.Wh, self.b)
        hs = rnn.forward(self.xs)
        self.assertEqual(
            hs.shape, (5, 8, 20)
        )  # Output shape should be (batch_size, sequence_length, hidden_size)

    def test_backward(self):
        # Test the backward method of TimeRNN
        rnn = TimeRNN(self.Wx, self.Wh, self.b)
        hs = rnn.forward(self.xs)
        dxs = rnn.backward(self.dhs)
        self.assertEqual(
            dxs.shape, (5, 8, 10)
        )  # Input gradient shape should be the same as input shape
        self.assertEqual(
            len(rnn.grads), 3
        )  # There should be gradients for Wx, Wh, and b

    def test_stateful(self):
        # Test the stateful option of TimeRNN
        rnn = TimeRNN(self.Wx, self.Wh, self.b, stateful=True)
        h1 = rnn.forward(self.xs[:, :4, :])
        h2 = rnn.forward(self.xs[:, 4:, :])
        np.testing.assert_allclose(
            h1[:, -1, :], rnn.h
        )  # The final hidden state of the first sequence should be the same as the hidden state of the entire sequence
        np.testing.assert_allclose(
            h2[:, 0, :], rnn.h
        )  # The initial hidden state of the second sequence should be the same as the final hidden state of the first sequence


if __name__ == "__main__":
    unittest.main()
