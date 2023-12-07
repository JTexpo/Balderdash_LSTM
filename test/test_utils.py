import os
import sys
import unittest

import numpy as np

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")

from balderdash_lstm.lstm_model.utils import (
    get_tokens,
    sigmoid,
    softmax,
    tanh_activation,
    tanh_derivative,
)


class UTILSTestCase(unittest.TestCase):
    def test_get_tokens(self):
        tokens, token_to_id, id_to_token = get_tokens()
        self.assertEqual(
            tokens,
            [
                "a",
                "b",
                "c",
                "d",
                "e",
                "f",
                "g",
                "h",
                "i",
                "j",
                "k",
                "l",
                "m",
                "n",
                "o",
                "p",
                "q",
                "r",
                "s",
                "t",
                "u",
                "v",
                "w",
                "x",
                "y",
                "z",
                ".",
            ],
        )
        self.assertEqual(token_to_id["a"], 0)
        self.assertEqual(id_to_token[0], "a")

    def test_sigmoid(self):
        self.assertEqual(sigmoid(0), 0.5)
        self.assertEqual(sigmoid(1), 0.7310585786300049)

    def test_softmax(self):
        # Test case 1: Test for positive input
        X1 = np.array([[1, 2], [3, 4]])
        expected_output1 = np.array(
            [[0.26894142, 0.73105858], [0.26894142, 0.73105858]]
        )
        assert np.allclose(softmax(X1), expected_output1)

        # Test case 2: Test for negative input
        X2 = np.array([[-1, -2], [-3, -4]])
        expected_output2 = np.array(
            [[0.73105858, 0.26894142], [0.73105858, 0.26894142]]
        )
        assert np.allclose(softmax(X2), expected_output2)

        # Test case 3: Test for zero input
        X3 = np.array([[0, 0], [0, 0]])
        expected_output4 = np.array([[0.5, 0.5], [0.5, 0.5]])
        assert np.allclose(softmax(X3), expected_output4)

    def test_tanh_activation(self):
        # Test case 1: Test for positive input
        X = np.array([1, 2, 3])
        expected_output = np.tanh(X)
        assert np.allclose(tanh_activation(X), expected_output)

        # Test case 2: Test for negative input
        X = np.array([-1, -2, -3])
        expected_output = np.tanh(X)
        assert np.allclose(tanh_activation(X), expected_output)

        # Test case 3: Test for zero input
        X = np.array([0, 0, 0])
        expected_output = np.tanh(X)
        assert np.allclose(tanh_activation(X), expected_output)

        # Test case 4: Test for large input
        X = np.array([1000, 2000, 3000])
        expected_output = np.tanh(X)
        assert np.allclose(tanh_activation(X), expected_output)

    def test_tanh_derivative(self):
        # Test case 1: Test for positive input
        X = np.array([1, 2, 3])
        expected_output = 1 - np.tanh(X) ** 2
        assert np.allclose(tanh_derivative(X), expected_output)

        # Test case 2: Test for negative input
        X = np.array([-1, -2, -3])
        expected_output = 1 - np.tanh(X) ** 2
        assert np.allclose(tanh_derivative(X), expected_output)

        # Test case 3: Test for zero input
        X = np.array([0, 0, 0])
        expected_output = 1 - np.tanh(X) ** 2
        assert np.allclose(tanh_derivative(X), expected_output)

        # Test case 4: Test for large input
        X = np.array([1000, 2000, 3000])
        expected_output = 1 - np.tanh(X) ** 2
        assert np.allclose(tanh_derivative(X), expected_output)


if __name__ == "__main__":
    unittest.main()
