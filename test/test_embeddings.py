import os
import sys
import unittest

import numpy as np

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")

from balderdash_lstm.lstm_model.embeddings import initialize_embeddings, get_batch_embeddings


class EmbeddingsTestCase(unittest.TestCase):

    def test_initialize_embeddings(self):
        mean = 0
        standard_deviation = 0.01
        tokens = ["a", "b", "c"]
        input_size = 3
        embeddings = initialize_embeddings(mean, standard_deviation, tokens, input_size)
        self.assertEqual(embeddings.shape, (3, 3))

    def test_get_batch_embeddings(self):
        mean = 0
        standard_deviation = 0.01
        tokens = ["a", "b", "c"]
        input_size = 3
        embeddings = initialize_embeddings(mean, standard_deviation, tokens, input_size)
        batch_dataset = np.array([[1, 2, 3], [4, 5, 6]])
        batch_embeddings = get_batch_embeddings(batch_dataset, embeddings)
        self.assertEqual(batch_embeddings.shape, (2, 3))


if __name__ == "__main__":
    unittest.main()
