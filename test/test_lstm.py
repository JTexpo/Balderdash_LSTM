import os
import sys
import unittest

import numpy as np

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")

from balderdash_lstm.lstm_model.lstm import (
    LSTM,
    initialize_lstm,
    get_lstm_activations,
)


class LSTMTestCase(unittest.TestCase):

    def test_lstm(self):
        # Test case 1
        input_size = 10
        hidden_size = 20
        output_size = 5
        mean = 0.0
        standard_deviation = 1.0

        lstm = initialize_lstm(input_size, hidden_size, output_size, mean, standard_deviation)

        assert isinstance(lstm, LSTM)
        assert lstm.forget_gate_weights.shape == (input_size + hidden_size, hidden_size)
        assert lstm.input_gate_weights.shape == (input_size + hidden_size, hidden_size)
        assert lstm.output_gate_weights.shape == (input_size + hidden_size, hidden_size)
        assert lstm.gate_gate_weights.shape == (input_size + hidden_size, hidden_size)
        assert lstm.hidden_output_weights.shape == (hidden_size, output_size)

        # Test case 2
        input_size = 5
        hidden_size = 10
        output_size = 3
        mean = -1.0
        standard_deviation = 0.5

        lstm = initialize_lstm(input_size, hidden_size, output_size, mean, standard_deviation)

        assert isinstance(lstm, LSTM)
        assert lstm.forget_gate_weights.shape == (input_size + hidden_size, hidden_size)
        assert lstm.input_gate_weights.shape == (input_size + hidden_size, hidden_size)
        assert lstm.output_gate_weights.shape == (input_size + hidden_size, hidden_size)
        assert lstm.gate_gate_weights.shape == (input_size + hidden_size, hidden_size)
        assert lstm.hidden_output_weights.shape == (hidden_size, output_size)

    def test_get_lstm_activations(self):
        # Test case 1: Default inputs
        batch_dataset = np.array([[1, 2], [3, 4]])
        prev_activation_matrix = np.array([[5, 6], [7, 8]])
        prev_cell_matrix = np.array([[9, 10], [11, 12]])

        input_size = 2
        hidden_size = 2
        output_size = 2
        mean = 0.0
        standard_deviation = 1.0

        lstm = initialize_lstm(input_size, hidden_size, output_size, mean, standard_deviation)

        lstm_activations, cell_memory_matrix, activation_matrix = get_lstm_activations(
            batch_dataset, prev_activation_matrix, prev_cell_matrix, lstm
        )

        assert lstm_activations.forget_gate_activation.shape == (2, 2)
        assert lstm_activations.input_gate_activation.shape == (2, 2)
        assert lstm_activations.output_gate_activation.shape == (2, 2)
        assert lstm_activations.gate_gate_activation.shape == (2, 2)
        assert cell_memory_matrix.shape == (2, 2)
        assert activation_matrix.shape == (2, 2)

        # Test case 2: Zero inputs
        batch_dataset = np.zeros((2, 2))
        prev_activation_matrix = np.zeros((2, 2))
        prev_cell_matrix = np.zeros((2, 2))
        
        lstm = initialize_lstm(input_size, hidden_size, output_size, mean, standard_deviation)

        lstm_activations, cell_memory_matrix, activation_matrix = get_lstm_activations(
            batch_dataset, prev_activation_matrix, prev_cell_matrix, lstm
        )

        assert lstm_activations.forget_gate_activation.shape == (2, 2)
        assert lstm_activations.input_gate_activation.shape == (2, 2)
        assert lstm_activations.output_gate_activation.shape == (2, 2)
        assert lstm_activations.gate_gate_activation.shape == (2, 2)
        assert cell_memory_matrix.shape == (2, 2)
        assert activation_matrix.shape == (2, 2)

        # Test case 3: Random inputs
        batch_dataset = np.random.rand(2, 2)
        prev_activation_matrix = np.random.rand(2, 2)
        prev_cell_matrix = np.random.rand(2, 2)
        
        lstm = initialize_lstm(input_size, hidden_size, output_size, mean, standard_deviation)

        lstm_activations, cell_memory_matrix, activation_matrix = get_lstm_activations(
            batch_dataset, prev_activation_matrix, prev_cell_matrix, lstm
        )

        assert lstm_activations.forget_gate_activation.shape == (2, 2)
        assert lstm_activations.input_gate_activation.shape == (2, 2)
        assert lstm_activations.output_gate_activation.shape == (2, 2)
        assert lstm_activations.gate_gate_activation.shape == (2, 2)
        assert cell_memory_matrix.shape == (2, 2)
        assert activation_matrix.shape == (2, 2)
        
if __name__ == "__main__":
    unittest.main()
