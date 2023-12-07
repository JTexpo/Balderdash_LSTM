from typing import Tuple, List

import numpy as np

from lstm_model.utils import sigmoid, softmax, tanh_activation
from lstm_model.embeddings import get_batch_embeddings

class LSTMDerivative:
    def __init__(
        self,
        forget_gate_derivative: np.array = None,
        input_gate_derivative: np.array = None,
        output_gate_derivative: np.array = None,
        gate_gate_derivative: np.array = None,
        hidden_output_derivative: np.array = None,
    ):
        self.forget_gate_derivative: np.array = forget_gate_derivative
        self.input_gate_derivative: np.array = input_gate_derivative
        self.output_gate_derivative: np.array = output_gate_derivative
        self.gate_gate_derivative: np.array = gate_gate_derivative
        self.hidden_output_derivative: np.array = hidden_output_derivative

class LSTMErrors:
    def __init__(
        self,
        forget_gate_error: np.array,
        input_gate_error: np.array,
        output_gate_error: np.array,
        gate_gate_error: np.array,
    ):
        self.forget_gate_error: np.array = forget_gate_error
        self.input_gate_error: np.array = input_gate_error
        self.output_gate_error: np.array = output_gate_error
        self.gate_gate_error: np.array = gate_gate_error

class LSTMActivations:
    def __init__(
        self,
        forget_gate_activation: np.array,
        input_gate_activation: np.array,
        output_gate_activation: np.array,
        gate_gate_activation: np.array,
    ):
        self.forget_gate_activation: np.array = forget_gate_activation
        self.input_gate_activation: np.array = input_gate_activation
        self.output_gate_activation: np.array = output_gate_activation
        self.gate_gate_activation: np.array = gate_gate_activation

class LSTM:
    def __init__(
        self,
        forget_gate_weights: np.array,
        input_gate_weights: np.array,
        output_gate_weights: np.array,
        gate_gate_weights: np.array,
        hidden_output_weights: np.array,
    ):
        self.forget_gate_weights: np.array = forget_gate_weights
        self.input_gate_weights: np.array = input_gate_weights
        self.output_gate_weights: np.array = output_gate_weights
        self.gate_gate_weights: np.array = gate_gate_weights
        self.hidden_output_weights: np.array = hidden_output_weights

def initialize_lstm(
    input_size: int,
    hidden_size: int,
    output_size: int,
    mean: float,
    standard_deviation: float,
) -> LSTM:
    """
    Initializes an LSTM model with random weights.

    Args:
        input_size (int): The size of the input embeddings.
        hidden_size (int): The size of the LSTM hidden state.
        output_size (int): The size of the output embeddings.
        mean (float): The mean value for weight initialization.
        standard_deviation (float): The standard deviation for weight initialization.

    Returns:
        LSTM: A LSTM model with random weights.
    """

    # Initialize the weights
    forget_gate_weights = np.random.normal(
        mean, standard_deviation, (input_size + hidden_size, hidden_size)
    )
    input_gate_weights = np.random.normal(
        mean, standard_deviation, (input_size + hidden_size, hidden_size)
    )
    output_gate_weights = np.random.normal(
        mean, standard_deviation, (input_size + hidden_size, hidden_size)
    )
    gate_gate_weights = np.random.normal(
        mean, standard_deviation, (input_size + hidden_size, hidden_size)
    )
    hidden_output_weights = np.random.normal(
        mean, standard_deviation, (hidden_size, output_size)
    )

    lstm = LSTM(
        forget_gate_weights=forget_gate_weights,
        input_gate_weights=input_gate_weights,
        output_gate_weights=output_gate_weights,
        gate_gate_weights=gate_gate_weights,
        hidden_output_weights=hidden_output_weights,
    )

    return lstm

def get_lstm_activations(
    batch_dataset: np.array,
    prev_activation_matrix: np.array,
    prev_cell_matrix: np.array,
    lstm: LSTM,
) -> Tuple[LSTMActivations, np.array, np.array]:
    """
    Compute the forward pass of a single LSTM cell.

    Args:
        batch_dataset (np.array): The input batch dataset of shape (batch_size, input_size).
        prev_activation_matrix (np.array): The previous activation matrix of shape (batch_size, hidden_size).
        prev_cell_matrix (np.array): The previous cell matrix of shape (batch_size, hidden_size).
        lstm (LSTM): The LSTM object containing the weights.

    Returns:
        Tuple[LSTMActivations, np.array, np.array]:
            - lstm_activations (LSTMActivations): The LSTMActivations object containing the activations.
            - new_cell_matrix (np.array): The new cell matrix of shape (batch_size, hidden_size).
            - new_activation_matrix (np.array): The new activation matrix of shape (batch_size, hidden_size).
    """
    # Concatenate batch dataset and prev_activation_matrix
    concat_dataset = np.concatenate((batch_dataset, prev_activation_matrix), axis=1)

    # Calculate forget gate activations
    forget_activations = sigmoid(np.matmul(concat_dataset, lstm.forget_gate_weights))

    # Calculate input gate activations
    input_activations = sigmoid(np.matmul(concat_dataset, lstm.input_gate_weights))

    # Calculate output gate activations
    output_activations = sigmoid(np.matmul(concat_dataset, lstm.output_gate_weights))

    # Calculate gate gate activations
    gate_activations = tanh_activation(np.matmul(concat_dataset, lstm.gate_gate_weights))

    # Calculate new cell memory matrix
    cell_memory_matrix = np.multiply(
        forget_activations, prev_cell_matrix
    ) + np.multiply(input_activations, gate_activations)

    # Calculate current activation matrix
    activation_matrix = np.multiply(
        output_activations, tanh_activation(cell_memory_matrix)
    )

    # Store the activations to be used in backpropagation
    lstm_activations = LSTMActivations(
        forget_gate_activation=forget_activations,
        input_gate_activation=input_activations,
        output_gate_activation=output_activations,
        gate_gate_activation=gate_activations,
    )

    return lstm_activations, cell_memory_matrix, activation_matrix

def get_lstm_output(activation_matrix: np.array, lstm: LSTM) -> np.array:
    """
    Generates the output for a given activation matrix using a LSTM model.

    Parameters:
    - activation_matrix: numpy.array
        The activation matrix used as input to the LSTM model.
    - lstm: LSTM
        The LSTM model used to generate the output.

    Returns:
    - numpy.array
        The output matrix generated by the LSTM model.
    """
    return softmax(np.matmul(activation_matrix, lstm.hidden_output_weights))

def forward_propagation(
    batches: List[np.array], lstm: LSTM, embeddings: np.array
) -> Tuple[
    List[np.array], List[np.array], List[np.array], List[np.array], List[np.array]
]:
    """
    Performs forward propagation on a given LSTM model.

    Args:
        batches (List[np.array]): A list of numpy arrays representing the input batches.
        lstm (LSTM): An instance of the LSTM model.
        embeddings (np.array): A numpy array representing the embeddings.

    Returns:
        Tuple[List[np.array, np.array, np.array, np.array, np.array]]:
            - embedding_history (List[np.array]): A list of numpy arrays representing the embedding matrices.
            - lstm_history (List[np.array]): A list of numpy arrays representing the LSTM activations.
            - activation_history (List[np.array]): A list of numpy arrays representing the activation matrices.
            - cell_history (List[np.array]): A list of numpy arrays representing the cell matrices.
            - output_history (List[np.array]): A list of numpy arrays representing the output matrices.
    """
    # get batch size
    batch_size = batches[0].shape[0]

    # to store the activations of all the unrollings.
    lstm_history = []
    activation_history = []
    cell_history = []
    output_history = []
    embedding_history = []

    # initial activation_matrix(a0) and cell_matrix(c0)
    a0 = np.zeros([batch_size, lstm.hidden_output_weights.shape[0]], dtype=np.float32)
    c0 = np.zeros([batch_size, lstm.hidden_output_weights.shape[0]], dtype=np.float32)

    # store the initial activations in history
    activation_history.append(a0)
    cell_history.append(c0)

    # unroll the names
    for i in range(len(batches) - 1):
        # get first first character batch
        batch_dataset = batches[i]

        # get embeddings
        batch_dataset = get_batch_embeddings(batch_dataset, embeddings)
        embedding_history.append(batch_dataset)

        # lstm cell
        lstm_activations, ct, at = get_lstm_activations(batch_dataset, a0, c0, lstm)

        # output cell
        ot = get_lstm_output(at, lstm)

        # store the time 't' activations in historys
        lstm_history.append(lstm_activations)
        activation_history.append(at)
        cell_history.append(ct)
        output_history.append(ot)

        # update a0 and c0 to new 'at' and 'ct' for next lstm cell
        a0 = at
        c0 = ct

    return (
        embedding_history,
        lstm_history,
        activation_history,
        cell_history,
        output_history,
    )

def predict(lstm:LSTM, embeddings:np.array, token_id_mapping:dict, tokens_size:int, end_token:str = ".", itterations:int = 20):
    # to store some predictions
    predictions = []

    # predict 20 names
    for i in range(itterations):
        # initial activation_matrix(a0) and cell_matrix(c0)
        a0 = np.zeros([1, lstm.hidden_output_weights.shape[0]], dtype=np.float32)
        c0 = np.zeros([1, lstm.hidden_output_weights.shape[0]], dtype=np.float32)

        # initalize blank name
        pred_item = ""

        # make a batch dataset of single char
        batch_dataset = np.zeros([1, tokens_size])

        # get random start character
        index = np.random.randint(0, tokens_size, 1)[0]

        # make that index 1.0
        batch_dataset[0, index] = 1.0

        # add first char to name
        pred_item += token_id_mapping[index]

        # get char from id_char dict
        char = token_id_mapping[index]

        # loop until algo predicts '.'
        for _ in range(60):
            # get embeddings
            batch_dataset = get_batch_embeddings(batch_dataset, embeddings)

            # lstm cell
            lstm_activations, ct, at = get_lstm_activations(batch_dataset, a0, c0, lstm)

            # output cell
            ot = get_lstm_output(at, lstm)

            # either select random.choice ot np.argmax
            pred = np.random.choice(tokens_size, 1, p=ot[0])[0]

            # get predicted char index
            # pred = np.argmax(ot)

            # add char to name
            pred_item += token_id_mapping[pred]

            char = token_id_mapping[pred]

            # change the batch_dataset to this new predicted char
            batch_dataset = np.zeros([1, tokens_size])
            batch_dataset[0, pred] = 1.0

            # update a0 and c0 to new 'at' and 'ct' for next lstm cell
            a0 = at
            c0 = ct

            if char == end_token:
                break

        # append the predicted name to names list
        predictions.append(pred_item)

    return predictions