import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")


from typing import Tuple, List
import time 

import numpy as np

from lstm_model.utils import tanh_activation, tanh_derivative, get_tokens, load_train_data
from lstm_model.lstm import LSTM, LSTMActivations, LSTMErrors, LSTMDerivative, forward_propagation, initialize_lstm, predict
from lstm_model.optimizer import Optimizer, initialize_optimizer
from lstm_model.embeddings import initialize_embeddings

'''
SAVE / LOAD
-----------
'''
def save_models(file_path:str, lstm:LSTM, optimizer:Optimizer, embeddings:np.array):
    np.savez(
        file_path,

        forget_gate_weights=lstm.forget_gate_weights,
        input_gate_weights=lstm.input_gate_weights,
        output_gate_weights=lstm.output_gate_weights,
        gate_gate_weights=lstm.gate_gate_weights,
        hidden_output_weights=lstm.hidden_output_weights,

        v_forget_gate_weights = optimizer.v_forget_gate_weights,
        v_input_gate_weights = optimizer.v_input_gate_weights,
        v_output_gate_weights = optimizer.v_output_gate_weights,
        v_gate_gate_weights = optimizer.v_gate_gate_weights,
        v_hidden_output_weights = optimizer.v_hidden_output_weights,

        s_forget_gate_weights = optimizer.s_forget_gate_weights,
        s_input_gate_weights = optimizer.s_input_gate_weights,
        s_output_gate_weights = optimizer.s_output_gate_weights,
        s_gate_gate_weights = optimizer.s_gate_gate_weights,
        s_hidden_output_weights = optimizer.s_hidden_output_weights,

        learning_rate = optimizer.learning_rate,
        beta_1 = optimizer.beta_1,
        beta_2 = optimizer.beta_2,

        embeddings=embeddings,
    )

def load_models(file_path:str) -> Tuple[LSTM, Optimizer, np.array]:
    loaded_models = np.load(file_path)

    embeddings = loaded_models["embeddings"]

    lstm = LSTM(
        forget_gate_weights=loaded_models["forget_gate_weights"],
        input_gate_weights=loaded_models["input_gate_weights"],
        output_gate_weights=loaded_models["output_gate_weights"],
        gate_gate_weights=loaded_models["gate_gate_weights"],
        hidden_output_weights=loaded_models["hidden_output_weights"],
    )

    optimizer = Optimizer(
        v_forget_gate_weights=loaded_models["v_forget_gate_weights"],
        v_input_gate_weights=loaded_models["v_input_gate_weights"],
        v_output_gate_weights=loaded_models["v_output_gate_weights"],
        v_gate_gate_weights=loaded_models["v_gate_gate_weights"],
        v_hidden_output_weights=loaded_models["v_hidden_output_weights"],
        s_forget_gate_weights=loaded_models["s_forget_gate_weights"],
        s_input_gate_weights=loaded_models["s_input_gate_weights"],
        s_output_gate_weights=loaded_models["s_output_gate_weights"],
        s_gate_gate_weights=loaded_models["s_gate_gate_weights"],
        s_hidden_output_weights=loaded_models["s_hidden_output_weights"],
        learning_rate=loaded_models["learning_rate"],
        beta_1=loaded_models["beta_1"],
        beta_2=loaded_models["beta_2"],
    )
    return lstm, optimizer, embeddings

'''
LOSS AND ERRORS
---------------
'''
def cal_loss_accuracy(
    batch_labels: List[np.array], output_history: dict
) -> Tuple[float, float, float]:
    """
    Calculate the loss, accuracy, and perplexity of a batch of labels and output history.

    Args:
        batch_labels (List[np.array]): A list of numpy arrays representing the labels for each time step in the batch.
        output_history (dict): A dictionary representing the output history for each time step.

    Returns:
        Tuple[float, float, float]:
            - perplexity (float): The calculated perplexity for the batch.
            - loss (float): The calculated loss for the batch.
            - accuracy (float): The calculated accuracy for the batch.
    """
    loss:float = 0  # to sum loss for each time step
    accuracy:float = 0  # to sum accuracy for each time step
    prob:float = 1  # probability product of each time step predicted char

    # batch size
    batch_size = batch_labels[0].shape[0]

    # loop through each time step
    for index, past_output in enumerate(output_history):
        # get true labels and predictions
        labels = batch_labels[index + 1]
        pred = past_output

        prob = np.multiply(
            prob, np.sum(np.multiply(labels, pred), axis=1).reshape(-1, 1)
        )
        loss += np.sum(
            (
                np.multiply(labels, np.log(pred))
                + np.multiply(1 - labels, np.log(1 - pred))
            ),
            axis=1,
        ).reshape(-1, 1)
        accuracy += np.array(
            np.argmax(labels, 1) == np.argmax(pred, 1), dtype=np.float32
        ).reshape(-1, 1)

    # calculate perplexity loss and accuracy
    perplexity = np.sum((1 / prob) ** (1 / len(output_history))) / batch_size
    loss = np.sum(loss) * (-1 / batch_size)
    accuracy = np.sum(accuracy) / (batch_size)
    accuracy = accuracy / len(output_history)

    return perplexity, loss, accuracy

def calculate_output_cell_error(
    batch_labels: List[np.array], output_history: List[np.array], lstm: LSTM
) -> Tuple[List[np.array], List[np.array]]:
    """
    Calculate the output cell error for each time step in a given batch.

    Args:
        batch_labels (List[np.array]): The true labels for each time step in the batch.
        output_history (List[np.array]): The output history for each time step in the batch.
        lstm (LSTM): The LSTM model used for the calculations.

    Returns:
        Tuple[List[np.array], List[np.array]]:
            - output_error_history (List[np.array]): The output error history for each time step in the batch.
            - activation_error_history (List[np.array]): The activation error history for each time step in the batch.
    """

    # store the output errors for each time step
    output_error_history:List[np.array] = []
    activation_error_history: List[np.array]= []

    # loop through each time step
    for index, past_output in enumerate(output_history):
        # get true labels and predictions
        labels = batch_labels[index + 1]
        pred = past_output

        # calculate the output_error for time step 't'
        error_output = pred - labels

        # calculate the activation error for time step 't'
        error_activation = np.matmul(error_output, lstm.hidden_output_weights.T)

        # store the output and activation error in dict
        output_error_history.append(error_output)
        activation_error_history.append(error_activation)

    return output_error_history, activation_error_history

def calculate_single_lstm_cell_error(
    activation_output_error: np.array,
    next_activation_error: np.array,
    next_cell_error: np.array,
    lstm: LSTM,
    lstm_activation: LSTMActivations,
    cell_activation: np.array,
    prev_cell_activation: np.array,
) -> Tuple[np.array, np.array, np.array, LSTMErrors]:
    """
    Calculate the error for a single LSTM cell.

    Args:
        activation_output_error (np.array): The error in the activation output.
        next_activation_error (np.array): The error in the next activation.
        next_cell_error (np.array): The error in the next cell.
        lstm (LSTM): The LSTM object.
        lstm_activation (LSTMActivations): The activation object for the LSTM.
        cell_activation (np.array): The current cell activation.
        prev_cell_activation (np.array): The previous cell activation.

    Returns:
        Tuple[np.array, np.array, np.array, dict]:
            - prev_activation_error (np.array): The previous activation error for the LSTM cell.
            - prev_cell_error (np.array): The previous cell error for the LSTM cell.
            - embed_error (np.array): The embedding error for the LSTM cell.
            - lstm_error (LSTMErrors): The LSTM error for the LSTM cell.
    """

    # Calculate the error for a single LSTM cell
    activation_error = activation_output_error + next_activation_error

    # Calculate the output error
    error_output = (
        activation_error
        * tanh_activation(cell_activation)
        * lstm_activation.output_gate_activation
        * (1 - lstm_activation.output_gate_activation)
    )

    # Calculate the cell error
    cell_error = (
        activation_error
        * lstm_activation.output_gate_activation
        * tanh_derivative(tanh_activation(cell_activation))
    ) + next_cell_error

    # Calculate the input error
    error_input = (
        cell_error
        * lstm_activation.gate_gate_activation
        * lstm_activation.input_gate_activation
        * (1 - lstm_activation.input_gate_activation)
    )

    # Calculate the gate error
    error_gate = (
        cell_error
        * lstm_activation.input_gate_activation
        * tanh_derivative(lstm_activation.gate_gate_activation)
    )

    # Calculate the forget error
    error_forget = (
        cell_error
        * prev_cell_activation
        * lstm_activation.forget_gate_activation
        * (1 - lstm_activation.forget_gate_activation)
    )

    # Calculate the previous cell error
    prev_cell_error = cell_error * lstm_activation.forget_gate_activation

    # Calculate the embed error
    embed_activation_error = (
        np.matmul(error_forget, lstm.forget_gate_weights.T)
        + np.matmul(error_input, lstm.input_gate_weights.T)
        + np.matmul(error_output, lstm.output_gate_weights.T)
        + np.matmul(error_gate, lstm.gate_gate_weights.T)
    )

    # Calculate the input and hidden units
    input_hidden_units = lstm.forget_gate_weights.shape[0]
    hidden_units = lstm.forget_gate_weights.shape[1]
    input_size = input_hidden_units - hidden_units

    # Calculate the previous activation error
    prev_activation_error = embed_activation_error[:, input_size:]
    embed_error = embed_activation_error[:, :input_size]

    lstm_error = LSTMErrors(
        forget_gate_error=error_forget,
        input_gate_error=error_input,
        output_gate_error=error_output,
        gate_gate_error=error_gate,
    )

    return prev_activation_error, prev_cell_error, embed_error, lstm_error

'''
DERIVATIVES
-----------
'''
def calculate_output_cell_derivatives(
    output_error_history: List[np.array], activation_history: List[np.array], lstm: LSTM
) -> np.array:
    """
    Calculate the derivatives of the output cell weights with respect to the hidden state.

    Parameters:
    - output_error_history (List[np.array]): A list of numpy arrays representing the error in the output at each time step.
    - activation_history (List[np.array]): A list of numpy arrays representing the activation values of the hidden state at each time step.
    - lstm (LSTM): An instance of the LSTM class.

    Returns:
    - np.array: A numpy array representing the sum of derivatives from each time step.
    """
    # to store the sum of derivatives from each time step
    d_hidden_output_weights = np.zeros(lstm.hidden_output_weights.shape)

    batch_size = activation_history[1].shape[0]

    # loop through the time steps
    for output_error, activation in zip(output_error_history, activation_history[1:]):
        # cal derivative and summing up!
        d_hidden_output_weights += np.matmul(activation.T, output_error) / batch_size

    return d_hidden_output_weights

def calculate_single_lstm_cell_derivatives(
    lstm_error: LSTMErrors, embedding_matrix: np.array, activation_matrix: np.array
) -> LSTMDerivative:
    """
    Calculate the derivatives for a single LSTM cell based on the given LSTM error, embedding matrix, and activation matrix.

    Args:
        lstm_error (dict): A dictionary containing the error values for the forget gate, input gate, output gate, and gate gate.
        embedding_matrix (np.array): The embedding matrix for the LSTM cell.
        activation_matrix (np.array): The activation matrix for the LSTM cell.

    Returns:
        dict: A dictionary containing the derivatives for the forget gate weights, input gate weights, output gate weights, and gate gate weights.
    """
    # get error for single time step

    # get input activations for this time step
    concat_matrix = np.concatenate((embedding_matrix, activation_matrix), axis=1)

    batch_size = embedding_matrix.shape[0]

    # cal derivatives for this time step
    derivatives = LSTMDerivative(
        forget_gate_derivative=np.matmul(concat_matrix.T, lstm_error.forget_gate_error)
        / batch_size,
        input_gate_derivative=np.matmul(concat_matrix.T, lstm_error.input_gate_error)
        / batch_size,
        output_gate_derivative=np.matmul(concat_matrix.T, lstm_error.output_gate_error)
        / batch_size,
        gate_gate_derivative=np.matmul(concat_matrix.T, lstm_error.gate_gate_error)
        / batch_size,
    )

    return derivatives

'''
BACKWARDS PROPAGATION
---------------------
'''
def backward_propagation(
    batch_labels: List[np.array],
    embedding_history: List[np.array],
    lstm_history: List[LSTMActivations],
    activation_history: List[np.array],
    cell_history: List[np.array],
    output_history: List[np.array],
    lstm: LSTM,
) -> Tuple[LSTMDerivative, List[np.array]]:
    """
    Performs backward propagation through time for a LSTM network.

    Args:
        batch_labels (List[np.array]): A list of numpy arrays containing the labels for each batch.
        embedding_history (List[np.array]): A list of numpy arrays containing the embedding values for each time step.
        lstm_history (List[LSTMActivations]): A list of LSTMActivations objects representing the LSTM activations for each time step.
        activation_history (List[np.array]): A list of numpy arrays containing the activation values for each time step.
        cell_history (List[np.array]): A list of numpy arrays containing the cell values for each time step.
        output_history (List[np.array]): A list of numpy arrays containing the output values for each time step.
        lstm (LSTM): An instance of the LSTM class.

    Returns:
        Tuple[LSTMDerivative, List[np.array]]:
            - LSTMDerivative: A LSTMDerivative object containing the derivatives for the LSTM cell weights.
            - List[np.array]: A list of numpy arrays containing the error values for each time step.
    """

    # calculate output errors
    output_error_history, activation_error_history = calculate_output_cell_error(
        batch_labels, output_history, lstm
    )

    # to store lstm error for each time step
    lstm_error_history = []

    # to store embeding errors for each time step
    embedding_error_history = []

    # next activation error
    # next cell error
    # for last cell will be zero
    eat = np.zeros(activation_error_history[1].shape)
    ect = np.zeros(activation_error_history[1].shape)

    # calculate all lstm cell errors (going from last time-step to the first time step)
    for index in range(len(lstm_history) - 1, -1, -1):
        # calculate the lstm errors for this time step 't'
        pae, pce, ee, le = calculate_single_lstm_cell_error(
            activation_error_history[index],
            eat,
            ect,
            lstm,
            lstm_history[index],
            cell_history[index + 1],
            cell_history[index],
        )

        # store the lstm error in dict
        lstm_error_history.insert(0, le)

        # store the embedding error in dict
        embedding_error_history.insert(0, ee)

        # update the next activation error and next cell error for previous cell
        eat = pae
        ect = pce

    # calculate output cell derivatives
    derivatives = LSTMDerivative(
        forget_gate_derivative=np.zeros(lstm.forget_gate_weights.shape),
        input_gate_derivative=np.zeros(lstm.input_gate_weights.shape),
        output_gate_derivative=np.zeros(lstm.output_gate_weights.shape),
        gate_gate_derivative=np.zeros(lstm.gate_gate_weights.shape),
        hidden_output_derivative=np.zeros(lstm.hidden_output_weights.shape),
    )
    derivatives.hidden_output_derivative = calculate_output_cell_derivatives(
        output_error_history, activation_history, lstm
    )

    # calculate lstm cell derivatives for each time step and store in lstm_derivatives dict
    for index, lstm_error in enumerate(lstm_error_history):
        derv = calculate_single_lstm_cell_derivatives(
            lstm_error,
            embedding_history[index],
            activation_history[index],
        )
        derivatives.forget_gate_derivative += derv.forget_gate_derivative
        derivatives.input_gate_derivative += derv.input_gate_derivative
        derivatives.output_gate_derivative += derv.output_gate_derivative
        derivatives.gate_gate_derivative += derv.gate_gate_derivative

    return derivatives, embedding_error_history

def update_lstm(lstm: LSTM, derivatives: LSTMDerivative, optimizer: Optimizer):
    # calculate the V lstm from V and current derivatives
    optimizer.v_forget_gate_weights = (
        optimizer.beta_1 * optimizer.v_forget_gate_weights
        + (1 - optimizer.beta_1) * derivatives.forget_gate_derivative
    )
    optimizer.v_input_gate_weights = (
        optimizer.beta_1 * optimizer.v_input_gate_weights
        + (1 - optimizer.beta_1) * derivatives.input_gate_derivative
    )
    optimizer.v_output_gate_weights = (
        optimizer.beta_1 * optimizer.v_output_gate_weights
        + (1 - optimizer.beta_1) * derivatives.output_gate_derivative
    )
    optimizer.v_gate_gate_weights = (
        optimizer.beta_1 * optimizer.v_gate_gate_weights
        + (1 - optimizer.beta_1) * derivatives.gate_gate_derivative
    )
    optimizer.v_hidden_output_weights = (
        optimizer.beta_1 * optimizer.v_hidden_output_weights
        + (1 - optimizer.beta_1) * derivatives.hidden_output_derivative
    )

    # calculate the S lstm from S and current derivatives
    optimizer.s_forget_gate_weights = optimizer.beta_2 * optimizer.s_forget_gate_weights + (
        1 - optimizer.beta_2
    ) * (derivatives.forget_gate_derivative**2)
    optimizer.s_input_gate_weights = optimizer.beta_2 * optimizer.s_input_gate_weights + (
        1 - optimizer.beta_2
    ) * (derivatives.input_gate_derivative**2)
    optimizer.s_output_gate_weights = optimizer.beta_2 * optimizer.s_output_gate_weights + (
        1 - optimizer.beta_2
    ) * (derivatives.output_gate_derivative**2)
    optimizer.s_gate_gate_weights = optimizer.beta_2 * optimizer.s_gate_gate_weights + (
        1 - optimizer.beta_2
    ) * (derivatives.gate_gate_derivative**2)
    optimizer.s_hidden_output_weights = optimizer.beta_2 * optimizer.s_hidden_output_weights + (
        1 - optimizer.beta_2
    ) * (derivatives.hidden_output_derivative**2)

    # update the lstm
    lstm.forget_gate_weights = lstm.forget_gate_weights - optimizer.learning_rate * (
        (optimizer.v_forget_gate_weights) / (np.sqrt(optimizer.s_forget_gate_weights) + 1e-6)
    )
    lstm.input_gate_weights = lstm.input_gate_weights - optimizer.learning_rate * (
        (optimizer.v_input_gate_weights) / (np.sqrt(optimizer.s_input_gate_weights) + 1e-6)
    )
    lstm.output_gate_weights = lstm.output_gate_weights - optimizer.learning_rate * (
        (optimizer.v_output_gate_weights) / (np.sqrt(optimizer.s_output_gate_weights) + 1e-6)
    )
    lstm.gate_gate_weights = lstm.gate_gate_weights - optimizer.learning_rate * (
        (optimizer.v_gate_gate_weights) / (np.sqrt(optimizer.s_gate_gate_weights) + 1e-6)
    )
    lstm.hidden_output_weights = lstm.hidden_output_weights - optimizer.learning_rate * (
        (optimizer.v_hidden_output_weights) / (np.sqrt(optimizer.s_hidden_output_weights) + 1e-6)
    )

    return lstm, optimizer


def update_embeddings(
    embeddings: np.array,
    embedding_error_history: List[np.array],
    batch_labels: List[np.array],
    optimizer: Optimizer
) -> np.array:
    """
    Update the embeddings based on the embedding error history using the given learning rate.

    Parameters:
        embeddings (np.array): The current embeddings.
        embedding_error_history (List[np.array]): A list of embedding errors at each time step.
        batch_labels (List[np.array]): A list of labels for each time step.
        learning_rate (float): The learning rate.

    Returns:
        np.array: The updated embeddings.
    """
    # to store the embeddings derivatives
    embedding_derivatives = np.zeros(embeddings.shape)

    batch_size = batch_labels[0].shape[0]

    # sum the embedding derivatives for each time step
    for index, embedding_error in enumerate(embedding_error_history):
        embedding_derivatives += (
            np.matmul(batch_labels[index].T, embedding_error) / batch_size
        )

    # update the embeddings
    embeddings = embeddings - optimizer.learning_rate * embedding_derivatives

    return embeddings

'''
TRAIN
-----
'''

def train(
    embeddings:np.array,
    lstm:LSTM,
    optimizer:Optimizer,
    train_dataset: List[np.array],
    iters=1000
):
    # to store the Loss, Perplexity and Accuracy for each batch
    losses = []
    perplexities = []
    accuracies = []

    for step in range(iters):
        # get batch dataset
        index = step % len(train_dataset)
        batches = train_dataset[index]

        # forward propagation
        (
            embedding_history,
            lstm_history,
            activation_history,
            cell_history,
            output_history,
        ) = forward_propagation(
            batches=batches, lstm=lstm, embeddings=embeddings)

        # calculate the loss, perplexity and accuracy
        perplexity, loss, accuracy = cal_loss_accuracy(batch_labels=batches, output_history=output_history)

        # backward propagation
        derivatives, embedding_error_history = backward_propagation(
            batch_labels=batches,
            embedding_history=embedding_history,
            lstm_history=lstm_history,
            activation_history=activation_history,
            cell_history=cell_history,
            output_history=output_history,
            lstm=lstm,
        )

        # update the lstm
        lstm, optimizer = update_lstm(lstm=lstm, derivatives=derivatives, optimizer=optimizer)

        # update the embeddings
        embeddings = update_embeddings(embeddings=embeddings, embedding_error_history=embedding_error_history, batch_labels=batches,optimizer=optimizer)

        losses.append(loss)
        perplexities.append(perplexity)
        accuracies.append(accuracy)

        # print loss, accuracy and perplexity
        if step % 100 == 0:
            print("For Single Batch :")
            print("Step       = {}".format(step))
            print("Loss       = {}".format(round(loss, 2)))
            print("Perplexity = {}".format(round(perplexity, 2)))
            print("Accuracy   = {}".format(round(accuracy * 100, 2)))
            print()
            time.sleep(1)

            save_models(file_path=f"./models/lstm_model_{step//100}", lstm=lstm, optimizer=optimizer, embeddings=embeddings)

    return embeddings, lstm, losses, perplexities, accuracies

if __name__ == "__main__":

    # load the vocabulary
    vocab, char_id, id_char = get_tokens()

    # load the dataset
    train_dataset = load_train_data(
        path="./assets/words.csv",
        tokens=vocab,
        token_to_id=char_id
    )

    # initialize the lstm
    lstm = initialize_lstm(
        input_size=len(vocab),
        hidden_size=len(vocab)*3,
        output_size=len(vocab),
        mean=0,
        standard_deviation=.01
    )
    optimizer = initialize_optimizer(
        lstm=lstm,
        beta_1=0.9,
        beta_2=0.99,
        learning_rate=.01
    )
    embeddings = initialize_embeddings(
        mean=0,
        standard_deviation=.01,
        tokens=vocab,
        input_size=len(vocab)
    )

    # lstm, optimizer, embeddings = load_models(file_path=f"./models/lstm_model_24.npz")

    embeddings, lstm, losses, perplexities, accuracies = train(
        embeddings=embeddings,
        lstm=lstm,
        optimizer=optimizer,
        train_dataset=train_dataset,
        iters=2501
    )

    print(predict(lstm=lstm, embeddings=embeddings, token_id_mapping=id_char,tokens_size=len(vocab)))
    