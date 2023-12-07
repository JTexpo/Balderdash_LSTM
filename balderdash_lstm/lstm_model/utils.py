from typing import Tuple, List, Dict

import numpy as np
import pandas as pd

'''
Getters
-------
'''
def get_tokens()->Tuple[List[str],dict,dict]:

    tokens = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z','.']
    tokens_to_id = {}
    id_to_tokens = {}
    for index, char in enumerate(tokens):
        id_to_tokens[index] = char
        tokens_to_id[char] = index

    return tokens, tokens_to_id, id_to_tokens

def load_train_data(path: str, tokens: List[str], token_to_id: Dict[str, int]) -> List[List[np.ndarray]]:
    data = pd.read_csv(path)
    names = data["WORD"].str.lower().values.reshape(-1, 1)
    transform_data = np.copy(names)

    max_length = max(len(name) for name in names[:, 0])
    transform_data = [name.ljust(max_length, '.') for name in transform_data[:, 0]]
    train_dataset = []

    batch_size = 20

    for i in range(len(transform_data) - batch_size + 1):
        start = i * batch_size
        end = start + batch_size

        batch_data = transform_data[start:end]

        if len(batch_data) != batch_size:
            break

        char_list = []
        for k in range(len(batch_data[0])):
            batch_dataset = np.zeros([batch_size, len(tokens)])
            for j in range(batch_size):
                name = batch_data[j]
                char_index = token_to_id[name[k]]
                batch_dataset[j, char_index] = 1.0

            char_list.append(batch_dataset)

        train_dataset.append(char_list)

    return train_dataset

'''
Activation Functions
--------------------
'''
# sigmoid
def sigmoid(X):
    return 1 / (1 + np.exp(-X))

# softmax activation
def softmax(X):
    exp_X = np.exp(X)
    exp_X_sum = np.sum(exp_X, axis=1).reshape(-1, 1)
    exp_X = exp_X / exp_X_sum
    return exp_X

# tanh activation
def tanh_activation(X):
    return np.tanh(X)

# derivative of tanh
def tanh_derivative(X):
    return 1 - (X**2)
