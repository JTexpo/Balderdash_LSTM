from typing import List

import numpy as np

def initialize_embeddings(mean:float, standard_deviation: float, tokens:List[str], input_size:int) -> np.array:
    """
    Generate a numpy array of embeddings for a given list of tokens.

    Args:
        mean (float): The mean value for the normal distribution.
        standard_deviation (float): The standard deviation value for the normal distribution.
        tokens (List[str]): A list of tokens for which embeddings are generated.
        input_size (int): The number of input units for the embeddings.

    Returns:
        np.array: A numpy array of embeddings with shape (len(tokens), input_size).
    """
    
    return np.random.normal(mean, standard_deviation, (len(tokens), input_size))

def get_batch_embeddings(batch_dataset: np.array, embeddings: np.array) -> np.array:
    """
    Calculates the embeddings for a batch dataset using a given set of embeddings.

    Args:
        batch_dataset (np.array): The batch dataset to calculate the embeddings for.
        embeddings (np.array): The embeddings to use for calculating the embeddings.

    Returns:
        np.array: The calculated embedding dataset.
    """
    return np.matmul(batch_dataset, embeddings)
