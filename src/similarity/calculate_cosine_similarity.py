import numpy as np
from typing import List
import numba as nb

@nb.njit
def calculate_cosine_similarity(vector_a: List[float], vector_b: List[float]) -> float:
    """
    Calculate the cosine similarity between two vectors.

    Parameters:
        vector_a (List[float]): The first input vector.
        vector_b (List[float]): The second input vector.

    Returns:
        float: The cosine similarity between the two vectors.
    """

    # Convert lists to NumPy arrays
    np_vector_a: np.ndarray = np.array(vector_a)
    np_vector_b: np.ndarray = np.array(vector_b)

    # Compute dot product of the two vectors
    dot_product: float = np.dot(np_vector_a, np_vector_b)

    # Calculate the norms of both vectors
    norm_a: float = np.linalg.norm(np_vector_a)
    if norm_a == 0:
        raise ValueError(f"Norm is zero for {vector_a}")

    norm_b: float = np.linalg.norm(np_vector_b)
    if norm_b == 0:
        raise ValueError(f"Norm is zero for {vector_b}")

    # Compute the cosine similarity
    # print(vector_a, vector_b, norm_a, norm_b)
    cosine_similarity: float = dot_product / (norm_a * norm_b)

    return cosine_similarity
