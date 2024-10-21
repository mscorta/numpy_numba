import unittest
from typing import List

from similarity.calculate_cosine_similarity import calculate_cosine_similarity


def repeat_list(input_list: List[float], n: int = 1000) -> List[float]:
    """Repeat each element in the input list n times.

    Args:
        input_list (List[float]): List of floats to be repeated.
        n (int, optional): Number of times to repeat each element. Defaults to 1000.

    Returns:
        List[float]: New list with each element repeated n times.
    """
    return input_list * n


class TestCosineSimilarity(unittest.TestCase):
    def test_cosine_similarity(self) -> None:  # No return value for this method
        N = 1000_000  # Constant for repetition

        # Test case 1: Simple test with identical vectors
        vector_a: List[float] = repeat_list([1.0, 2.0, 3.0], N)  # Repeated list
        vector_b: List[float] = repeat_list([1.0, 2.0, 3.0], N)  # Repeated list
        result: float = calculate_cosine_similarity(vector_a, vector_b)
        self.assertAlmostEqual(result, 1.0)

        # Test case 2: Orthogonal vectors
        vector_a = repeat_list([1.0, 0.0, 0.0], N)  # Repeated list
        vector_b = repeat_list([0.0, 1.0, 0.0], N)  # Repeated list
        result = calculate_cosine_similarity(vector_a, vector_b)
        self.assertAlmostEqual(result, 0.0)

        # Test case 3: Opposite vectors
        vector_a = repeat_list([1.0, 2.0, 3.0], N)  # Repeated list
        vector_b = repeat_list([-1.0, -2.0, -3.0], N)  # Repeated list
        result = calculate_cosine_similarity(vector_a, vector_b)
        self.assertAlmostEqual(result, -1.0)

        # Test case 4: Different magnitudes, same direction
        vector_a = repeat_list([1.0, 2.0, 3.0], N)  # Repeated list
        vector_b = repeat_list([2.0, 4.0, 6.0], N)  # Repeated list
        result = calculate_cosine_similarity(vector_a, vector_b)
        self.assertAlmostEqual(result, 1.0)

        # Test case 5: Vectors with a zero vector
        vector_a = repeat_list([0.0, 0.0, 0.0], N)  # Repeated list
        vector_b = repeat_list([1.0, 2.0, 3.0], N)  # Repeated list
        with self.assertRaises(ValueError):
            calculate_cosine_similarity(vector_a, vector_b)

        # Test case 6: Vectors with only zeros
        vector_a = repeat_list([0.0, 0.0, 0.0], N)  # Repeated list
        vector_b = repeat_list([0.0, 0.0, 0.0], N)  # Repeated list
        with self.assertRaises(ValueError):
            calculate_cosine_similarity(vector_a, vector_b)

        # Additional test case: Testing the repeat_list function
        repeated_vector_a = repeat_list([1.0, 2.0, 3.0], N)  # Repeated list
        repeated_vector_b = repeat_list([1.0, 2.0, 3.0], N)  # Repeated list
        result = calculate_cosine_similarity(repeated_vector_a, repeated_vector_b)
        print(f"Cosine Similarity of repeated vectors: {result}")

if __name__ == "__main__":
    unittest.main()
