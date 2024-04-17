import unittest
import numpy as np
from llm_similarity_analysis.utilities.utility_functions import (
    vectorise_dataset,
    combine_into_corpus,
)


class test_utility_functions(unittest.TestCase):
    def test_vectorise_dataset_default(self):
        """
        Test that the vectorization function returns the correct type of output, as the vectorisation itself is already tested.
        """
        text_tokens1 = {"1": ["apple", "pear", "banana", "mango", "kiwi"]}

        vector = vectorise_dataset(text_tokens1)
        self.assertIsInstance(vector, dict)
        self.assertEqual(type(vector["apple pear banana mango kiwi"]), np.ndarray)

    def test_combine_into_corpus(self):
        """Test that the function correctly vectorises into a single corpus"""
        text_tokens1 = {
            "1": ["apple", "pear", "banana", "mango", "kiwi"],
            "2": ["dog", "cat", "monkey", "rabbit", "koala"],
        }

        corpus = combine_into_corpus(text_tokens1)
        valid = [
            "apple",
            "pear",
            "banana",
            "mango",
            "kiwi",
            "dog",
            "cat",
            "monkey",
            "rabbit",
            "koala",
        ]
        self.assertListEqual(corpus["Full Text Corpus"], valid)


if __name__ == "__main__":
    unittest.main()
