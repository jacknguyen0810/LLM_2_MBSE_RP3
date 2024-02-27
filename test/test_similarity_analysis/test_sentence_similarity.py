import unittest
import numpy as np
from llm_similarity_analysis.similarity.sentence_similarity import (
    SentenceSimilarityAnalysis,
)


class TestSentenceSimilarityAnalysis(unittest.TestCase):

    def test_run_default(self):
        text_tokens1 = {"1": ["apple", "pear", "banana", "mango", "kiwi"]}

        text_tokens2 = {"1": ["dog", "cat", "monkey", "rabbit", "koala"]}

        analysis = SentenceSimilarityAnalysis(text_tokens1, text_tokens2)

        self.assertEqual(analysis.metric, "cosine")
        self.assertEqual(analysis.vector_model, "all-mpnet-base-V2")
        self.assertIsNone(analysis.vectors1)
        self.assertIsNone(analysis.vectors2)
        self.assertIsNone(analysis.output)

        analysis.run()

        valid = 0.42273152
        self.assertEqual(type(analysis.output), np.ndarray)
        self.assertAlmostEqual(analysis.output.tolist()[0][0], valid)

    def test_run_default_same_tokens(self):
        text_tokens1 = {
            "1": ["apple", "pear", "banana", "mango", "kiwi"],
        }

        text_tokens2 = {
            "1": ["apple", "pear", "banana", "mango", "kiwi"],
        }

        analysis = SentenceSimilarityAnalysis(text_tokens1, text_tokens2)

        self.assertEqual(analysis.metric, "cosine")
        self.assertEqual(analysis.vector_model, "all-mpnet-base-V2")
        self.assertIsNone(analysis.vectors1)
        self.assertIsNone(analysis.vectors2)
        self.assertIsNone(analysis.output)

        analysis.run()

        valid = 1
        self.assertEqual(type(analysis.output), np.ndarray)
        self.assertAlmostEqual(analysis.output.tolist()[0][0], valid, 5)

    def test_run_default_multiple_sentences(self):
        text_tokens1 = {
            "1": ["apple", "pear", "banana", "mango", "kiwi"],
            "2": ["dog", "cat", "monkey", "rabbit", "koala"],
            "3": ["car", "plane", "submarine", "satellite", "train"],
        }

        text_tokens2 = {
            "1": ["apple", "pear", "banana", "mango", "kiwi"],
            "2": ["dog", "cat", "monkey", "rabbit", "koala"],
            "3": ["car", "plane", "submarine", "satellite", "train"],
        }

        analysis = SentenceSimilarityAnalysis(text_tokens1, text_tokens2)

        self.assertEqual(analysis.metric, "cosine")
        self.assertEqual(analysis.vector_model, "all-mpnet-base-V2")
        self.assertIsNone(analysis.vectors1)
        self.assertIsNone(analysis.vectors2)
        self.assertIsNone(analysis.output)

        analysis.run()

        # Checking the dimensions are correct/
        self.assertEqual(type(analysis.output), np.ndarray)
        self.assertEqual(analysis.output.shape, (3, 3))


if __name__ == "__main__":
    unittest.main()
