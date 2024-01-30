import unittest
from llm_similarity_analysis.similarity_analysis.sentence_similarity import SentenceSimilarityAnalysis


class TestSentenceSimilarityAnalysis(unittest.TestCase):
    
    def test_run_default(self):
        text_tokens1 = {
            '1': ['apple', 'pear', 'banana', 'mango', 'kiwi']
        }
        
        text_tokens2 = {
            '1': ['dog', 'cat', 'monkey', 'rabbit', 'koala']
        }
        
        analysis = SentenceSimilarityAnalysis(text_tokens1, text_tokens2)
        
        self.assertEqual(analysis.metric, 'cosine')
        self.assertEqual(analysis.vector_type, 'sg')
        self.assertEqual(analysis.vector_size, 100)
        self.assertIsNone(analysis.vectors1)
        self.assertIsNone(analysis.vectors2)
        self.assertIsNone(analysis.output)

        analysis.run()
        
        print(analysis.vectors1)
        print(analysis.vectors2)
        
        result = analysis.output
        valid = [[0]]
        self.assertEqual(result, valid)
        

if __name__ == "__main__":
    unittest.main()
        