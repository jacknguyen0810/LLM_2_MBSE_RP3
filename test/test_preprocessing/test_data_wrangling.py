import unittest
from llm_similarity_analysis.preprocessing.text_cleaning import TextCleaning

class test_TextCleaning(unittest.TestCase):
    
    def test_expand_contractions(self):
        test_dict = {
            "1": "I can't understand what's going on + - & .",
            "2": "The following sentence: is GIBBerish 482 ;']'"
        }
        cleaner = TextCleaning(text_dict=test_dict)
        expanded = cleaner.expand_contractions(test_dict["1"])
        test = "I cannot understand what is going on + - % ."
        self.assertEqual(expanded, test)
        
        
if __name__ == "__main__":
    unittest.main()