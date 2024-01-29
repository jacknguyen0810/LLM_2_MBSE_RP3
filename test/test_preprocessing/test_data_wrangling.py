import unittest
from llm_similarity_analysis.preprocessing.text_cleaning import TextCleaning

class TestTextCleaning(unittest.TestCase):
    test_dict = {
        "1": "I can't understand what's going on + - & .",
        "2": "The following sentence: is GIBBerish 482 ;']'"
    }
    
    def test_expand_contractions(self):
        cleaner = TextCleaning(self.test_dict)
        sentence = self.test_dict["1"]
        expanded = cleaner.expand_contractions(sentence)
        test = "I can not understand what is going on + - % ."
        self.assertEqual(expanded, test)
        
        
if __name__ == "__main__":
    unittest.main()