import unittest
from llm_similarity_analysis.preprocessing.text_cleaning import TextCleaning

class test_TextCleaning(unittest.TestCase):
    
    def test_expand_contractions(self):
        test_dict = {
            "1": "I can't understand what's going on + - & .",
            "2": "The following sentence: is GIBBerish 482 ;']'"
        }
        cleaner = TextCleaning(text_dict = test_dict)
        expanded = cleaner.expand_contractions(test_dict["1"])
        test = "I cannot understand what is going on + - & ."
        self.assertEqual(expanded, test)
        
    def test_remove_symbols(self):
        test_dict = {
            "1": "I can't understand what's going on + - & .",
            "2": "The following sentence: is GIBBerish 482 ;']'"
        }
        cleaner = TextCleaning(text_dict = test_dict)
        no_symbols = cleaner.remove_symbols(test_dict["1"])
        test = "I cant understand whats going on    "
        self.assertEqual(no_symbols, test)
        
    def test_remove_stopwords(self):
        test_dict = {
            "1": "I can't understand what's going on + - & .",
            "2": "The following sentence: is GIBBerish 482 ;']'"
        }
        cleaner = TextCleaning(text_dict = test_dict)
        no_stop = cleaner.remove_stopwords(['why', 'does', 'Jack', 'the', 'dog', 'understand', 'English'])
        test = ['Jack', 'dog', 'understand', 'English']
        self.assertEqual(no_stop, test)
        
    def test_lemmatize_text(self):
        test_dict = {
            "1": "I can't understand what's going on + - & .",
            "2": "The following sentence: is GIBBerish 482 ;']'"
        }
        cleaner = TextCleaning(text_dict = test_dict)
        lemmatized = cleaner.lemmatize_text('following understood went Babylon')
        test = "follow understood go Babylon"
        self.assertEqual(lemmatized, test)
        
    def test_clean(self):
       test_dict = {
            "1": "I can't understand what's going on + - & .",
            "2": "The following sentence: is GIBBerish 482 ;']'"
        }
       cleaner = TextCleaning(test_dict)
       cleaner.clean()
       test = {
           "1": ["I", "understand", "go"],
           "2": ["follow", "sentence", "gibberish", "482"]
       }
       self.assertDictEqual(test, cleaner.clean_text)
           
        
if __name__ == "__main__":
    unittest.main()