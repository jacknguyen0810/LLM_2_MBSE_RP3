import string
import contractions
import re
import json
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


class TextCleaning:
    """
    This class class is to be used as a tool to clean raw LLM text output to prepare transformation into
    machine readable vectors.
    """

    def __init__(self, text_dict: dict = None, json_fp: str = None):
        """ """
        self.raw_data = None
        self.clean_text = {}

        if text_dict is None and json_fp is None:
            raise ValueError(
                "Please input either a dictionary or a filepath to a .json file containing the uncleaned text data."
            )
        
        if isinstance(text_dict, dict) and json_fp is None:
            self.raw_data = text_dict
        elif isinstance(json_fp, str) and text_dict is None:
            self.raw_data = json.load(json_fp)
        else:
            raise ValueError("Incorrect combination of inputs received.")

    def clean(self):
        """

        Returns:
            _type_: _description_
        """
        if self.raw_data is None:
            raise ValueError("No text data loaded")
        # Loop through the input dictionary and clean the text data
        for key, text in self.raw_data.items():
            # Make the text lowercase
            text = text.lower()
            # Expand the contractions in the text
            text = self.remove_symbols(text)
            # Remove symbols and ascii characters
            text = self.remove_symbols(text)
            # Tokenize the text data
            tokenized = word_tokenize(text)
            # Lemmatize the words in the sentence
            wnl = WordNetLemmatizer()
            lemmatized = []
            for words in tokenized:
                lemmatized.append(wnl.lemmatize(words))
            # Remove stopwords
            cleaned_text = self.remove_stopwords(lemmatized)
            self.clean_text[key] = cleaned_text
        return self.clean_text

    @staticmethod
    def expand_contractions(text: str):
        expanded_words = []
        for word in text.split():
            expanded_words.append(contractions.fix(word))
        # Rejoin the sentence together
        expanded_text = " ".join(expanded_words)
        return expanded_text

    @staticmethod
    def remove_symbols(text: str):
        # Using regular expressions to remove no-ASCII characters
        text = re.sub(r"[^\x00-\x7F]+", "", text)
        cleaned = text.translate(str.maketrans(" ", " ", string.punctuation))
        return cleaned

    @staticmethod
    def remove_stopwords(tokenized_text: list):
        """
        Function to remove stopwords from a tokenized list

        Args:
            tokenized_text (list): Tokenized text data.

        Returns:
            list: Input string with stopwords and punctuation removed.
        """
        # Get list of english stop words
        stop_words = stopwords.words("english") + list(string.punctuation)
        filtered_words = []
        # Loop through tokenized text and remove stop words
        for word in tokenized_text:
            if word not in stop_words:
                filtered_words.append(word)
        return filtered_words
    
    
if __name__ == '__main__':
    test_dict = {
        "1": "I can't understand what's going on + - & .",
        "2": "The following sentence: is GIBBerish 482 ;']'"
    }
    
    cleaner = TextCleaning(text_dict=test_dict)
    expanded = cleaner.expand_contractions(test_dict["1"])
    no_symbols = cleaner.remove_symbols(test_dict["1"])
    no_stop = cleaner.remove_stopwords(['why', 'does', 'Jack', 'the', 'dog', 'understand', 'English'])
    print(expanded)
    print(no_symbols)
    print(no_stop)
