from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string
import contractions
import re


class TextCleaning:
    """
    This class class is to be used as a tool to clean raw LLM text output to prepare transformation into 
    machine readable vectors.
    """
    
    def __init__(self, filepath: str, output_path: str = None):
        """
        """
        self.filepath = filepath
        self.output_path = output_path
        self.raw_data = {}
        self.clean_text = {}
    
    def clean(self):
        """

        Returns:
            _type_: _description_
        """
        # Loop through the input dictionary and clean the text data
        for key, text in self.text_dict.items():
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
    
    def expand_contractions(self, text: str):
        expanded_words = []
        for word in text.split():
            expanded_words.append(contractions.fix(word))
        # Rejoin the sentence together
        expanded_text = " ".join(expanded_words)
        return expanded_text
  
    def remove_symbols(self, text: str):
        # Using regular expressions to remove no-ASCII characters
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        cleaned = text.translate(str.maketrans(" ", " ", string.punctuation))
        return cleaned

    def remove_stopwords(self, tokenized_text: list):
        """
        Function to remove stopwords from a tokenized list 

        Args:
            tokenized_text (list): Tokenized text data. 

        Returns:
            list: Input string with stopwords and punctuation removed. 
        """
        # Get list of english stop words
        stop_words = stopwords.words('english') + list(string.punctuation)      
        filtered_words = []   
        # Loop through tokenized text and remove stop words
        for word in tokenized_text:
            if word not in stop_words:
                filtered_words.append(word)
        return filtered_words