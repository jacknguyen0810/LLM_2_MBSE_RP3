import nltk

nltk.download("punkt")
nltk.download("wordnet")
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string
import contractions
import re


class LLMDataWrangling:
    """
    This class class is to be used as a tool to clean raw LLM text output to prepare transformation into
    machine readable vectors.
    """

    def __init__(self, text_dict: dict) -> None:
        """
        Inputs:
        :param raw_text: dict: Dictionary containing the raw text data .
        :param vector_type: str: "sentence" (default), is to split and vectorise the text by sentence.
                                 "corpus", is to vectorise the whole corpus of text as one.

        :output dict: Dictionary containing the cleaned text output, vectors, and ID number.
        """
        self.text_dict = text_dict
        self.clean_text = {}
        return

    def clean_function(self):
        """
        Main function to fully process raw text data into cleaned, tokenised data stored within a dictionary

        Returns:
            dict: Contains all of the tokenised, cleaned text data.
        """
        print("Starting cleaning")

        # Loop through the input dictionary and clean the text data
        for key, text in self.text_dict.items():
            # Make the text lowercase
            text = text.lower()

            # Expand the contractions in the text
            text = self.remove_symbols(text)

            # Remove symbols and ascii characters
            text = self.remove_symbols(text)

            # Tokenise the text data
            tokenised = word_tokenize(text)

            # Lemmatise the words in the sentence
            wnl = WordNetLemmatizer()
            lemmatised = []
            for words in tokenised:
                lemmatised.append(wnl.lemmatize(words))

            # Remove stopwords
            cleaned_text = self.remove_stopwords(lemmatised)

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
        text = re.sub(r"[^\x00-\x7F]+", "", text)
        cleaned = text.translate(str.maketrans(" ", " ", string.punctuation))
        return cleaned

    def remove_stopwords(self, tokenised_text: list):
        """
        Function to remove stopwords from a tokenised list

        Args:
            tokenised_text (list): Tokenised text data.

        Returns:
            list: Input string with stopwords and punctuation removed.
        """
        # Get list of english stop words
        stop_words = stopwords.words("english") + list(string.punctuation)

        filtered_words = []

        # Loop through tokenised text and remove stop words
        for word in tokenised_text:
            if word not in stop_words:
                filtered_words.append(word)

        return filtered_words


if __name__ == "__main__":
    example = {
        "1": "Hi my Name is Jack, and I am working on LLMs",
        "2": "I am currently testing the data cleaning pipeline",
    }

    cleaner = LLMDataWrangling(example)
    cleaner.clean_function()
    print(cleaner.clean_text)
