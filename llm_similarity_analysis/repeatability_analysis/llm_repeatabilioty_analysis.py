import os
from llm_similarity_analysis.preprocessing.text_preprocessing import RawData2Python
from llm_similarity_analysis.preprocessing.text_cleaning import TextCleaning
from llm_similarity_analysis.similarity_analysis.sentence_similarity import SentenceSimilarityAnalysis


class LLMRepeatabilityAnalysis:
    """A class to test the repeatability of LLM responses.
    """
    
    def __init__(self, data_folder_fp: str, validation_case_fp: str):
        # Data is a list of .txt files that contain the raw LLM output
        self.data_folder_fp = data_folder_fp
        
        # Empty dictionary to hold the processed data
        self.clean_data = {}
        self.number_of_requirements = {}
        
    
    def run(self):
        # Access the directory with all of the required .txt. files
        directory = os.fsencode(self.data_folder_fp)
        # Counter variable to count the data set being used
        corpus_no = 0
        
        # Loop through the files in the directory
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            # Check for .txt files
            if filename.endswith('.txt') is False:
                continue
            else:
                # Clean the data and add to a dictionary
                preprocessor = RawData2Python()
                raw_output = preprocessor.txt_to_dict(file)
                # Filter out the terms text that aren't requirements
                raw_output = {k: v for k, v in raw_output.items() if v[0] == '-'}
                cleaner = TextCleaning(text_dict=raw_output)
                cleaned_data = cleaner.clean()
                self.number_of_requirements[str(corpus_no)] = len(cleaned_data)
                self.clean_data[str(corpus_no)] = cleaned_data
                
        # Perform corpus-to-corpus comparison
    
        
                
                
        