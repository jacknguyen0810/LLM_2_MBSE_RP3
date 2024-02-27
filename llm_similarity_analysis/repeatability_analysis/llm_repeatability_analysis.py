import os
from llm_similarity_analysis.preprocessing.text_preprocessing import RawData2Python
from llm_similarity_analysis.preprocessing.text_cleaning import TextCleaning
from llm_similarity_analysis.similarity_analysis.corpus_similarity import CorpusSimilarityAnalysis


class LLMRepeatabilityAnalysis:
    """A class to test the repeatability of LLM responses.
    """
    
    def __init__(self, data_folder_fp: str, validation_case_fp: str = None):
        # Data is a list of .txt files that contain the raw LLM output
        self.data_folder_fp = data_folder_fp
        self.validation_case_fp = validation_case_fp
        
        # Empty dictionaries to hold the processed data
        self.clean_data = {}
        self.number_of_requirements = {}
        self.similarities = {}
        
    
    def run(self):
        # Access the directory with all of the required .txt. files
        directory = os.fsencode(self.data_folder_fp)
        # Counter variable to count the data set being used
        corpus_no = 0
        preprocessor = RawData2Python()
        # Loop through the files in the directory
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            # Check for .txt files
            if filename.endswith('.txt') is False:
                continue
            else:
                # Clean the data and add to a dictionary
                raw_output = preprocessor.txt_to_dict(file)
                # Filter out the terms text that aren't requirements
                raw_output = {k: v for k, v in raw_output.items() if v[0:4] == ' '}
                cleaner = TextCleaning(text_dict=raw_output)
                cleaned_data = cleaner.clean()
                self.number_of_requirements[str(corpus_no)] = len(cleaned_data)
                self.clean_data[str(corpus_no)] = cleaned_data
        
        # Get validation case:
        if self.validation_case_fp is None:
            validation_data = self.clean_data['0']
        else:
            raw_output = preprocessor.txt_to_dict(self.validation_case_fp)
            cleaner = TextCleaning(raw_output)
            validation_data = cleaner.clean()
              
        # Perform corpus-to-corpus comparison against the validation data
        for corpus_no, corpus in self.clean_data.values():
            analysis = CorpusSimilarityAnalysis(validation_data, list(corpus.values()[str(corpus_no)]))
            analysis.run()
            self.similarities[str(corpus_no)] = analysis.output()
            
        
            
