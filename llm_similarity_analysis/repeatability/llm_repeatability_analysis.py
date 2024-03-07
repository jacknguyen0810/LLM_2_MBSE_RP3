import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from llm_similarity_analysis.preprocessing.text_preprocessing import RawData2Python
from llm_similarity_analysis.preprocessing.text_cleaning import TextCleaning
from llm_similarity_analysis.similarity.corpus_similarity import CorpusSimilarityAnalysis


class LLMRepeatabilityAnalysis:
    """A class to test the repeatability of LLM responses.
    """
    
    def __init__(self, data_folder_fp: str, validation_case_fp: str = None):
        # Data is a folder of .txt files that contain the raw LLM output
        self.data_folder_fp = data_folder_fp
        self.validation_case_fp = validation_case_fp
        
        # Empty dictionaries to hold the processed data
        self.clean_data = {}
        self.output_num = {}
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
            if filename.endswith('.txt'):
                txt_fp = os.path.join(self.data_folder_fp, filename)
                # Clean the data and add to a dictionary
                raw_output = preprocessor.txt_to_dict(txt_fp)
                # Filter out the terms text that aren't target text
                raw_output = {k: v for k, v in raw_output.items() if v[0] == '-'}
                cleaner = TextCleaning(text_dict=raw_output)
                cleaned_data = cleaner.clean()
                self.output_num[str(corpus_no)] = len(cleaned_data)
                self.clean_data[str(corpus_no)] = cleaned_data
                corpus_no += 1
        
        # Get validation case:
        # If a validation case is not given, use the first value
        if self.validation_case_fp is None:
            validation_data = self.clean_data['0']
        else:
            raw_output = preprocessor.txt_to_dict(self.validation_case_fp)
            cleaner = TextCleaning(raw_output)
            validation_data = cleaner.clean()
              
        # Perform corpus-to-corpus comparison against the validation data
        for corpus_no, corpus in self.clean_data.items():
            analysis = CorpusSimilarityAnalysis(validation_data, corpus, metric='cosine', vector_model='allenai-specter')
            analysis.run()
            self.similarities[str(corpus_no)] = analysis.output
            
    def stat_analysis(self, plot=True):
        # Looking at the number of outputs
        if plot:
            fig, ax = plt.subplots()
            sns.histplot(self.output_num.values(), stat='density', kde=True,)
            ax.set_ylabel('Number of Outputs')
            
    @property
    def mean_output_num(self):
        return np.mean(list(self.output_num.values()))
        
if __name__ == '__main__':
    input_fp = r'data\demo_sets\mini_PROVE_outputs'
    repeat = LLMRepeatabilityAnalysis(input_fp)
    repeat.run()
    repeat.stat_analysis()