import json
from pathlib import Path
import os


class TextPreprocessing:
    
    def __init__(self, filepath: str, output: str):
        """
        A file to turn a .txt file into a useable json for data.

        Args:
        :filename : str: The path to the .txt file that contains the data that needs to be processed. 
        """
        self.filepath = filepath
        self.output = output
        return
    
    def turn_full_text_to_json(self, filename):
        """

        Args:
            filename (str): Filename of the .txt file to convert. 
            output_path (str): Output path for the json file. 

        Returns:
            _type_: The text data is in a dictionary, (for our project, it is each of the sentences.)
        """
        data = {}
        with open(self.filepath) as fh:
            for number, line in enumerate(fh):
                data[number] = line.strip()
                
        out_path = os.path.join(self.output, filename)
        out_file = open(out_path, 'w')
        json.dump(data, out_file, indent = 4, sort_keys = False)
                
        return data
    
        
    
    

if __name__ == "__main__":
    filename = r"C:\Users\Jack\OneDrive\Documents\Python Scripts\LLM_2_MBSE_RP3\data\test_data.txt"
    output = r"C:\Users\Jack\OneDrive\Documents\Python Scripts\LLM_2_MBSE_RP3\data"
    preprocessing = TextPreprocessing(filename, output)
    preprocessing.turn_full_text_to_json(r"test_data.json")
    
    