import json
import os


class RawData2Python:
    """
    Class that contains functions for turning raw text data into python code. 
    """
    def __init__(self):
        """
        Initialise the class to hold the ra and output data.
        """
        self.output_dict = None     
     
    def txt_to_dict(self, filepath: str, filename: str = None, output_path: str = None):
        """Turns .txt file into a dictionary, and output to a 

        Args:
            filepath (str): Filepath to .txt file.
            filename (str, optional): Name of output file. Defaults to None.
            output_path (str, optional): If not None, then will output a .json file to specified output path. Defaults to None.

        Returns:
            _type_: _description_
        """
        data = {}
        with open(filepath, encoding="utf-8") as fh:
            for number, line in enumerate(fh):
                data[number] = line.strip()         
        out_path = os.path.join(output_path, filename)
        out_file = open(out_path, 'w', encoding="utf-8")
        json.dump(data, out_file, indent = 4, sort_keys = False)    
        return data 

if __name__ == "__main__":
    name = r"C:\Users\Jack\OneDrive\Documents\Python Scripts\LLM_2_MBSE_RP3\data\test_data.txt"
    output = r"C:\Users\Jack\OneDrive\Documents\Python Scripts\LLM_2_MBSE_RP3\data"
    preprocessing = RawData2Python()
    preprocessing.txt_to_dict(name, r"test_data.json", output)
    