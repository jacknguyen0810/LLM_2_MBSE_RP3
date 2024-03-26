import json
import os
import textwrap


class RawData2Python:
    """
    Class that contains functions for turning raw text data into python code.
    """

    def __init__(self):
        """
        Initialise the class to hold the ra and output data.
        """
        self.output_dict = None

    def txt_to_dict(
        self,
        filepath: str,
        output_to_json: bool = False,
        filename: str = None,
        output_path: str = None,
    ):
        """Turns .txt file into a dictionary, and output to a .json file if output = True

        Args:
            filepath (str): Filepath to .txt file.
            output (bool, optional): If True, output the data into a json file, with corresponding filename and output path. Defaults to False
            filename (str, optional): Name of output file. Defaults to None.
            output_path (str, optional): If not None, then will output a .json file to specified output path. Defaults to None.

        Returns:
            _type_: _description_
        """
        # Empty dictionary to hold text data
        data = {}
        # Open the file
        with open(filepath, encoding="utf-8") as fh:
            # Loop through the lines in the .txt file
            for number, line in enumerate(fh):
                # Check for empty lines of code
                if line in ['\n', '\r\n']:
                    continue
                # If the line has text
                else:
                    # Unindent the string
                    unindent = textwrap.dedent(line)
                    # Add the line to the dictionary
                    data[str(number + 1)] = unindent.strip()

        # Output to .json if requested.
        if output_to_json:
            out_path = os.path.join(output_path, filename)
            out_file = open(out_path, "w", encoding="utf-8")
            json.dump(data, out_file, indent=4, sort_keys=False)

        return data


if __name__ == "__main__":
    name = r"C:\Users\Jack\OneDrive\Documents\Python Scripts\LLM_2_MBSE_RP3\data\test_data.txt"
    output_fp = r"C:\Users\Jack\OneDrive\Documents\Python Scripts\LLM_2_MBSE_RP3\data"
    preprocessing = RawData2Python()
    preprocessing.txt_to_dict(name, r"test_data.json", output_fp)
