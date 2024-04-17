import unittest
from llm_similarity_analysis.preprocessing.text_preprocessing import RawData2Python


class TestRawData2Python(unittest.TestCase):

    def test_txt_to_dict(self):
        txt2json = RawData2Python()
        fp = r"test\test_data\test_txt.txt"
        filename = r"test_data"
        output_path = r"test\test_data"
        text_dict = txt2json.txt_to_dict(fp, False, filename, output_path)
        test_dict = {
            "1": "This is the first line.",
            "2": "This is the second line.",
            "3": "This is the third line.",
            "4": "This is the fourth line.",
            "5": "This is the fifth line.",
        }

        self.assertDictEqual(test_dict, text_dict)


if __name__ == "__main__":
    unittest.main()
