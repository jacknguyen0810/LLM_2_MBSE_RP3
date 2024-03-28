from textwrap import wrap
from llm_similarity_analysis.similarity.sentence_similarity import SentenceSimilarityAnalysis
from llm_similarity_analysis.preprocessing.text_preprocessing import RawData2Python
from llm_similarity_analysis.preprocessing.text_cleaning import TextCleaning


def main():
    # Get the filepath to the .txt file containing the raw text data
    corpus_fp = r"data\demo_sets\imaging_functions_test.txt"

    # Create preprocessor object
    preprocessor = RawData2Python()

    # Convert the .txt file into a dictionary
    text_dict = preprocessor.txt_to_dict(corpus_fp)

    print(text_dict.values())
    label_1 = list(text_dict.values())
    label_1 = ['\n'.join(wrap(l, 50)) for l in label_1]

    # Clean the raw text data
    cleaner = TextCleaning(text_dict)
    clean_data = cleaner.clean()

    print(clean_data.values())

    # Perform similarity analysis
    analysis = SentenceSimilarityAnalysis(clean_data, clean_data)
    analysis.run()
    analysis.plot(title="Cosine Similarity", xticks=label_1, yticks=label_1)


if __name__ == "__main__":
    main()
