from textwrap import wrap
from llm_similarity_analysis.similarity.corpus_similarity import CorpusSimilarityAnalysis
from llm_similarity_analysis.similarity.sentence_similarity import SentenceSimilarityAnalysis
from llm_similarity_analysis.preprocessing.text_preprocessing import RawData2Python
from llm_similarity_analysis.preprocessing.text_cleaning import TextCleaning

def main():
    # Process the validation dataset
    valid_fp = r'data\validation_data\PROVE_components.txt'
    # Create preprocessor object and turn txt file to dict
    preprocessor = RawData2Python()
    valid_dict = preprocessor.txt_to_dict(valid_fp)
    valid_ticks = list(valid_dict.values())
    valid_ticks = ['\n'.join(wrap(l, 25)) for l in valid_ticks]
    # Clean data
    cleaner = TextCleaning(text_dict=valid_dict)
    clean_valid = cleaner.clean()
    
    # Process the generated dataset
    gen_fp = r'data\PROVE_outputs\PROVE_components\PROVE_components_0.txt'
    # Create preprocessor object and turn txt file to dict
    preprocessor = RawData2Python()
    gen_dict = preprocessor.txt_to_dict(gen_fp)
    gen_ticks = list(gen_dict.values())
    gen_ticks = ['\n'.join(wrap(l, 100)) for l in gen_ticks]
    # Clean data
    cleaner = TextCleaning(text_dict=gen_dict)
    clean_gen = cleaner.clean()
    
    # Sentence Similarity Analysis
    sentence = SentenceSimilarityAnalysis(clean_valid, clean_gen)
    sentence.run()
    sentence.plot("PROVE Component Similarity using Cosine Similarity", 'Validation Components', 'Generated Components', None, None, 8)
    print(f'\nThe average function similarity is {sentence.average_sentence_sim}')
    
    # Corpus Similarity Analysis
    corpus = CorpusSimilarityAnalysis(clean_valid, clean_gen)
    corpus.run()
    print(f'\nThe overall corpus-to-corpus similarity is {corpus.output_value}')
if __name__ == '__main__':
    main()