from textwrap import wrap
from llm_similarity_analysis.similarity.corpus_similarity import (
    CorpusSimilarityAnalysis,
)
from llm_similarity_analysis.similarity.sentence_similarity import (
    SentenceSimilarityAnalysis,
)
from llm_similarity_analysis.preprocessing.text_preprocessing import RawData2Python
from llm_similarity_analysis.preprocessing.text_cleaning import TextCleaning


def main():
    # Process the validation dataset
    valid_fp = r"data\validation_data\JWST_components.txt"
    # Create preprocessor object and turn txt file to dict
    preprocessor = RawData2Python()
    valid_dict = preprocessor.txt_to_dict(valid_fp)
    valid_ticks = list(valid_dict.values())
    valid_ticks = ["\n".join(wrap(l, 50)) for l in valid_ticks]
    # Clean data
    cleaner = TextCleaning(text_dict=valid_dict)
    clean_valid = cleaner.clean()

    # Process the generated dataset
    gen_fp = (
        r"data\JWST_outputs\JWST_system_architectures\JWST_system_architecture_0.txt"
    )
    # Create preprocessor object and turn txt file to dict
    preprocessor = RawData2Python()
    gen_dict = preprocessor.txt_to_dict(gen_fp)
    gen_ticks = list(gen_dict.values())
    gen_ticks = ["\n".join(wrap(l, 100)) for l in gen_ticks]
    # Clean data
    cleaner = TextCleaning(text_dict=gen_dict)
    clean_gen = cleaner.clean()

    # Sentence Similarity Analysis
    sentence = SentenceSimilarityAnalysis(clean_valid, clean_gen)
    sentence.run()
    sentence.plot(
        "JWST System Architecture Similarity using Cosine Similarity",
        "Validation Components",
        "Generated Components",
        None,
        None,
        5,
        False,
    )
    print(f"\nThe average function similarity is {sentence.average_sentence_sim}")

    # Corpus Similarity Analysis
    corpus = CorpusSimilarityAnalysis(clean_valid, clean_gen)
    corpus.run()
    print(f"\nThe overall corpus-to-corpus similarity is {corpus.output_value}")

    # Perform for each subsystem
    subs_valid = [
        r"data\validation_data\JWST_comms.txt",
        r"data\validation_data\JWST_acs.txt",
        r"data\validation_data\JWST_cdh.txt",
        r"data\validation_data\JWST_electrical.txt",
        r"data\validation_data\JWST_isim.txt",
        r"data\validation_data\JWST_ote.txt",
        r"data\validation_data\JWST_sunshield.txt",
        r"data\validation_data\JWST_prop.txt",
    ]

    subs_gen = [
        r"data\JWST_outputs\JWST_subsystems\JWST_comms_output.txt",
        r"data\JWST_outputs\JWST_subsystems\JWST_acs_output.txt",
        r"data\JWST_outputs\JWST_subsystems\JWST_cdg_output.txt",
        r"data\JWST_outputs\JWST_subsystems\JWST_eps_output.txt",
        r"data\JWST_outputs\JWST_subsystems\JWST_isim_output.txt",
        r"data\JWST_outputs\JWST_subsystems\JWST_ote_output.txt",
        r"data\JWST_outputs\JWST_subsystems\JWST_sun_output.txt",
        r"data\JWST_outputs\JWST_subsystems\JWST_thermal_output.txt",
    ]

    sub_names = [
        "Communications Subsystem",
        "Attitude Control Subsystem",
        "Control and Data Handling Subsystem",
        "Electrical Power Subsystem",
        "Integrated Science Instrument Module Subsystem",
        "Optical Telescope Subsystem",
        "Sunshield Subsystem",
        "Propulsion/Thermal Subsystem",
    ]

    for num, sub in enumerate(subs_valid):
        # Process the validation dataset
        valid_fp = sub
        # Create preprocessor object and turn txt file to dict
        preprocessor = RawData2Python()
        valid_dict = preprocessor.txt_to_dict(valid_fp)
        valid_ticks = list(valid_dict.values())
        valid_ticks = ["\n".join(wrap(l, 50)) for l in valid_ticks]
        # Clean data
        cleaner = TextCleaning(text_dict=valid_dict)
        clean_valid = cleaner.clean()

        # Process the generated dataset
        gen_fp = subs_gen[num]
        # Create preprocessor object and turn txt file to dict
        preprocessor = RawData2Python()
        gen_dict = preprocessor.txt_to_dict(gen_fp)
        gen_ticks = list(gen_dict.values())
        gen_ticks = ["\n".join(wrap(l, 50)) for l in gen_ticks]
        # Clean data
        cleaner = TextCleaning(text_dict=gen_dict)
        clean_gen = cleaner.clean()

        # Sentence Similarity Analysis
        sentence_sub = SentenceSimilarityAnalysis(clean_valid, clean_gen)
        sentence_sub.run()
        sentence_sub.plot(
            title=sub_names[num],
            xlabel="Validation Components",
            ylabel="Generated Components",
            xticks=valid_ticks,
            yticks=gen_ticks,
            fontsize=10,
            text=True,
            axes=None,
        )
        print(
            f"\nThe average function similarity is {sentence_sub.average_sentence_sim}"
        )

        # Corpus Similarity Analysis
        corpus = CorpusSimilarityAnalysis(clean_valid, clean_gen)
        corpus.run()
        print(f"\nThe overall corpus-to-corpus similarity is {corpus.output_value}")


if __name__ == "__main__":
    main()
