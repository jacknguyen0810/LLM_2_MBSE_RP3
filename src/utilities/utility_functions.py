from gensim.models.word2vec import Word2Vec


def vectorise_dataset(
    tokens: dict,
    vector_type: str = "cbow",
    vector_size: int = 100,
    min_count: int = 1,
    window: int = 5,
) -> dict:
    vectors = {}
    if vector_type == "cbow":
        sg = 0
    elif vector_type == "skip gram":
        sg = 1
    else:
        raise ValueError("Not a valid vectorization method was chosen. ")

    for req_num, sentence in tokens.items():
        vector = Word2Vec(
            sentence, min_count=1, vector_size=vector_size, window=5, sg=sg
        )
        vectors[req_num] = vector
    return vectors


def combine_into_corpus(tokens: dict):
    # Combine all values from dictionary of text tokens into one corpus
    corpus_dict = {}  # Empty list to hold text corpus
    corpus = []
    for tokens in tokens.values():
        # Combine all of the tokens to make a text corpus
        for token in tokens:
            corpus.append(token)
    corpus_dict["Full Text Corpus"] = corpus
    return corpus_dict  # Make this the text token
