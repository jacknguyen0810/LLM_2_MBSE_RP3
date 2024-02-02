from sentence_transformers import SentenceTransformer


def vectorise_dataset(tokens: dict, model: str = "all-mpnet-base-v2") -> dict:
    """_summary_

    Args:
        tokens (dict): Dictionary containing lists of tokenized text data.
        model (str, optional): Sentence-BERT model name. Defaults to "all-mpnet-base-v2".
                    "all-mpnet-base-v2" is more general purpose.
                    "allenai-specter" trained on a scientific dataset. (https://arxiv.org/abs/2004.07180)
    Returns:
        dict: A dictionary containing the sentence and its equivalent text embedding/vector
    """
    transformer = SentenceTransformer(model)

    embeddings = {}

    for token in tokens.values():
        # Join the tokens together to create a string
        sentence = " ".join(token)
        embedding = transformer.encode(sentence)
        embeddings[sentence] = embedding

    return embeddings


def combine_into_corpus(tokens: dict) -> dict:
    """Combine a dictionary full of tokens into a dictionary with a single value, which is all of the tokens combined.

    Args:
        tokens (dict): Tokenized text data.

    Returns:
        dict: Single list containing all of the text tokens
    """
    # Combine all values from dictionary of text tokens into one corpus
    corpus_dict = {}  # Empty list to hold text corpus
    corpus = []
    for tokens in tokens.values():
        # Combine all of the tokens to make a text corpus
        for token in tokens:
            corpus.append(token)
    corpus_dict["Full Text Corpus"] = corpus
    return corpus_dict  # Make this the text token
