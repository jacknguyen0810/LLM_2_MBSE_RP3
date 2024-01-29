import matplotlib.pyplot as plt
from gensim.models.word2vec import Word2Vec
from src.similarity_analysis.similarity_metrics import *


class CorpusSimilarityAnalysis:
    """Class for the similarity analysis using a range of similarity metrics"""

    def __init__(
        self,
        text_tokens1: dict,
        text_tokens2: dict,
        metric: str = None,
        vector_type: str = None,
    ) -> None:
        self.text_tokens1 = text_tokens1
        self.text_tokens2 = text_tokens2

        if metric is None:
            self.metric = "cosine"
        else:
            self.metric = metric
        if vector_type is None:
            self.vector_type = "cbow"
        else:
            self.vector_type = vector_type

        self.vectors1 = None
        self.vectors2 = None
        self.output = None

        self.similarity_metric_functions = {
            "cosine": cosine_sim,
            "rbf": rbf,
            "euclidean": euclidean_dist,
            "manhattan": manhattan_dist,
        }

    def run(self) -> None:
        # Combine the sentences into a text corpus
        self.text_tokens1 = self.combine_into_corpus(self.text_tokens1)
        self.text_tokens2 = self.combine_into_corpus(self.text_tokens2)
        # Turn the text token datasets into vectors using Word2Vec
        self.vectors1 = self.vectorise_dataset(self.text_tokens1)
        self.vectors2 = self.vectorise_dataset(self.text_tokens2)

        # Extract the similarity
        comp_metric = self.similarity_metric_functions.get(
            self.metric, self.metric_error
        )

        # Compare the two vectors using the specified similarity method
        self.output = comp_metric(self.vectors1, self.vectors2)

    def plot(
        self,
        title: str,
        xlabel: str = "Dataset 1 Sentence IDs",
        ylabel: str = "Dataset 2 Sentence IDs",
    ) -> None:
        # Plotting pairwise comparison of each of the requirements
        plt.imshow(self.output, "Greens")
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xticks(self.text_tokens1.keys())
        plt.yticks(self.text_tokens2.keys())

    @staticmethod
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

    @staticmethod
    def metric_error(*args) -> None:
        raise ValueError("Invalid Similarity Metric")

    def vectorise_dataset(self, tokens: dict) -> dict:
        vectors = {}
        if self.vector_type == "cbow":
            sg = 0
        elif self.vector_type == "skip gram":
            sg = 1
        else:
            raise ValueError("Not a valid vectorization method was chosen. ")

        for req_num, sentence in tokens.items():
            vector = Word2Vec(sentence, min_count=1, vector_size=100, window=5, sg=sg)
            vectors[req_num] = vector
        return vectors

    # TODO: Research appropriate vector size for full corpus
