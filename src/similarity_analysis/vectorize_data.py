"""import numpy for array processing"""

import numpy as np

"""matplotlib.pyplot for plotting"""
import matplotlib.pyplot as plt  #

"""gensim.word2vec for feature extraction"""
from gensim.models.word2vec import Word2Vec

"""sklearn.metrics.pairwise for the comparison of the """
from sklearn.metrics.pairwise import (
    cosine_similarity,
    euclidean_distances,
    manhattan_distances,
    rbf_kernel,
)


class SentenceSimilarityAnalysis:
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
            "cosine": self.cosine_sim,
            "rbf": self.rbf,
            "euclidean": self.euclidean_dist,
            "manhattan": self.manhattan_distances,
        }

    def run(self) -> None:
        # Turn the text token datasets into vectors using Word2Vec
        self.vectors1 = self.vectorise_dataset(self.text_tokens1)
        self.vectors2 = self.vectorise_dataset(self.text_tokens2)

        # Extract the similarity
        comp_metric = self.similarity_metric_functions.get(
            self.metric, self.metric_error
        )

        # Compare the two vectors using the specified similarity method
        self.output = comp_metric(self.vectors1, self.vectors2)

    def plot(self, title: str, xlabel: str, ylabel: str) -> None:
        # Plotting pairwise comparison of each of the requirements
        plt.imshow(self.output)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

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

    @staticmethod
    def cosine_sim(data1, data2):
        return cosine_similarity(data1, data2)

    @staticmethod
    def rbf(data1, data2):
        return rbf_kernel(data1, data2)

    @staticmethod
    def euclidean_dist(data1, data2):
        return euclidean_distances(data1, data2)

    @staticmethod
    def manhattan_distances(data1, data2):
        return manhattan_distances(data1, data2)
