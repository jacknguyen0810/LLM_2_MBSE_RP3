import matplotlib.pyplot as plt
import numpy as np
from llm_similarity_analysis.similarity import similarity_metrics
from llm_similarity_analysis.utilities.utility_functions import vectorise_dataset


class SentenceSimilarityAnalysis:
    """Class for the similarity analysis using a range of similarity metrics"""

    def __init__(
        self,
        text_tokens1: dict,
        text_tokens2: dict,
        metric: str = None,
        vector_model: str = None,
    ) -> None:
        self.text_tokens1 = text_tokens1
        self.text_tokens2 = text_tokens2

        if metric is None:
            self.metric = "cosine"
        else:
            self.metric = metric
        if vector_model is None:
            self.vector_model = "all-mpnet-base-V2"
        else:
            self.vector_model = vector_model

        self.vectors1 = None
        self.vectors2 = None
        self.output = None

        self.similarity_metric_functions = {
            "cosine": similarity_metrics.cosine_sim,
            "rbf": similarity_metrics.rbf,
            "euclidean": similarity_metrics.euclidean_dist,
            "manhattan": similarity_metrics.manhattan_dist,
        }

    def run(self) -> None:
        # Turn the text token datasets into vectors
        self.vectors1 = vectorise_dataset(self.text_tokens1, self.vector_model)
        self.vectors2 = vectorise_dataset(self.text_tokens2, self.vector_model)

        # Extract the similarity
        comp_metric = self.similarity_metric_functions.get(
            self.metric, self.metric_error
        )

        # Compare the two vectors using the specified similarity method
        self.output = comp_metric(
            list(self.vectors1.values()), list(self.vectors2.values())
        )

    def plot(
        self,
        title: str = None,
        xlabel: str = "Dataset 1 Sentence IDs",
        ylabel: str = "Dataset 2 Sentence IDs",
        xticks: list = None,
        yticks: list = None,
        fontsize: int = 6,
        text: bool = True,
        axes = None
    ) -> None:
        # Plotting pairwise comparison of each of the requirements
        if axes is not None:
            ax = plt.imshow(np.transpose(self.output), "Greens", ax=axes)
        else:
            ax = plt.imshow(np.transpose(self.output), "Greens")
        if title is not None:
            plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.colorbar()
        if text:
            for (i, j), label in np.ndenumerate(self.output):
                plt.text(i, j, round(label, 2), ha="center", va="center", fontsize=fontsize)

        if xticks is not None:
            plt.xticks(np.arange(0, len(xticks), 1), xticks, rotation='vertical', fontsize=fontsize)

        if yticks is not None:
            plt.yticks(np.arange(0, len(yticks), 1), yticks, fontsize=fontsize)

        if axes is None:
            plt.tight_layout()
            plt.show()

    @staticmethod
    def metric_error(*args) -> None:
        """Generic error for incorrect similarity metric

        Raises:
            ValueError: Generic error for incorrect similarity metric
        """
        raise ValueError("Invalid Similarity Metric")
    
    @property
    def average_sentence_sim(self) -> float:
        return np.average(self.output)
