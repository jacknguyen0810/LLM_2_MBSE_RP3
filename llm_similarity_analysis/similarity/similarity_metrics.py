from sklearn.metrics.pairwise import (
    cosine_similarity,
    euclidean_distances,
    manhattan_distances,
    rbf_kernel,
)


def cosine_sim(data1: list, data2: list):
    return cosine_similarity(data1, data2)


def rbf(data1: list, data2: list):
    return rbf_kernel(data1, data2)


def euclidean_dist(data1: list, data2: list):
    return euclidean_distances(data1, data2)


def manhattan_dist(data1: list, data2: list):
    return manhattan_distances(data1, data2)
