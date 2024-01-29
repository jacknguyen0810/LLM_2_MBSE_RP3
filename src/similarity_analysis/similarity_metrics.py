from sklearn.metrics.pairwise import (
    cosine_similarity,
    euclidean_distances,
    manhattan_distances,
    rbf_kernel,
)


def cosine_sim(data1, data2):
    return cosine_similarity(data1, data2)


def rbf(data1, data2):
    return rbf_kernel(data1, data2)


def euclidean_dist(data1, data2):
    return euclidean_distances(data1, data2)


def manhattan_dist(data1, data2):
    return manhattan_distances(data1, data2)
