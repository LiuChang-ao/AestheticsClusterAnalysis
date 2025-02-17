import numpy as np
from sklearn.cluster import KMeans

from cluster_interface import ClusterInterface


def build_feature_matrix(user_pair_matrix_in: np.ndarray) -> np.ndarray:
    mask_neg = (user_pair_matrix_in == -1)
    mask_pos = (user_pair_matrix_in == 1)

    sum_j_neg = mask_neg.sum(axis=2)
    sum_j_pos = mask_pos.sum(axis=2)

    sum_k_neg = mask_neg.sum(axis=1)
    sum_k_pos = mask_pos.sum(axis=1)

    feature_matrix = (sum_j_neg - sum_j_pos) + (sum_k_pos - sum_k_neg)
    return feature_matrix


class KmeansCluster(ClusterInterface):
    def __init__(self, num_clusters=5, random_state=42):
        super().__init__()
        self.name = "KmeansCluster"
        self.num_clusters = num_clusters
        self.random_state = random_state

    def cluster(self, user_pair_matrix_in: np.ndarray) -> (np.ndarray, np.ndarray):
        km = KMeans(n_clusters=self.num_clusters, random_state=self.random_state)
        feature_matrix = build_feature_matrix(user_pair_matrix_in)
        labels = km.fit_predict(feature_matrix)
        return labels, feature_matrix

    def get_name(self) -> str:
        return self.name

    def get_num_clusters(self) -> int:
        return self.num_clusters
