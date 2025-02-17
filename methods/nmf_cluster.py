import numpy as np
import torch
from scipy.sparse import vstack
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF

from cluster_interface import ClusterInterface


class NMFCluster(ClusterInterface):
    def __init__(self, num_clusters=5, random_state=42, num_features=10):
        super().__init__()
        self.name = "NMFCluster"
        self.num_clusters = num_clusters
        self.random_state = random_state
        self.num_features = num_features
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def cluster(self, user_pair_matrix_in: np.ndarray) -> (np.ndarray, np.ndarray):
        # 将每个用户的矩阵展平为一维向量
        processed_data = []
        for m in user_pair_matrix_in:
            # 替换-1为2，并转换为浮点类型
            processed = np.where(m == -1, 2, m).astype(np.float32)
            # 展平矩阵为一维数组
            flattened = processed.flatten()
            processed_data.append(flattened)

        # 转换为二维数组，形状为(num_users, flattened_dim)
        data_matrix = np.vstack(processed_data)

        # 应用NMF进行特征提取
        nmf = NMF(
            n_components=self.num_features,
            random_state=self.random_state,
            solver='cd',
            init='nndsvd'
        )
        user_features = nmf.fit_transform(data_matrix)

        # 应用KMeans聚类
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=self.random_state)
        labels = kmeans.fit_predict(user_features)

        return labels, user_features

    def get_name(self) -> str:
        return self.name

    def get_num_clusters(self) -> int:
        return self.num_clusters
