from typing import Dict, List

from sklearn.metrics import silhouette_score

from cluster_interface import ClusterInterface
from methods.KmeansCluster import KmeansCluster
import pandas as pd
import numpy as np

DataDict = Dict[str, int]


def get_max_num(data_dict: DataDict) -> int:
    return max(data_dict.values()) + 1


def get_user_name(idx_in: int) -> str:
    return "User_" + str(idx_in).zfill(3)


def get_sample_name(idx_in: int) -> str:
    return "Sample_" + str(idx_in).zfill(3)


# Build User-Pair Matrix
def build_user_pair_matrix_from_raw_data(df_in):
    # all usernames & all sample names
    user_names = df_in['User'].unique()
    sample_names = pd.concat([df_in['Compair_01'], df_in['Compair_02']]).unique()
    # e.g. {'User_001': 1,'User_002':2,...}
    user_index_in = {name: int(name.split('_')[1]) for name in user_names}
    # e.g. {'Sample_001': 1,'Sample_002':2,...}
    sample_index_in = {name: int(name.split('_')[1]) for name in sample_names}
    user_pair_matrix_in = np.zeros(
        (get_max_num(user_index_in), get_max_num(sample_index_in), get_max_num(sample_index_in)))
    for _, row in df_in.iterrows():
        user_idx = user_index_in[row['User']]
        sample_left_idx = sample_index_in[row['Compair_01']]
        sample_right_idx = sample_index_in[row['Compair_02']]
        user_pair_matrix_in[user_idx, sample_left_idx, sample_right_idx] = row['Result']
    return user_index_in, sample_index_in, user_pair_matrix_in


# data path (no need to modify)
raw_data_path = './data/input/raw_result_0209.csv'
output_path_prefix = './data/output/'
raw_data_df = pd.read_csv(raw_data_path)
print(f"Load {raw_data_df.shape[0]} records.")
user_index, sample_index, user_pair_matrix = build_user_pair_matrix_from_raw_data(raw_data_df)
max_user_cnt, max_sample_cnt = get_max_num(user_index) - 1, get_max_num(sample_index) - 1
print(f'Load {max_user_cnt} users and {max_sample_cnt} samples.')
# all cluster methods
cluster_methods: List[ClusterInterface] = [KmeansCluster(num_clusters=5, random_state=42),
                                           KmeansCluster(num_clusters=7, random_state=42), ]

for i, method in enumerate(cluster_methods):
    print(f"{i + 1}.Start {method.get_name()} clustering, target cluster count: {method.get_num_clusters()}")
    cluster_result, feature_matrix = method.cluster(user_pair_matrix)
    cluster_result_2d = np.array([cluster_result[cluster_result == i] for i in range(method.get_num_clusters())],
                                 dtype=object)
    for idx, cluster in enumerate(cluster_result_2d):
        print(f'Cluster {idx + 1} has {len(cluster)} users.')
    sil_score = silhouette_score(feature_matrix, cluster_result)
    print(f'Silhouette Score: {sil_score}')
