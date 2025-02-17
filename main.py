import csv
import os
from typing import Dict, List, Tuple
import torch
from sklearn.metrics import silhouette_score

from cluster_interface import ClusterInterface
from methods.kmeans_cluster import KmeansCluster
import pandas as pd
import numpy as np
import torch.optim as optim

from bt_model import BTModel
from methods.nmf_cluster import NMFCluster

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


def aggregate_cluster_pairs(clusters_in: np.ndarray, user_pair_matrix_in: np.ndarray) -> Tuple[
    np.ndarray, Dict[int, int]]:
    # Collect all unique users and filter out invalid ones (<=0 or out of bounds)
    unique_users = np.unique(clusters_in)
    samples = set()
    n_users = user_pair_matrix_in.shape[0]

    # Collect all samples that have been rated by any user in the cluster
    for u in unique_users:
        if u < 1 or u >= n_users:
            continue  # Skip invalid user IDs
        user_matrix = user_pair_matrix_in[u]
        # Find all non-zero entries where i and j are valid sample IDs (>=1)
        rows, cols = np.nonzero(user_matrix)
        for i, j in zip(rows, cols):
            if i >= 1 and j >= 1:
                samples.add(i)
                samples.add(j)

    if not samples:
        return np.zeros((0, 0), dtype=int), {}

    # Create mapping from old sample ID to new ID (sorted)
    sorted_samples = sorted(samples)
    old_to_new = {old_in: idx_in for idx_in, old_in in enumerate(sorted_samples)}
    k = len(sorted_samples)
    new_matrix = np.zeros((k, k), dtype=int)

    # Aggregate the scores into the new matrix
    for u in unique_users:
        if u < 1 or u >= n_users:
            continue
        user_matrix = user_pair_matrix_in[u]
        rows, cols = np.nonzero(user_matrix)
        for i, j in zip(rows, cols):
            if i >= 1 and j >= 1:
                # Check if the sample is in the mapping (should always be true)
                if i in old_to_new and j in old_to_new:
                    x = old_to_new[i]
                    y = old_to_new[j]
                    new_matrix[x, y] -= user_matrix[i, j]
                    new_matrix[y, x] += user_matrix[i, j]

    return new_matrix, old_to_new


def generate_valid_pairs(valid_matrix: np.ndarray) -> List[Tuple[int, int, float]]:
    result = []
    for ii in range(len(valid_matrix)):
        for jj in range(ii + 1, len(valid_matrix)):
            if valid_matrix[ii, jj] < 0:
                ## ii win jj for valid_matrix[ii,jj] rounds
                result.append((ii, jj, float(-valid_matrix[ii, jj])))
            elif valid_matrix[ii, jj] > 0:
                result.append((jj, ii, float(valid_matrix[ii, jj])))
    return result


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Calculate strength values by MLE for each clusters
# Output: {sample_idx:strength_value} (only sample in input cluster will be calculated)
def calculate_strength_values(clusters_in: np.ndarray, user_pair_matrix_in: np.ndarray) -> Dict[int, float]:
    valid_matrix, old_to_new = aggregate_cluster_pairs(clusters_in, user_pair_matrix_in)
    num_valid_samples = len(valid_matrix)
    model = BTModel(num_valid_samples).to(device)
    n_iterations = 30
    valid_pairs = generate_valid_pairs(valid_matrix)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    for iteration in range(n_iterations):
        tot_loss = 0
        for data in valid_pairs:
            id_i, id_j, label = data
            id_i = torch.tensor(id_i).to(device)
            id_j = torch.tensor(id_j).to(device)
            label = torch.tensor(label, dtype=torch.float32).to(device)
            optimizer.zero_grad()
            pred = model.forward_sigmoid(id_i, id_j)
            loss = model.loss(pred, torch.tensor(label, dtype=torch.float32))
            loss.backward()
            optimizer.step()
            tot_loss += loss.item()
        if iteration % 5 == 0: print(f"Epoch {iteration}, Loss: {tot_loss}")

    strength_array = model.reward.detach().cpu().numpy()
    result = {}
    for key, val in old_to_new.items():
        result[key] = strength_array[val]
    return result


# data path (no need to modify)
raw_data_path = './data/input/raw_result_0209.csv'
output_path_prefix = './data/output/'
raw_data_df = pd.read_csv(raw_data_path)
print(f"Load {raw_data_df.shape[0]} records.")
user_index, sample_index, user_pair_matrix = build_user_pair_matrix_from_raw_data(raw_data_df)
max_user_cnt, max_sample_cnt = get_max_num(user_index) - 1, get_max_num(sample_index) - 1
print(f'Load {max_user_cnt} users and {max_sample_cnt} samples.')
# all cluster methods
rand_seed = 42
cluster_methods: List[ClusterInterface] = [
    # KmeansCluster(num_clusters=5, random_state=42),
    # KmeansCluster(num_clusters=7, random_state=42),
    NMFCluster(num_clusters=5, random_state=rand_seed, num_features=12),
]

for i, method in enumerate(cluster_methods):
    print(f"{i + 1}.Start {method.get_name()} clustering, target cluster count: {method.get_num_clusters()}")
    cluster_result, feature_matrix = method.cluster(user_pair_matrix)
    cluster_result_2d = [np.where(cluster_result == s)[0] for s in range(method.get_num_clusters())]
    for idx, cluster in enumerate(cluster_result_2d):
        print(f'Cluster {idx + 1} has {len(cluster)} users.')
    sil_score = silhouette_score(feature_matrix, cluster_result)
    print(f'Silhouette Score: {sil_score}')
    for cluster_idx, cluster in enumerate(cluster_result_2d):
        file_name = f'Method_{i + 1}_{method.get_name()}_Cluster_{cluster_idx}.csv'
        file_path = os.path.join(output_path_prefix, file_name)
        strength_values = calculate_strength_values(cluster, user_pair_matrix)
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            for x, y in strength_values.items():
                writer.writerow([f"Sample_{x:03}", f"{y:.6f}"])
        print(f'Save {file_name}')
