import torch
from sklearn.cluster import KMeans
from torch import nn, optim

from cluster_interface import ClusterInterface
import numpy as np


class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)  # latent_dim 是K维特征的维度
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid()  # 输出重建的矩阵
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded


def preprocess_data(data):
    # 展平每个 M*M 的矩阵，得到 N 个 M^2 的特征向量
    n, m, _ = data.shape
    flattened_data = data.reshape(n, m * m)
    return flattened_data


def train_autoencoder(data, input_dim, latent_dim, learning_rate, epochs, device):
    # 预处理数据
    flattened_data = preprocess_data(data)
    flattened_data = torch.Tensor(flattened_data).to(device)

    model = Autoencoder(input_dim, latent_dim).to(device)
    criterion = nn.MSELoss()  # 使用均方误差损失函数
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    encoded = None

    for epoch in range(epochs):
        optimizer.zero_grad()
        # 输入数据
        reconstructed, encoded = model(flattened_data)
        loss = criterion(reconstructed, flattened_data)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Self Encoder Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}")

    return encoded.detach().cpu().numpy()


class SelfEncoderCluster(ClusterInterface):
    def __init__(self, num_clusters=5, random_state=42, num_features=10, learning_rate=0.01, num_epochs=100,
                 idx=None):
        super().__init__()
        self.name = "SelfEncoderCluster"
        self.num_clusters = num_clusters
        self.random_state = random_state
        self.num_features = num_features
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.idx = idx

    def cluster(self, user_pair_matrix_in: np.ndarray) -> (np.ndarray, np.ndarray):
        n, m, _ = user_pair_matrix_in.shape
        features = train_autoencoder(user_pair_matrix_in, m * m, self.num_features, self.learning_rate, self.num_epochs,
                                     self.device)
        km = KMeans(n_clusters=self.num_clusters, random_state=self.random_state)
        labels = km.fit_predict(features)
        return labels, features

    def get_name(self) -> str:
        return self.name

    def get_num_clusters(self) -> int:
        return self.num_clusters
