# Abstract Cluster class
from abc import abstractmethod, ABC

import numpy as np


class ClusterInterface(ABC):
    def __init__(self):
        self.idx = None

    @abstractmethod
    # input:[user,sample_left,sample_right]->result(-1 for left,1 for right)
    # output:[user] -> group_id ; [user] -> features
    def cluster(self, user_pair_matrix_in: np.ndarray) -> (np.ndarray,np.ndarray):
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def get_num_clusters(self) -> int:
        pass
