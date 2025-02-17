import numpy as np

# 随机生成一个N*M*M的反对称矩阵
def generate_antisymmetric_array(N, M):
    # 获取下三角（不含对角线）的索引
    rows, cols = np.tril_indices(M, -1)
    # 生成下三角随机数（正态分布，可替换为其他分布如np.random.uniform）
    lower_tri = np.random.randn(N, len(rows))
    # 初始化结果数组
    result = np.zeros((N, M, M))
    # 填充下三角和上三角
    result[:, rows, cols] = lower_tri
    result[:, cols, rows] = -lower_tri
    return result