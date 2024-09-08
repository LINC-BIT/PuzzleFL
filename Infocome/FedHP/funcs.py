import torch
from torch import nn
import numpy as np


def compute_consensus_distance(local_idx, local_model, neighbor_models, neighbor_cids, cids):
    """计算id为cid的本地模型与邻居模型之间的共识距离: 即欧式距离
    
    Args:
        local_model (nn.Module): 本地模型
        neighbor_models (List[nn.Module]): 邻居模型列表
        neighbor_cids (List[int]): 邻居模型的id列表
        cids (List): 所有worker的cid
        
    Returns:
        D_h_i_j (List(float)): 共识距离
    """
    def model_to_vector(model: nn.Module):
        """将模型参数转化为向量"""
        return torch.cat([param.data.view(-1) for param in model.parameters()])
    
    local_vector = model_to_vector(local_model)
    
    D_h_i_j = [np.inf for _ in cids]
    D_h_i_j[local_idx] = 0
    
    for neighbor_model, neighbor_cid in zip(neighbor_models, neighbor_cids):
        neighbor_vector = model_to_vector(neighbor_model)
        distance = torch.norm(local_vector - neighbor_vector).item()
        D_h_i_j[neighbor_cid] = distance
    
    return D_h_i_j


def compose_consensus_matrix(D_h_i_j_list, cids):
    """根据每个worker的邻域共识距离构建所有worker的共识距离矩阵D_h
    
    Args:
        D_h_i_j_list (List[int]): 包含所有worker的D_i_j列表的列表，是一个二维数组
        Ah: 邻接矩阵, 表示worker之间的连接关系
        cids (List): 所有worker的cid
    
    Returns:
        D_h (np.ndarray): 包含所有worker之间的共识距离的矩阵， 形状为len(cids)*len(cids)
    """
    # 把D_h_i_j_list转换为np.array类型的D_h_nei，这个矩阵是一个二维矩阵, 每个元素 D_h_nei[i][j]表示worker i和worker j之间的共识距离；由于存在没有直接连接的worker对，所以D_h_nei中存在0元素
    D_h_nei = np.array(D_h_i_j_list) 
    
    # 计算出每个worker之间的最短距离矩阵
    shortest_dist_matrix = floyd_warshall(D_h_nei)

    # 基于邻接矩阵D_h_nei和最短距离矩阵shortest_dist_matrix构建全局的共识距离矩阵D_h, 即将D_h_nei中的0元素替换为最短距离矩阵中的对应元素
    for i in range(len(cids)):
        for j in range(len(cids)):
            if D_h_nei[i][j] == np.inf:
                D_h_nei[i][j] = shortest_dist_matrix[i][j]
    
    D_h = D_h_nei
    return D_h
    
    
    
def floyd_warshall(graph):
    """
    使用Floyd-Warshall算法计算所有工作者对之间的最短路径。
    参数:
    graph: 二维数组，表示工作者之间的直接共识距离，其中graph[i][j]表示工作者i和工作者j之间的距离。
    返回:
    dist: 矩阵，包含所有工作者对之间的最短路径距离。
    """
    n = len(graph)
    dist = np.array(graph)

    # 计算通过第三个工作者的最短路径
    for k in range(n):
        for i in range(n):
            for j in range(n):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

    return dist


def update_topology_by_D_h(D_h, cids, neighbor_num):
    """根据D_h矩阵，构建worker之间的邻接矩阵Ah，即确定worker之间的连接关系。
    其中neighbor_num用来决定与每个worker的邻居worker的数量，比如第i个worker的邻接worker的数量应该为neighbor_num个。
    更新后的Ah需要满足：
    1. Ah是一个完整的连通图。
    2. 每个worker只与neighbor_num个worker相连。
    3. 每个worker的邻接worker之间的距离都是其所能相连的worker中距离前beibor_num个最小的。（根据D_h中的距离可以进行排序计算）

    Args:
        D_h (np.ndarray): 包含所有worker之间的共识距离的二维矩阵
        cids (List[int]): 所有worker的cid
        neighbor_num (int): 每个worker的邻居worker数量
        
    Returns:
        Ah (np.ndarray): 邻接矩阵，形状为len(cids)*len(cids)
    """
    num_workers = len(cids)
    Ah = np.zeros((num_workers, num_workers))
    
    # Step 1: Use Kruskal's algorithm to find the minimum spanning tree (MST)
    edges = []
    for i in range(num_workers):
        for j in range(i + 1, num_workers):
            if D_h[i][j] > 0:
                edges.append((D_h[i][j], i, j))
    
    # Sort edges based on distance
    edges.sort()
    
    parent = list(range(num_workers))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        rootX = find(x)
        rootY = find(y)
        if rootX != rootY:
            parent[rootY] = rootX
    
    # Kruskal's algorithm to form MST
    for distance, i, j in edges:
        if find(i) != find(j):
            union(i, j)
            Ah[i][j] = 1
            Ah[j][i] = 1
    
    # Step 2: Ensure each worker has exactly neighbor_num connections
    for i in range(num_workers):
        distances = [(D_h[i][j], j) for j in range(num_workers) if i != j and Ah[i][j] == 0]
        distances.sort()
        current_neighbors = sum(Ah[i])
        
        for distance, j in distances:
            if current_neighbors >= neighbor_num:
                break
            Ah[i][j] = 1
            Ah[j][i] = 1
            current_neighbors += 1

    return Ah


import networkx as nx

def generate_adj_matrix(num_workers, num_neighbors):
    while True:
        # 生成随机图
        G = nx.random_regular_graph(num_neighbors, num_workers)
        
        # 确保图是连通的
        if nx.is_connected(G):
            break

    adj_matrix = np.eye(num_workers, dtype=int)  # 初始化对角线为1的邻接矩阵
    
    for edge in G.edges():
        i, j = edge
        adj_matrix[i, j] = 1
        adj_matrix[j, i] = 1

    return adj_matrix


class Coordinator():
    def __init__(self, user_num, neibor_num) -> None:
        self.dh = None
        self.cids = [i for i in range(user_num)]
        self.neibor_num = neibor_num
    
    def set_dh(self, dh):
        self.dh = dh
    
    def get_topology(self):
        if self.dh is not None:
            Ah = update_topology_by_D_h(self.dh, self.cids, self.neibor_num)
        else:
            Ah = generate_adj_matrix(len(self.cids), self.neibor_num)   # 第一轮聚合直接使用随机聚合
        return Ah
    
    def get_neibors(self, idx, Ah):
        nei_ids = np.where(Ah[idx] == 1)[0].tolist()    # nei_ids: List
        if len(nei_ids) > self.neibor_num:
            nei_ids.remove(idx)
        return nei_ids
        
        