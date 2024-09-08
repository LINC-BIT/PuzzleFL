import dgl
from dgl.nn.pytorch import GraphConv
import torch.nn as nn
import torch.nn.functional as F
class GCN(nn.Module):
    def __init__(self, in_dim=1, hidden_dim=8, n_classes=8):
        super(GCN, self).__init__()
        self.gcn1 = GraphConv(in_dim, hidden_dim,bias=False)  # 定义第一层图卷积
        self.gcn2 = GraphConv(hidden_dim, hidden_dim,bias=False)  # 定义第二层图卷积
        self.last = nn.Linear(hidden_dim, n_classes,bias=False)  # 定义分类器

    def forward(self, g, task=0):
        """g表示批处理后的大图，N表示大图的所有节点数量，n表示图的数量
        """
        # 我们用节点的度作为初始节点特征。对于无向图，入度 = 出度
        h = g.in_degrees().view(-1, 1).float()  # [N, 1]
        # 执行图卷积和激活函数
        h = F.relu(self.gcn1(g, h))  # [N, hidden_dim]
        h = F.relu(self.gcn2(g, h))  # [N, hidden_dim]
        g.ndata['h'] = h  # 将特征赋予到图的节点
        # 通过平均池化每个节点的表示得到图表示
        hg = dgl.mean_nodes(g, 'h')  # [n, hidden_dim]
        return self.last(hg)  # [n, n_classes]