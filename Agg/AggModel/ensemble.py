import torch
import torch.nn as nn
# TODO Max, Avg, Majority vote(hard max [1,0,0,0])
import torch.nn.functional as F

def average(outputs):
    """Compute the average over a list of tensors with the same size."""
    return sum(outputs) / len(outputs)

class AvgEnsemble(nn.Module):
    def __init__(self, net_list):
        super(AvgEnsemble, self).__init__()
        self.estimators = nn.ModuleList(net_list)

    def forward(self, x,t):
        outputs = [
            F.softmax(estimator(x,t), dim=1) for estimator in self.estimators
        ]
        proba = average(outputs)

        return proba