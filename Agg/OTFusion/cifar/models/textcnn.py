import torch
import torch.nn as nn
import torch.nn.functional as F
class Noemb_TextCNN(nn.Module):
    def __init__(self):
        super(Noemb_TextCNN, self).__init__()

        embedding_dimension = 100
        class_num = 4
        chanel_num = 1
        filter_num = 1
        filter_sizes =[3,4,5]
        self.convs = nn.ModuleList(
            [nn.Conv2d(chanel_num, filter_num, (size, embedding_dimension),bias=False) for size in filter_sizes])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(len(filter_sizes) * filter_num, class_num,bias=False)

    def forward(self, x,task=0):
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits