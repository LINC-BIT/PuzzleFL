from ClientTrain.utils.MultiModelDict import get_model_dict
import torch
from ClientTrain.AggModel.sixcnn import SixCNN
import torch.nn as nn
class LModel(nn.Module):
    def __init__(self, input,output):
        super().__init__()
        self.last = nn.Linear(input, output)

    def forward(self, x, t=-1, pre=False, is_cifar=True, avg_act=False):
        h,hidden = self.feature_net(x, avg_act)
        output = self.last(h)
        if is_cifar and t != -1:
            # make sure we predict classes within the current task
            if pre:
                offset1 = 0
                offset2 = int(t * self.nc_per_task)
            else:
                offset1 = int(t * self.nc_per_task)
                offset2 = int((t + 1) * self.nc_per_task)
            if offset1 > 0:
                output[:, :offset1].data.fill_(-10e10)
            if offset2 < self.outputsize:
                output[:, offset2:self.outputsize].data.fill_(-10e10)
        if avg_act is True:
            return output,hidden
        return output
# multi_model_dict = get_model_dict()

# for i, model in enumerate(multi_model_dict):
#     torch.save(model['model'].state_dict(), 'different_model/cv/'+str(i)+'.pth')
# net_glob = SixCNN([3,224,224],outputsize=100)
# torch.save(net_glob.state_dict(), 'different_model/cv/'+'kd.pth')
# from ClientTrainGNN.models.GCN import SimpleGCN
# hidden = 4
# for i in range(10):
#     hidden *= 2
#     model = SimpleGCN(1, hidden_dim = hidden, n_classes=8 * 10)
#     torch.save(model.state_dict(),'different_model/GNN/'+str(i)+'.pt')
lm = LModel(1000,100)
torch.save(lm.state_dict(),'different_model/md/2.pth')