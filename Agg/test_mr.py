import copy

from torch import nn
import numpy as np
import torch

from ClientTrain.dataset.Cifar100 import Cifar100Task


def compute_conv_output_size(Lin, kernel_size, stride=1, padding=0, dilation=1):
    return int(np.floor((Lin + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1))


class ModelPrune():

    def __init__(self,prune_instructions=0.95, prunable_types=(nn.Conv2d, nn.Linear),device =None):

        self.prune_instructions = prune_instructions
        self.prunable_types = prunable_types
        self.device = device
        self.PATH = None
        self.prun_epoch = 5
        self.tune_epoch = 10
        self.current_task = 0
        self.mask = {}  # 3-dimensions: task (list), layer (dict), parameter mask (tensor)

    def prune(self, model, prune_quantile):
        # Calculate Quantil
        all_prunable = torch.tensor([]).to(self.device)
        for name, param_layer in model.named_parameters():
            if 'bias' not in name:
                prev_mask = torch.zeros(param_layer.size(), dtype=torch.bool, requires_grad=False).to(self.device)
                p = param_layer.masked_select(~prev_mask)
                if p is not None:
                    all_prunable = torch.cat((all_prunable.view(-1), p), -1)
        B = torch.abs(all_prunable.cpu()).detach().numpy()
        cutoff = np.quantile(B, q=prune_quantile)
        with torch.no_grad():
            for name, param_layer in model.named_parameters():
                if 'bias' not in name:
                    curr_mask = torch.abs(param_layer).ge(cutoff)  # q
                    param_layer *= curr_mask
                    self.mask[name] = curr_mask

    def fine_tune_mask(self, model):
        mask_idx = 0
        for name, param_layer in model.named_parameters():
            if 'bias' not in name and  param_layer.grad is not None:
                param_layer.grad *= self.mask[name]
                mask_idx += 1


    def fix_biases(self, model):

        for name, param_layer in model.named_parameters():
            if 'bias' in name:
                param_layer.requires_grad = False

    def fix_batch_norm(self, model):

        for mod in model.modules():
            if isinstance(mod, nn.BatchNorm2d):
                mod.affine = False
                for param_layer in mod.parameters():
                    param_layer.requires_grad = False

    def apply_eval_mask(self, model):
        with torch.no_grad():
            for name, param_layer in model.named_parameters():
                if 'bias' not in name:

                    # get indices of all weights from previous masks
                    prev_mask = torch.zeros(param_layer.size(), dtype=torch.bool, requires_grad=False).to(self.device)
                    prev_mask |= self.mask[name]

                    # zero out all weights that are not in the mask for this task
                    param_layer *= prev_mask

    def on_after_backward(self, pl_module):
        self.fine_tune_mask(pl_module)

    def on_epoch_end(self, pl_module):
        self.prune(model=pl_module,
                    prune_quantile=self.prune_instructions)

    def decode_model(self,model, state_dict):
        for name, parameter in model.named_parameters():
            cur_all_data = torch.flatten(parameter.data)
            cur_position = state_dict[name]['position']
            big_weight = state_dict[name]['weight']
            new_weight = torch.zeros(cur_all_data.shape)
            new_weight.scatter_(0, cur_position, big_weight)
            parameter.data = new_weight.view(parameter.data.shape)

    def encode_model(self,prune_model):
        pos_weight = {}
        for name, parameter in prune_model.named_parameters():
            pos_weight[name] = {}
            d = torch.flatten(parameter.data)
            cur_position = torch.nonzero(d)
            cur_position = torch.squeeze(cur_position)
            print(cur_position.shape)
            big_weight = torch.index_select(d, 0, cur_position)
            print(big_weight.shape)
            pos_weight[name]['position'] = torch.LongTensor(cur_position)
            pos_weight[name]['weight'] = big_weight
        encode_model_dict = prune_model.state_dict()

        for key in encode_model_dict.keys():
            encode_model_dict[key] = pos_weight[key]
        return encode_model_dict

class SixCNN(nn.Module):
    def __init__(self, inputsize, outputsize=100,nc_per_task = 10):
        super().__init__()
        self.outputsize = outputsize
        ncha, size, _ = inputsize
        self.conv1 = nn.Conv2d(ncha, 32, kernel_size=3, padding=1, bias=False)
        s = compute_conv_output_size(size, 3, padding=1)  # 32
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        s = compute_conv_output_size(s, 3, padding=1)  # 32
        s = s // 2  # 16
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        s = compute_conv_output_size(s, 3, padding=1)  # 16
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        s = compute_conv_output_size(s, 3, padding=1)  # 16
        s = s // 2  # 8
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        s = compute_conv_output_size(s, 3, padding=1)  # 8
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        s = compute_conv_output_size(s, 3, padding=1)  # 8
        #         self.conv7 = nn.Conv2d(128,128,kernel_size=3,padding=1)
        #         s = compute_conv_output_size(s,3, padding=1) # 8
        s = s // 2  # 4
        self.fc1 = nn.Linear(s * s * 128, 1024, bias=False)  # 2048
        self.drop1 = nn.Dropout(0.25)
        self.drop2 = nn.Dropout(0.5)
        self.MaxPool = torch.nn.MaxPool2d(2)
        self.avg_neg = []
        # self.fc2 = nn.Linear(256, 100)
        self.relu = torch.nn.ReLU()
        self.last = nn.Linear(1024, outputsize, bias=False)
        self.nc_per_task = nc_per_task
    def forward(self, x, t=-1, pre=False, is_cifar=True, avg_act=False):
        if x.size(1) != 3:
            bsz = x.size(0)
            x = x.view(bsz, 3, 32, 32)
        act1 = self.relu(self.conv1(x))
        act2 = self.relu(self.conv2(act1))
        h = self.drop1(self.MaxPool(act2))
        act3 = self.relu(self.conv3(h))
        act4 = self.relu(self.conv4(act3))
        h = self.drop1(self.MaxPool(act4))
        act5 = self.relu(self.conv5(h))
        act6 = self.relu(self.conv6(act5))
        h = self.drop1(self.MaxPool(act6))
        h = h.view(x.shape[0], -1)
        act7 = self.relu(self.fc1(h))
        h = self.drop2(act7)
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
        return output
def eval(model,dataloaders):
    total_loss = 0
    total_acc = 0
    total_num = 0
    model = model.cuda()
    # Loop batches
    model.eval()
    with torch.no_grad():
        for images, targets in dataloaders:
            images = images.cuda()
            targets = targets.cuda()
            # Forward
            output = model.forward(images,t=-1)
            _, pred = output.max(1)
            hits = (pred == targets).float()
            # Log
            total_acc += hits.sum().data.cpu().numpy()
            total_num += len(images)

    return total_acc / total_num



# model = SixCNN([3,32,32],10).cuda()
# for parameter in model.parameters():
#     a = torch.flatten(parameter.data)
#     cur_shape = parameter.data.shape
#     b = a.view(cur_shape)
#     print(parameter.data.shape)
#     print(a.shape)
#     print(b.shape)
#     print("*"*100)
task = Cifar100Task('/data/lpyx/FedAgg/data/cifar100', task_num=10)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0006)
train, test = task.getTaskDataSet()
train_dataset = train[0]
test_dataset = test[0]
train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=32,
                                           shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                           batch_size=32,
                                           shuffle=False)
ce = torch.nn.CrossEntropyLoss()
# torch.save(model.state_dict(),'test_mr_model/30model')
# for epoch in range(30):
#     model.train()
#     for images, targets in train_dataloader:
#         images = images.cuda()
#         targets = targets.cuda()
#         outputs = model(images, is_cifar=False)
#         loss = ce(outputs,targets)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#     print("acc:",eval(model,test_dataloader))
# torch.save(model.state_dict(),'test_mr_model/30model')
# model.load_state_dict(torch.load('test_mr_model/30model.pt'))
# device = torch.device('cuda:0')
# pack = PackNet(device = device)
# print("acc:",eval(model,test_dataloader))
# pack.on_epoch_end(model)
# print("acc:",eval(model,test_dataloader))
# for epoch in range(10):
#     model.train()
#     for images, targets in train_dataloader:
#         images = images.cuda()
#         targets = targets.cuda()
#         outputs = model(images, is_cifar=False)
#         loss = ce(outputs,targets)
#         optimizer.zero_grad()
#         loss.backward()
#         pack.on_after_backward(model)
#         optimizer.step()
#     print("acc:",eval(model,test_dataloader))
# pack.apply_eval_mask(model)
# print("last acc:",eval(model,test_dataloader))
# torch.save(model.state_dict(),'test_mr_model/30model_prune.pt')

# print(a)
# 119698
# 2393952
# encode('test_mr_model/30model_prune.pt')
# state_dict = torch.load('test_mr_model/30model_prune_encode.pt')
# new_model = SixCNN([3,32,32],10)
# decode(new_model,state_dict,test_dataloader)
# print(eval(new_model, test_dataloader))
# a = [1,2,3,4,5]
# b = copy.deepcopy(a)
# b.append(6)
# print(a)
# model = SixCNN([3,32,32],10).cuda()
# a = model.state_dict()
# for name,parameter in model.named_parameters():
#     parameter.data = parameter.data * 0
# b = model.state_dict()
# print(a)
# print(b)


from scipy import io
import os
import shutil


def move_valimg(val_dir='./val', devkit_dir='./ILSVRC2012_devkit_t12'):
    """
    move valimg to correspongding folders.
    val_id(start from 1) -> ILSVRC_ID(start from 1) -> WIND
    organize like:
    /val
       /n01440764
           images
       /n01443537
           images
        .....
    """
    # load synset, val ground truth and val images list
    synset = io.loadmat(os.path.join(devkit_dir, 'data', 'meta.mat'))

    ground_truth = open(os.path.join(devkit_dir, 'data', 'ILSVRC2012_validation_ground_truth.txt'))
    lines = ground_truth.readlines()
    labels = [int(line[:-1]) for line in lines]

    root, _, filenames = next(os.walk(val_dir))
    for filename in filenames:
        # val image name -> ILSVRC ID -> WIND
        val_id = int(filename.split('.')[0].split('_')[-1])
        ILSVRC_ID = labels[val_id - 1]
        WIND = synset['synsets'][ILSVRC_ID - 1][0][1][0]
        print("val_id:%d, ILSVRC_ID:%d, WIND:%s" % (val_id, ILSVRC_ID, WIND))

        # move val images
        output_dir = os.path.join(root, WIND)
        if os.path.isdir(output_dir):
            pass
        else:
            os.mkdir(output_dir)
        shutil.move(os.path.join(root, filename), os.path.join(output_dir, filename))


if __name__ == '__main__':
    move_valimg()
