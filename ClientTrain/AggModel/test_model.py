import torch
from torch import nn
import numpy as np
from ClientTrain.dataset.Cifar100 import Cifar100Task
from ClientTrain.AggModel.sixcnn import SixCNN
from ClientTrain.AggModel.resnet import Resnet18,WideResnet
from ClientTrain.AggModel.mobilenet import Mobilenetv2
from ClientTrain.AggModel.densenet import Densenet121
from ClientTrain.AggModel.vit import SixLayerViT
from ClientTrain.AggModel.vit import RepTailTinyPiT
import copy
def compute_offsets(task, nc_per_task, is_cifar=True):
    """
        Compute offsets for cifar to determine which
        outputs to select for a given task.
    """
    if is_cifar:
        offset1 = task * nc_per_task
        offset2 = (task + 1) * nc_per_task
    else:
        offset1 = 0
        offset2 = nc_per_task
    return offset1, offset2
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
            offset1, offset2 = compute_offsets(0, 10)
            output = model.forward(images, 0)[:, offset1:offset2]
            _, pred = output.max(1)
            hits = (pred == targets).float()
            # Log
            total_acc += hits.sum().data.cpu().numpy()
            total_num += len(images)

    return total_acc / total_num


def train_model(div_model,train_dataloader,test_dataloader):
    lrs = [0.0005]
    optmehtod = ['Adam']
    for optm in optmehtod:
        for lr in lrs:
            model = copy.deepcopy(div_model).cuda()
            if optm == 'Adam':
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            else:
                optimizer = torch.optim.SGD(model.parameters(), lr=lr)
            print('cur method: ' + optm + '     lr:' + str(lr))
            ce = torch.nn.CrossEntropyLoss()
            for epoch in range(10):
                model.train()
                for images, targets in train_dataloader:
                    images = images.cuda()
                    targets = targets.cuda()
                    offset1, offset2 = compute_offsets(0, 10)
                    outputs = model.forward(images, 0)[:, offset1:offset2]
                    loss = ce(outputs,targets)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                if epoch % 9 == 0:
                    print("acc:",eval(model,test_dataloader))


model_dict={}
model_dict['sixvit_model'] = SixLayerViT([3,32,32],outputsize=100) # Adam 0.0005 2
model_dict['pit_model'] = RepTailTinyPiT([3,32,32],outputsize=100)# Adam 0.001 2
model_dict['res18_model'] = Resnet18([3,32,32],outputsize=100) # Adam 0.0005/0.001 2
model_dict['wideres_model'] = WideResnet([3,32,32],outputsize=100) # Adam 0.001 1
model_dict['mobile_model'] = Mobilenetv2([3,32,32],outputsize=100)# Adam 0.001 2
# model_dict['alex_model'] = alexnet(num_classes=10)
model_dict['dense_model'] = Densenet121([3,32,32],outputsize=100)# Adam 0.0005 1

task = Cifar100Task('/data/lpyx/FedAgg/data/cifar100', task_num=10)

train, test = task.getTaskDataSet()
train_dataset = train[0]
test_dataset = test[0]
train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=256,
                                           shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                           batch_size=512,
                                           shuffle=False)

for key in model_dict.keys():
    print('test:',key)
    print('*'*100)
    train_model(model_dict[key], train_dataloader, test_dataloader)
# train_model(model_dict['pit_model'],train_dataloader,test_dataloader)