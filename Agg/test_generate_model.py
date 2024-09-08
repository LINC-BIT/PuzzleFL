import os


import torchvision
import dgl

from dgl.nn.pytorch import GraphConv

from Agg.FedDag import FedDag
from Agg.OTFusion import utils, parameters, wasserstein_ensemble

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchtext.vocab import Vectors
import torchtext.data as data
from torch.autograd import Variable

# Taken from https://github.com/kuangliu/pytorch-cifar
class our_filed(data.Field):
    def __init__(self, sequential=True, use_vocab=True, init_token=None,
                 eos_token=None, fix_length=None, dtype=torch.long,
                 preprocessing=None, postprocessing=None, lower=False,
                 tokenize=None, tokenizer_language='en', include_lengths=False,
                 batch_first=False, pad_token="<pad>", unk_token="<unk>",
                 pad_first=False, truncate_first=False, stop_words=None,
                 is_target=False):
        super(our_filed,self).__init__(sequential=sequential, use_vocab=use_vocab, init_token=init_token,
                 eos_token=eos_token, fix_length=fix_length, dtype=dtype,
                 preprocessing=preprocessing, postprocessing=postprocessing, lower=lower,
                 tokenize=tokenize, tokenizer_language=tokenizer_language, include_lengths=include_lengths,
                 batch_first=batch_first, pad_token=pad_token, unk_token=unk_token,
                 pad_first=pad_first, truncate_first=truncate_first, stop_words=stop_words,
                 is_target=is_target)

    def pad(self, minibatch):
        minibatch = list(minibatch)
        if not self.sequential:
            return minibatch
        if self.fix_length is None:
            max_len = max(len(x) for x in minibatch)
        else:
            max_len = self.fix_length + (
                self.init_token, self.eos_token).count(None) - 2
        max_len = max(max_len, 10)
        padded, lengths = [], []
        for x in minibatch:
            if self.pad_first:
                padded.append(
                    [self.pad_token] * max(0, max_len - len(x))
                    + ([] if self.init_token is None else [self.init_token])
                    + list(x[-max_len:] if self.truncate_first else x[:max_len])
                    + ([] if self.eos_token is None else [self.eos_token]))
            else:
                padded.append(
                    ([] if self.init_token is None else [self.init_token])
                    + list(x[-max_len:] if self.truncate_first else x[:max_len])
                    + ([] if self.eos_token is None else [self.eos_token])
                    + [self.pad_token] * max(0, max_len - len(x)))
            lengths.append(len(padded[-1]) - max(0, max_len - len(x)))
        if self.include_lengths:
            return (padded, lengths)
        return padded

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, use_batchnorm=True):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if not use_batchnorm:
            self.bn1 = self.bn2 = nn.Sequential()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes) if use_batchnorm else nn.Sequential()
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, use_batchnorm=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        if not use_batchnorm:
            self.bn1 = self.bn2 = self.bn3 = nn.Sequential()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes) if use_batchnorm else nn.Sequential()
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, use_batchnorm=True, linear_bias=True):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.use_batchnorm = use_batchnorm
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64) if use_batchnorm else nn.Sequential()
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes, bias=linear_bias)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.use_batchnorm))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, task=-1):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(num_classes=10, use_batchnorm=True, linear_bias=True):
    return ResNet(BasicBlock, [2,2,2,2], num_classes=num_classes, use_batchnorm=use_batchnorm, linear_bias=linear_bias)

def ResNet34(num_classes=10, use_batchnorm=True, linear_bias=True):
    return ResNet(BasicBlock, [3,4,6,3], num_classes=num_classes, use_batchnorm=use_batchnorm, linear_bias=linear_bias)

def ResNet50(num_classes=10, use_batchnorm=True, linear_bias=True):
    return ResNet(Bottleneck, [3,4,6,3], num_classes=num_classes, use_batchnorm=use_batchnorm, linear_bias=linear_bias)

def ResNet101(num_classes=10, use_batchnorm=True, linear_bias=True):
    return ResNet(Bottleneck, [3,4,23,3], num_classes=num_classes, use_batchnorm=use_batchnorm, linear_bias=linear_bias)

def ResNet152(num_classes=10, use_batchnorm=True, linear_bias=True):
    return ResNet(Bottleneck, [3,8,36,3], num_classes=num_classes, use_batchnorm=use_batchnorm, linear_bias=linear_bias)
def compute_conv_output_size(Lin, kernel_size, stride=1, padding=0, dilation=1):
    return int(np.floor((Lin + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1))


class Cifar100Net(nn.Module):
    def __init__(self, inputsize):
        super().__init__()

        ncha, size, _ = inputsize
        self.conv1 = nn.Conv2d(ncha, 32, kernel_size=3, padding=1,bias=False)
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
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1,bias=False)
        s = compute_conv_output_size(s, 3, padding=1)  # 8
        #         self.conv7 = nn.Conv2d(128,128,kernel_size=3,padding=1)
        #         s = compute_conv_output_size(s,3, padding=1) # 8
        s = s // 2  # 4
        self.fc1 = nn.Linear(s*s*128, 1024,bias=False)  # 2048
        self.drop1 = nn.Dropout(0.25)
        self.drop2 = nn.Dropout(0.5)
        self.MaxPool = torch.nn.MaxPool2d(2)
        self.avg_neg = []
        # self.fc2 = nn.Linear(256, 100)
        self.relu = torch.nn.ReLU()
        self.fc2 = nn.Linear(1024,10,bias=False)
    def forward(self, x, avg_act=False):
        if x.size(1) !=3:
            bsz = x.size(0)
            x = x.view(bsz,3,32,32)
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
        h = self.fc2(h)
        # h = self.fc2(h)
        self.grads = {}
        # def save_grad(name):
        #     def hook(grad):
        #         self.grads[name] = grad
        #         return hook
        # if avg_act == True:
        #     names = [0, 1, 2, 3, 4, 5, 6]
        #     act = [act1, act2, act3, act4, act5, act6, act7]
        #
        #     self.act = []
        #     for i in act:
        #         self.act.append(i.detach())
        #     for idx, name in enumerate(names):
        #         act[idx].register_hook(save_grad(name))
        return h

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

class GCN(nn.Module):
    def __init__(self, in_dim=1, hidden_dim=8, n_classes=8):
        super(GCN, self).__init__()
        self.gcn1 = GraphConv(in_dim, hidden_dim,bias=False)  # 定义第一层图卷积
        self.gcn2 = GraphConv(hidden_dim, hidden_dim,bias=False)  # 定义第二层图卷积
        self.fc = nn.Linear(hidden_dim, n_classes,bias=False)  # 定义分类器

    def forward(self, g, task = 0):
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
        return self.fc(hg)  # [n, n_classes]

def get_avg_parameters(networks, weights=None):
    avg_pars = []
    for par_group in zip(*[net.parameters() for net in networks]):
        print([par.shape for par in par_group])
        if weights is not None:
            weighted_par_group = [par * weights[i] for i, par in enumerate(par_group)]
            avg_par = torch.sum(torch.stack(weighted_par_group), dim=0)
        else:
            # print("shape of stacked params is ", torch.stack(par_group).shape) # (2, 400, 784)
            avg_par = torch.mean(torch.stack(par_group), dim=0)
        print(avg_par.shape)
        avg_pars.append(avg_par)
    return avg_pars


def naive_ensembling(args, networks, test_loader):
    # simply average the weights in networks
    if args.width_ratio != 1:
        print("Unfortunately naive ensembling can't work if models are not of same shape!")
        return -1, None
    net_number = len(networks)
    weights = [1/net_number for i in range(net_number)]
    avg_pars = get_avg_parameters(networks, weights)
    ensemble_network = ResNet18(10,False,False)
    # put on GPU
    if args.gpu_id!=-1:
        ensemble_network = ensemble_network.cuda(args.gpu_id)

    # check the test performance of the method before
    log_dict = {}
    log_dict['test_losses'] = []
    # log_dict['test_counter'] = [i * len(train_loader.dataset) for i in range(args.n_epochs + 1)]

    # set the weights of the ensembled network
    for idx, (name, param) in enumerate(ensemble_network.state_dict().items()):
        ensemble_network.state_dict()[name].copy_(avg_pars[idx].data)

    # check the test performance of the method after ensembling
    log_dict = {}
    log_dict['test_losses'] = []
    # log_dict['test_counter'] = [i * len(train_loader.dataset) for i in range(args.n_epochs + 1)]
    return ensemble_network

class MlpNet(nn.Module):
    def __init__(self, args, width_ratio=-1):
        super(MlpNet, self).__init__()
        input_dim = 784
        if width_ratio != -1:
            self.width_ratio = width_ratio
        else:
            self.width_ratio = 1

        self.fc1 = nn.Linear(input_dim, int(args.num_hidden_nodes1/self.width_ratio), bias=not args.disable_bias)
        self.fc2 = nn.Linear(int(args.num_hidden_nodes1/self.width_ratio), int(args.num_hidden_nodes2/self.width_ratio), bias=not args.disable_bias)
        self.fc3 = nn.Linear(int(args.num_hidden_nodes2/self.width_ratio), int(args.num_hidden_nodes3/self.width_ratio), bias=not args.disable_bias)
        self.fc4 = nn.Linear(int(args.num_hidden_nodes3/self.width_ratio), 10, bias=not args.disable_bias)
        self.enable_dropout = args.enable_dropout

    def forward(self, x, disable_logits=False):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        if self.enable_dropout:
            x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        if self.enable_dropout:
            x = F.dropout(x, training=self.training)
        x = F.relu(self.fc3(x))
        if self.enable_dropout:
            x = F.dropout(x, training=self.training)
        x = self.fc4(x)

        if disable_logits:
            return x
        else:
            return F.log_softmax(x)




def train_model(model, device, train_loader, optimizer, epoch):
    model.train()                    #PyTorch提供的训练方法
    for batch_index, (data, label) in enumerate(train_loader):
        #部署到DEVICE
        data, label = data.to(device), label.to(device)
        #梯度初始化为0
        optimizer.zero_grad()
        #训练后的结果
        output = model(data)
        #计算损失（针对多分类任务交叉熵，二分类用sigmoid）
        loss = F.cross_entropy(output, label)
        #找到最大概率的下标
        pred = output.argmax(dim=1)
        #反向传播Backpropagation
        loss.backward()
        #参数的优化
        optimizer.step()
        if batch_index % 3000 == 0:
            print("Train Epoch : {} \t Loss : {:.6f}".format(epoch, loss.item()))
def test_model(model, device, test_loader):
    #模型验证
    model.eval()
    #统计正确率
    correct = 0.0
    #测试损失
    test_loss = 0.0
    with torch.no_grad():    # 不计算梯度，不反向传播
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)
            #测试数据
            output = model(data)
            #计算测试损失
            test_loss += F.cross_entropy(output, label).item()
            #找到概率值最大的下标
            pred = output.argmax(dim=1)
            #累计正确率
            correct += pred.eq(label.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        print("Test —— Average loss : {:.4f}, Accuracy : {:.3f}\n".format(test_loss, 100.0 * correct / len(test_loader.dataset)))

def get_optimizer(config, model_parameters):
    """
    Create an optimizer for a given model
    :param model_parameters: a list of parameters to be trained
    :return: Tuple (optimizer, scheduler)
    """
    print('lr is ', config['optimizer_learning_rate'])
    if config['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(
            model_parameters,
            lr=config['optimizer_learning_rate'],
            momentum=config['optimizer_momentum'],
            weight_decay=config['optimizer_weight_decay'],
        )
    else:
        raise ValueError('Unexpected value for optimizer')

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=config['optimizer_decay_at_epochs'],
        gamma=1.0/config['optimizer_decay_with_factor'],
    )

    return optimizer, scheduler

def cifar_train():
    DEVICE = torch.device("cuda")
    dataset = torchvision.datasets.CIFAR10
    data_mean = (0.4914, 0.4822, 0.4465)
    data_stddev = (0.2023, 0.1994, 0.2010)
    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(data_mean, data_stddev),
    ])

    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(data_mean, data_stddev),
    ])
    training_set = dataset(root='./files', train=True, download=True, transform=transform_train)
    test_set = dataset(root='./files', train=False, download=True, transform=transform_test)
    training_loader = torch.utils.data.DataLoader(
        training_set,
        batch_size=512,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=512,
        shuffle=False,
    )
    model = ResNet18(10, False,False).cuda()
    config = dict(
        dataset='Cifar10',
        model='resnet18',
        optimizer='SGD',
        optimizer_decay_at_epochs=[150, 250],
        optimizer_decay_with_factor=10.0,
        optimizer_learning_rate=0.1,
        optimizer_momentum=0.9,
        optimizer_weight_decay=0.0001,
        batch_size=256,
        num_epochs=300,
        seed=42,
    )
    optimizer, scheduler = get_optimizer(config, model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    for i in range(300):
        train_model(model, DEVICE, training_loader, optimizer, i)
        scheduler.step(i)
        if i %10 ==0:
            test_model(model,DEVICE,test_loader)
            torch.save(model.state_dict(), 'trained_models/CIFAR10/RESNET/SGD' + str(i) + '.pth')


def cifar100_train():
    from ClientTrain.dataset.Cifar100 import Cifar100Task
    DEVICE = torch.device("cuda:0")
    from ClientTrain.models.Nets import RepTail
    task = Cifar100Task('../data/cifar100', task_num=10)
    train, test = task.getTaskDataSet()
    train_dataset = train[0]
    val_dataset = test[0]
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=128,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=256,
                                             shuffle=False)
    model = Cifar100Net([3,32,32]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.0005)
    for i in range(300):
        train_model(model, DEVICE, train_loader, optimizer,i)
        if i % 3==0:
            test_model(model,DEVICE,test_loader)
            torch.save(model.state_dict(), 'trained_models/CIFAR100/6CNN/Adam' + str(i) + '.pth')




def test_fusion_model_resnet_cifar10(root_dir):
    args = parameters.get_parameters()
    args.gpu_id = 0
    DEVICE = torch.device("cuda:" + str(args.gpu_id))
    ls = os.listdir(root_dir)
    center_models=[]
    candidate_models=[]
    for i in ls:
        if i[0] is not 'c':
            model = ResNet18(10, False, False)
            model.load_state_dict(torch.load(root_dir+i))
            candidate_models.append(model.to(DEVICE))
        else:
            model = ResNet18(10,False,False)
            model_state_dict = torch.load(root_dir+i,map_location=(
                            lambda s, _: torch.serialization.default_restore_location(s, 'cuda:' + str(args.gpu_id))
                        ),)['model_state_dict']
            model.load_state_dict(model_state_dict)
            center_models.append({'client':1,'model':model.to(DEVICE)})

    dataset = torchvision.datasets.CIFAR10
    data_mean = (0.4914, 0.4822, 0.4465)
    data_stddev = (0.2023, 0.1994, 0.2010)
    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(data_mean, data_stddev),
    ])

    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(data_mean, data_stddev),
    ])
    training_set = dataset(root='./files', train=True, download=True, transform=transform_train)
    test_set = dataset(root='./files', train=False, download=True, transform=transform_test)
    training_loader = torch.utils.data.DataLoader(
        training_set,
        batch_size=512,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=512,
        shuffle=False,
    )

    Agg = FedDag(2,7,[3,32,32],model= ResNet18(10,False,False))
    aggs=Agg.update(center_models,candidate_models)
    for i in aggs:
        test_model(i, DEVICE, test_loader)
    # global_network = naive_ensembling(args,aggs,test_loader)
    # test_model(global_network,DEVICE,test_loader)

    # for i in center_models:
    #     test_model(i['model'],DEVICE,test_loader)

def test_fusion_model_6cnn_cifar100(root_dir):
    from ClientTrain.dataset.Cifar100 import Cifar100Task
    args = parameters.get_parameters()
    DEVICE = torch.device("cuda:" + str(args.server_gpu))
    ls = os.listdir(root_dir)
    model = Cifar100Net([3,32,32])
    kd_models=[]
    # center_models = []
    # candidate_models = []
    for e,i in enumerate(ls):
        model_state_dict = torch.load(root_dir + i)
        kd_models.append({'client':e , 'models': model_state_dict})
    # task = Cifar100Task('../data/cifar100', task_num=10)
    # train, test = task.getTaskDataSet()
    # train_dataset = train[0]
    # val_dataset = test[0]
    # # train_loader = torch.utils.data.DataLoader(train_dataset,
    # #                                            batch_size=128,
    # #                                            shuffle=True)
    #
    # test_loader = torch.utils.data.DataLoader(val_dataset,
    #                                           batch_size=256,
    #                                           shuffle=False)
    Agg = FedDag(len(ls), len(ls)*5, [3, 32, 32],dataname='CIFAR100',model=model)
    aggs = Agg.update(kd_models,0)
    # for i in aggs:
    #     test_model(i['model'], DEVICE, test_loader)
def test_fusion_model_textcnn_onlineshoping():
    from ClientTrain.dataset.online_shoping import OnlineTask
    # online = OnlineTask()
    # trains, tests = online.getTaskDataSet()
    # train_dataset = trains[0]['dataset']
    # test_datasets = tests[0]['dataset']
    # train_iter, dev_iter = data.Iterator.splits(
    #     (train_dataset, test_datasets),
    #     batch_sizes=(128, 128),
    #     sort_key=lambda x: len(x.text))
    model = Noemb_TextCNN()
    model_dir = ['/data/lpyx/FedAgg/Agg/test_text_model/'+i for i in os.listdir('/data/lpyx/FedAgg/Agg/test_text_model/')]
    all_models_states = [{'client':i, 'models':model.state_dict()} for i,md in enumerate(model_dir)]
    serverAgg = FedDag(len(model_dir),int(len(model_dir) * 5),datasize=[3,32,32],dataname='online_shopping',model=model)
    agg_models = serverAgg.update(all_models_states,0)

def get_nobias_dict(all_models_states):
    import collections
    all_no_bias = []
    for c in all_models_states:
        no_bias = collections.OrderedDict()
        cur_dict = c['models']
        for k in cur_dict.keys():
            if 'bias' not in k:
                no_bias[k] = cur_dict[k]
        all_no_bias.append({'client':c['client'], 'models':no_bias})
    return all_no_bias

def test_fusion_model_gcn_miniggraph():
    model = GCN()
    model_dir = ['/data/lpyx/FedAgg/ClientTrainGNN/test_models/' + i for i in os.listdir('/data/lpyx/FedAgg/ClientTrainGNN/test_models/')]
    all_models_states = [{'client': i, 'models': torch.load(md)} for i, md in enumerate(model_dir)]
    agg_models_states = get_nobias_dict(all_models_states)
    serverAgg = FedDag(len(model_dir), int(len(model_dir) * 5), datasize=[3, 32, 32], dataname='MiniGC',
                       model=model)
    agg_model = serverAgg.update(agg_models_states, 0)

def test_fusion_model_resnet():
    model = ResNet18()
    # model_dir = ['/data/lpyx/FedAgg/ClientTrainGNN/test_models/' + i for i in os.listdir('/data/lpyx/FedAgg/ClientTrainGNN/test_models/')]
    agg_models_states = [{'client': i, 'models': ResNet18().state_dict()} for i in range(2)]
    serverAgg = FedDag(2, 5, datasize=[3, 32, 32], dataname='CIFAR100',
                       model=model)
    agg_model = serverAgg.update(agg_models_states, 0)

# test_fusion_model_resnet_cifar10('trained_models/CIFAR10/RESNET/')
test_fusion_model_resnet()
# test_fusion_model_6cnn_cifar100('trained_models/CIFAR100/6CNN/')

# DEVICE = torch.device("cuda" )
# args = parameters.get_parameters()
# cifar_train()

# dataset = torchvision.datasets.CIFAR10
# data_mean = (0.4914, 0.4822, 0.4465)
# data_stddev = (0.2023, 0.1994, 0.2010)
# train_loader = torch.utils.data.DataLoader(
#           torchvision.datasets.MNIST('./files/', train=True, download=args.to_download,
#                                      transform=torchvision.transforms.Compose([
#                                        torchvision.transforms.ToTensor(),
#                                        torchvision.transforms.Normalize(
#                                            # only 1 channel
#                                            (0.1307,), (0.3081,))
#                                      ])),
#           batch_size=512,shuffle=True
#         )
#
#
# test_loader = torch.utils.data.DataLoader(
#   torchvision.datasets.MNIST('./files/', train=False, download=args.to_download,
#                              transform=torchvision.transforms.Compose([
#                                torchvision.transforms.ToTensor(),
#                                torchvision.transforms.Normalize(
#                                    (0.1307,), (0.3081,))
#                              ])),
#   batch_size=512, shuffle=True
# )
# center_models = []
# center_model1 = MlpNet(args)
#
#
# model_state_dict1 = torch.load('trained_models/final0.checkpoint',map_location=(
#                 lambda s, _: torch.serialization.default_restore_location(s, 'cuda:' + str(0))
#             ),)['model_state_dict']
# center_model1.load_state_dict(model_state_dict1)
# center_models.append({'client':1,'model':center_model1.cuda()})
#
# center_model2 = MlpNet(args)
# model_state_dict2 = torch.load('trained_models/final1.checkpoint',map_location=(
#                 lambda s, _: torch.serialization.default_restore_location(s, 'cuda:' + str(0))
#             ),)['model_state_dict']
# center_model2.load_state_dict(model_state_dict2)
# center_models.append({'client':2,'model':center_model2.cuda()})
#
# candidate_models=[]
# test_model(center_model1, DEVICE, test_loader)
#
# ls = os.listdir('trained_models')
#
# for i in ls:
#     if i[0] is not 'f':
#         model = MlpNet(args)
#         model.load_state_dict(torch.load('trained_models/'+i))
#         candidate_models.append(model.cuda())
#
# Agg = FedDag(center_models,candidate_models,[1,28,28])
# aggs=Agg.update(center_models,candidate_models)
#
# for i in aggs:
#     test_model(i, DEVICE, test_loader)




# networks=[candidate_models[0],candidate_models[0]]
# agg = naive_ensembling(args,networks,test_loader)
# test_model(agg,DEVICE,test_loader)
# for i in candidate_models:
#     i=i.cuda()
#     test_model(i, DEVICE, test_loader)
# test_model(model,DEVICE,test_loader)
# optimizer = optim.Adam(model.parameters())
# for epoch in range(1,20):
#     train_model(model, DEVICE, train_loader, optimizer, epoch)
#     test_model(model, DEVICE, test_loader)
#     torch.save(model.state_dict(), 'trained_models/Adam_model'+str(epoch)+'.pth')