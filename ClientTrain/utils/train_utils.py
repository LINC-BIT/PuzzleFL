# Modified from: https://github.com/pliang279/LG-FedAvg/blob/master/utils/train_utils.py
# credit goes to: Paul Pu Liang

from torchvision import datasets, transforms
# from models.Nets import CNNCifar, CNNCifar100, RNNSent, MLP
from ClientTrain.AggModel.densenet import Densenet121
from ClientTrain.AggModel.mobilenet import Mobilenetv2
from ClientTrain.AggModel.resnet import Resnet18
from ClientTrain.AggModel.sixcnn import SixCNN
from ClientTrain.AggModel.vit import RepTailTinyPiT, RepTailTinyViT, RepTailViT
from ClientTrain.dataset.Tinyimagenet import TinyimageTask
from ClientTrain.utils.sampling import noniid
import os
import json
from ClientTrain.dataset.Cifar100 import Cifar100Task
from ClientTrain.dataset.fc100 import FC100Task
from ClientTrain.dataset.miniimagenet import MiniImageTask
# from dataset.fc100 import FC100Task
# from dataset.core50 import Core50Task

import ClientTrain.AggModel.weitnet.nets as weitnets


trans_mnist = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))])
trans_cifar10_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])])
trans_cifar10_val = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])
trans_cifar100_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                               std=[0.267, 0.256, 0.276])])
trans_cifar100_val = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                              std=[0.267, 0.256, 0.276])])


def get_data(args):
    if args.dataset == 'mnist':
        dataset_train = datasets.MNIST('data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        dict_users_train, rand_set_all = noniid(dataset_train, args.num_users, args.shard_per_user, args.num_classes)
        dict_users_test, rand_set_all = noniid(dataset_test, args.num_users, args.shard_per_user, args.num_classes, rand_set_all=rand_set_all, testb=True)
    elif args.dataset == 'cifar10':
        dataset_train = datasets.CIFAR10('data/cifar10', train=True, download=True, transform=trans_cifar10_train)
        dataset_test = datasets.CIFAR10('data/cifar10', train=False, download=True, transform=trans_cifar10_val)
        dict_users_train, rand_set_all = noniid(dataset_train, args.num_users, args.shard_per_user, args.num_classes)
        dict_users_test, rand_set_all = noniid(dataset_test, args.num_users, args.shard_per_user, args.num_classes, rand_set_all=rand_set_all)
    elif args.dataset == 'cifar100':
        # 引入原始的抽样方式
        from ClientTrain.utils.sampling import noniid
        
        rand_set_all = [
            [2, 7, 5, 6, 1, 4, 3, 9], 
            [6, 7, 9, 8, 1, 4, 5, 0],
            [3, 0, 8, 5, 4, 7, 6, 2], 
            [4, 7, 0, 2, 3, 9, 8, 5],
            [3, 5, 7, 1, 2, 0, 6, 8], 
            [2, 1, 4, 3, 9, 6, 5, 8], 
            [4, 1, 2, 5, 0, 9, 3, 8],
            [7, 2, 1, 4, 6, 0, 9, 5],
            [0, 2, 3, 6, 7, 8, 9, 1],
            [0, 6, 7, 8, 4, 1, 9, 3]]
        # label_num = [0]*10
        # for i in range(10):
        #     for j in range(8):
        #         label_num[rand_set_all[i][j]]+=1
        cifar100 = Cifar100Task('/data/lpyx/FedFPKD/data/cifar100',task_num=10)
        # dataset_train = datasets.CIFAR100('data/cifar100', train=True, download=True, transform=trans_cifar100_train)
        # dataset_test = datasets.CIFAR100('data/cifar100', train=False, download=True, transform=trans_cifar100_val)
        dataset_train,dataset_test = cifar100.getTaskDataSet()
        # dict_users_train, rand_set_all = noniid(dataset_train, args.num_users, args.shard_per_user, args.num_classes)
        # dict_users_test, rand_set_all = noniid(dataset_test, args.num_users, args.shard_per_user, args.num_classes,
        #                                        rand_set_all=rand_set_all)
        # for dataset_train,dataset_test in zip(dataset_trains,dataset_tests):
        # dict_users_train, rand_set_all = noniid(dataset_train[0], args.num_users, args.shard_per_user, args.num_classes,rand_set_all=rand_set_all)
        if args.num_users == 10:
            dict_users_train, rand_set_all = noniid(dataset_train[0], args.num_users, args.shard_per_user, args.num_classes,rand_set_all=rand_set_all)
            dict_users_test, rand_set_all = noniid(dataset_test[0], args.num_users, 10, args.num_classes)
        else:
            # dict_users_test, _ = noniid(dataset_test[0], args.num_users//2, args.shard_per_user, args.num_classes,rand_set_all=rand_set_all[0:25])
            # dict_users_test2, _ = noniid(dataset_test[0], args.num_users//2, args.shard_per_user, args.num_classes,
            #                             rand_set_all=rand_set_all[25:])
            # for k,v in dict_users_test2.items():
            #     dict_users_test[k+ args.num_users//2] = v
            
            # TODO: 这里用FedViT里的设定
            # 引入原始的抽样方式
            from ClientTrain.utils.sampling_raw import noniid
            dict_users_train, rand_set_all = noniid(dataset_train[0], args.num_users, args.shard_per_user, args.num_classes)
            dict_users_test, rand_set_all = noniid(dataset_test[0], args.num_users, args.shard_per_user, args.num_classes, rand_set_all=rand_set_all)

    elif args.dataset == 'miniimagenet':
        Miniimagenet = MiniImageTask(root='/data/lpyx/FedFPKD/data/mini-imagenet/',json_path="/data/lpyx/FedFPKD/data/mini-imagenet/classes_name.json",task_num=10)
        dataset_train, dataset_test = Miniimagenet.getTaskDataSet()
        
        # 引入原始的抽样方式
        from ClientTrain.utils.sampling_raw import noniid
        
        dict_users_train, rand_set_all = noniid(dataset_train[0], args.num_users, args.shard_per_user, args.num_classes,dataname='miniimagenet')
        dict_users_test, rand_set_all = noniid(dataset_test[0], args.num_users, args.shard_per_user, args.num_classes,
                                               rand_set_all=rand_set_all,dataname='miniimagenet')
    elif args.dataset == 'tinyimagenet':
        Tinyimagenet = TinyimageTask(root='/data/lpyx/FedFPKD/data/tiny-imagenet-200', task_num=20)
        dataset_train, dataset_test = Tinyimagenet.getTaskDataSet()
        
        # 引入原始的抽样方式
        from ClientTrain.utils.sampling_raw import noniid
        
        dict_users_train, rand_set_all = noniid(dataset_train[0], args.num_users, args.shard_per_user, args.num_classes,
                                                dataname='tinyimagenet')
        dict_users_test, rand_set_all = noniid(dataset_test[0], args.num_users, args.shard_per_user, args.num_classes,
                                               rand_set_all=rand_set_all, dataname='tinyimagenet')
    elif args.dataset == 'FC100':
        Fc100 = FC100Task(root='/data/lpyx/FedFPKD/data/FC100',task_num=10)
        dataset_train, dataset_test = Fc100.getTaskDataSet()
        dict_users_train, rand_set_all = noniid(dataset_train[0], args.num_users, args.shard_per_user, args.num_classes,
                                                dataname='FC100')
        dict_users_test, rand_set_all = noniid(dataset_test[0], args.num_users, args.shard_per_user, args.num_classes,
                                               rand_set_all=rand_set_all, dataname='FC100')
    elif args.dataset == 'Corn50':
        Corn50 = Core50Task(root='data', task_num=11)
        dataset_train, dataset_test = Corn50.getTaskDataSet()
        dict_users_train, rand_set_all = noniid(dataset_train[0], args.num_users, args.shard_per_user, args.num_classes,
                                                dataname='Corn50')
        dict_users_test, rand_set_all = noniid(dataset_test[0], args.num_users, args.shard_per_user, args.num_classes,
                                               rand_set_all=rand_set_all, dataname='Corn50')
    else:
        exit('Error: unrecognized dataset')

    return dataset_train, dataset_test, dict_users_train, dict_users_test

def read_data(train_data_dir, test_data_dir):
    '''parses data in given train and test data directories
    assumes:
    - the data in the input directories are .json files with 
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users
    
    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''
    clients = []
    groups = []
    train_data = {}
    test_data = {}

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.json')]
    for f in train_files:
        file_path = os.path.join(train_data_dir,f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        train_data.update(cdata['user_data'])

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.json')]
    for f in test_files:
        file_path = os.path.join(test_data_dir,f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        test_data.update(cdata['user_data'])

    clients = list(train_data.keys())

    return clients, groups, train_data, test_data


def get_model_bak(args):
    if args.model == 'cnn' and 'cifar100' in args.dataset:
        net_glob = CNNCifar100(args=args).to(args.device)
    elif args.model == 'cnn' and 'cifar10' in args.dataset:
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'mlp' and 'mnist' in args.dataset:
        net_glob = MLP(dim_in=784, dim_hidden=256, dim_out=args.num_classes).to(args.device)
    elif args.model == 'cnn' and 'femnist' in args.dataset:
        net_glob = CNN_FEMNIST(args=args).to(args.device)
    elif args.model == 'mlp' and 'cifar' in args.dataset:
        net_glob = MLP(dim_in=3072, dim_hidden=512, dim_out=args.num_classes).to(args.device)
    elif 'sent140' in args.dataset:
        net_glob = model = RNNSent(args,'LSTM', 2, 25, 128, 1, 0.5, tie_weights=False).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)

    return net_glob


def get_model(args):
    ouputsize = 100
    
    if args.task == 20 and 'tinyimagenet' in args.dataset:
        ouputsize = 200
        
    if args.model == 'cnn' and 'cifar100' in args.dataset:
        net_glob = SixCNN([3,32,32],outputsize=ouputsize)
    elif args.model == 'tiny_vit':
        net_glob = RepTailTinyViT(inputsize=None, outputsize=ouputsize, nc_per_task=10)
    elif args.model == 'tiny_pit':
        net_glob = RepTailTinyPiT(inputsize=None, outputsize=ouputsize, nc_per_task=10)
    elif args.model == 'resnet':
        net_glob = Resnet18(inputsize=None, outputsize=ouputsize, nc_per_task=10)
    elif args.model == 'mobinet':
        net_glob = Mobilenetv2(inputsize=None, outputsize=ouputsize, nc_per_task=10)
    elif args.model == 'dense':
        net_glob = Densenet121(inputsize=None, outputsize=ouputsize, nc_per_task=10)
    
    
    elif args.model == 'weit_tiny_vit':
        net_glob = weitnets.WEITTinyViT(output=ouputsize, nc_per_task=args.num_classes)
    elif args.model == 'weit_tiny_pit':
        net_glob = weitnets.WEITTinyPiT(output=ouputsize, nc_per_task=args.num_classes)
    elif args.model == 'weit_cnn':
        net_glob = weitnets.WEIT6CNN(output=ouputsize, nc_per_task=args.num_classes)
    elif args.model == 'weit_resnet':
        net_glob = weitnets.WEITResnet18(output=ouputsize, nc_per_task=args.num_classes)
    elif args.model == 'weit_mobinet':
        net_glob = weitnets.WEITMobiNet(output=ouputsize, nc_per_task=args.num_classes)
    elif args.model == 'weit_densenet':
        net_glob = weitnets.WEITDense(output=ouputsize, nc_per_task=args.num_classes)
    else:
        exit('Error: unrecognized model')
    print(net_glob)

    return net_glob


if __name__ == '__main__':
    # model = Mobilenetv2(inputsize=None, outputsize=100, nc_per_task=10)
    model = RepTailTinyPiT(inputsize=None, outputsize=100, nc_per_task=10)
    # import torch
    # x = torch.randn((4,3,224,224))
    # print(model(x))

    p_num = 0
    for p in model.parameters():
        p_num += p.numel()

    print('param num:', p_num)