import copy
import itertools
from random import shuffle
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from ClientTrain.utils.options import args_parser
from ClientTrain.utils.train_utils import get_data, get_model, read_data
from ClientTrain.models.Update import LocalUpdate,DatasetSplit
from ClientTrain.models.test import test_img_local_all

from Infocome.HDFL.HDFL import Appr, LongLifeTrain
from ClientTrain.models.Nets import RepTail
from torch.utils.data import DataLoader
import time
from Infocome.HDFL.agg import Aggregator
from randseed import set_rand


if __name__ == '__main__':
    
    # parse args
    args = args_parser()
    set_rand(args.seed)
   
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # all task data
    dataset_train, dataset_test, dict_users_train, dict_users_test = get_data(args)
    for idx in dict_users_train.keys():
        np.random.shuffle(dict_users_train[idx])

    # algin the task sequence for each client
    client_task = [[j for j in range(args.task)] for i in range(args.num_users)]
    for i in client_task:
        shuffle(i)
        

    net_glob = get_model(args)
    net_glob.train()

    # training
    serverAgg = Aggregator(copy.deepcopy(net_glob.to(args.device)))
    indd = None  # indices of embedding for sent140
    loss_train = []
    accs = []
    times = []
    accs10 = 0
    accs10_glob = 0
    start = time.time()
    task=-1
    
    
    # HDFL setting  ########################################
    num_clients = args.num_users
    num_cell = 3
    tau_2 = 3
    
    # write = SummaryWriter('./Dist-79/ClientTrainOur/log/HDFL/' + args.dataset+'_'+'round' + str(args.round) + '_frac' + str(args.frac) + f'_tau_{tau_2}_' + str(seed) )
    write = SummaryWriter(f'/data/zxj/projects/vscodes/Dist-79/explogs-{args.dataset}/{args.alg}/' + args.dataset+'_' + args.model +'_' +'round' + str(args.round) + "_epoch_" + str(args.local_ep) + '_frac' + str(args.frac) + '_' + str(args.seed)
                          + "_" + str(int(time.time()))[-4:])
    
    # Calculate cell indices
    # cells = [list(range(i * (num_clients // num_cell), (i + 1) * (num_clients // num_cell))) for i in range(num_cell)]
    
    # remaining_clients = list(range(num_cell * (num_clients // num_cell), num_clients))
    # for i, client in enumerate(remaining_clients):
    #     cells[-1].append(client)

    # 创建一个包含所有用户 ID 的列表
    user_ids = list(range(num_clients))

    # 随机打乱用户 ID 列表
    random.shuffle(user_ids)

    # 将用户 ID 分配到各个 cell 中
    cells = [user_ids[i * (num_clients // num_cell):(i + 1) * (num_clients // num_cell)] for i in range(num_cell)]

    # 处理剩余的用户
    remaining_clients = user_ids[num_cell * (num_clients // num_cell):]
    for i, client in enumerate(remaining_clients):
        cells[i % num_cell].append(client)
        
    # 权重 for cell 内部聚合
    W = np.zeros((num_clients, num_clients))
    for ce in cells:
        # 每个cell内部的用户之间的权重，权重和为1
        for i in ce:
            for j in ce:
                W[i][j] = 1 / len(ce)
    
    # 权重 for cell 之间聚合
    S = np.zeros((num_clients, num_clients))
    for ce in cells:
        for i in ce:
            for j in ce:
                S[i][j] = 1/num_cell
    #######################################################

    
    if 'EWC' in str(args.alg):
        from Infocome.HDFL.HDFL_EWC import Appr, LongLifeTrain
        apprs = [Appr(copy.deepcopy(net_glob.to(args.device)), tr_dataloader=None, args=args, lr=args.lr, nepochs=args.local_ep) for i in range(args.num_users)]
    elif 'GEM' in str(args.alg):
        from Infocome.HDFL.HDFL_GEM import Appr, LongLifeTrain
        sample_shape = (3, 32, 32)
        apprs = [Appr(net_glob.to(args.device), sample_shape, 100,10, args) for i in range(args.num_users)]  # TODO: 目前自考虑 CIFAR100
    else:
        print("No CL algorithm is specified!!!")
        
    # apprs = [Appr(copy.deepcopy(net_glob.to(args.device)), tr_dataloader=None, args=args, lr=args.lr, nepochs=args.local_ep, client_task = client_task[i], idx=i, cell_neighbors=None) for i in range(args.num_users)]
    print(args.round)
    
    for ce in cells:
        for i in ce:
            apprs[i].set_cell_neibors(ce)
    
    # train loop
    for iter in range(args.epochs):     # 这里 iter 是一个通信 Round
        if iter % (args.round) == 0:
            task+=1
        
        loss_locals = []
        
        all_users = [i for i in range(args.num_users)]
        
        times_in = []
        total_len = 0
        tr_dataloaders= None
        all_local_models = []

        # local train in each client
        for ind, idx in enumerate(all_users):
            start_in = time.time()

            tr_dataloaders = DataLoader(DatasetSplit(dataset_train[client_task[idx][task]],dict_users_train[idx][:args.m_ft], tran_task=[task,client_task[idx][task]]),batch_size=args.local_bs, shuffle=True)
            appr = apprs[idx]
            # appr.set_trData(tr_dataloaders)
            # appr.set_model(net_local.to(args.device))
            last = iter == args.epochs
            
            # local train 
            if 'EWC' in args.alg: 
                appr.set_trData(tr_dataloaders)
                local_model,loss, _ = LongLifeTrain(args,appr,iter, idx)   
            elif 'GEM' in args.alg:
                local_model,loss, indd = LongLifeTrain(args,appr,tr_dataloaders,iter,idx)
            else:
                print("No CL algorithm is specified!!!")
                
            # local_model,loss, _ = LongLifeTrain(args,appr,iter,idx)   # TODO: 传递一个参数，表示本地训练的迭代数
            loss_locals.append(copy.deepcopy(loss))
        

        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train.append(loss_avg)


        acc_test, loss_test = test_img_local_all(None, args, dataset_test, dict_users_test,task,apprs=apprs,w_locals=None,return_all=False,write=write,round=iter,client_task=client_task)
        accs.append(acc_test)

        print('Round {:3d}, Train loss: {:.3f}, Test loss: {:.3f}, Test accuracy: {:.2f}'.format(
                iter, loss_avg, loss_test, acc_test))

        ############## HDFL 聚合 ###################
        if iter % tau_2 == 0:
            agg_W = W * S
        else:
            agg_W = W
        
        glob_models = []
        for idx, appr in enumerate(apprs):
            if iter % tau_2 == 0 and iter > 0:
                # 只与其他cell的用户进行聚合
                temp = copy.deepcopy(all_users)
                for user in apprs[idx].get_cell_neibors():
                    temp.remove(user)
                    
                neibor_ids = random.sample(temp, args.neibour)  # 从所有用户中随机选择两个用户
                
            else:
                cell_users = apprs[idx].get_cell_neibors()   # 只与所在cell内的进行聚合
                neibor_ids = copy.deepcopy(cell_users)
                neibor_ids.remove(idx)
                # neibor_ids = random.sample(neibor_ids, args.neibour)  # 从所有用户中随机选择两个用户
                    
            # 聚合
            local_models = []
            mix_weights = []
            local_models.append(appr.model)
            mix_weights.append(agg_W[idx][idx])
            
            for nei_idx in neibor_ids:
                local_models.append(apprs[nei_idx].model)    
                mix_weights.append(agg_W[idx][nei_idx])
            agg_model = serverAgg.update(local_models, mix_weights)
            
            glob_models.append(copy.deepcopy(agg_model))
            
        # 更新
        for idx in range(args.num_users):
            apprs[idx].model.load_state_dict(glob_models[idx])

    end = time.time()
    print(end - start)
