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

# from Infocome.FedHP.FedHP import Appr, LongLifeTrain
from ClientTrain.models.Nets import RepTail
from torch.utils.data import DataLoader
import time
from baselines.FedHP.agg import Aggregator
import baselines.FedHP.funcs as funcs
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
    
    # write = SummaryWriter('./Dist-79/ClientTrainOur/log/FedHP/' + args.dataset+'_'+'round' + str(args.round) + '_frac' + str(args.frac) + str(seed) )
    write = SummaryWriter(f'/data/zxj/projects/vscodes/Dist-79/explogs-{args.dataset}/{args.alg}/' + args.dataset+'_' + args.model +'_' +'round' + str(args.round) + "_epoch_" + str(args.local_ep) + '_frac' + str(args.frac) + '_' + str(args.seed)
                          + "_" + str(int(time.time()))[-4:])
    
    # FedHP setting  ########################################
    coord = funcs.Coordinator(args.num_users, args.neibour)
    #######################################################

    if 'EWC' in str(args.alg):
        from Infocome.FedHP.FedHP_EWC import Appr, LongLifeTrain
        apprs = [Appr(copy.deepcopy(net_glob.to(args.device)), tr_dataloader=None, args=args, lr=args.lr, nepochs=args.local_ep) for i in range(args.num_users)]
    elif 'GEM' in str(args.alg):
        from Infocome.FedHP.FedHP_GEM import Appr, LongLifeTrain
        sample_shape = (3, 32, 32)
        apprs = [Appr(net_glob.to(args.device), sample_shape, 100,10, args) for i in range(args.num_users)]  # TODO: 目前自考虑 CIFAR100
    else:
        print("No CL algorithm is specified!!!")
        
    # apprs = [Appr(copy.deepcopy(net_glob.to(args.device)), tr_dataloader=None, args=args, lr=args.lr, nepochs=args.local_ep) for i in range(args.num_users)]
    print(args.round)

    
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
            
            # local train 
            if 'EWC' in args.alg: 
                appr.set_trData(tr_dataloaders)
                local_model,loss, _ = LongLifeTrain(args,appr,iter, idx)   
            elif 'GEM' in args.alg:
                local_model,loss, indd = LongLifeTrain(args,appr,tr_dataloaders,iter,idx)
            else:
                print("No CL algorithm is specified!!!")
                
            loss_locals.append(copy.deepcopy(loss))
            
            last = iter == args.epochs

        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train.append(loss_avg)


        acc_test, loss_test = test_img_local_all(None, args, dataset_test, dict_users_test,task,apprs=apprs,w_locals=None,return_all=False,write=write,round=iter,client_task=client_task)
        accs.append(acc_test)

        print('Round {:3d}, Train loss: {:.3f}, Test loss: {:.3f}, Test accuracy: {:.2f}'.format(
                iter, loss_avg, loss_test, acc_test))

        ############## FedHP 聚合 ###################
        # FedHP 获取网络拓扑结构
        Ah = coord.get_topology()
        
        glob_models = []
        for idx, appr in enumerate(apprs):
            # 获取邻居客户端,这个neibor_ids中也包含了当前client自己
            # neibor_ids = Ah[idx].where(Ah[idx] == 1)
            neibor_ids = coord.get_neibors(idx, Ah)
            
            # 聚合
            local_models = []
            local_models.append(appr.model) # 当前client模型放在第一位
            
            for nei_idx in neibor_ids:
                local_models.append(apprs[nei_idx].model)    
                
            agg_model = serverAgg.update(local_models)
            
            glob_models.append(copy.deepcopy(agg_model))
            
        # 更新本地模型
        for idx in range(args.num_users):
            apprs[idx].model.load_state_dict(glob_models[idx])
        
        # 收集每个client的邻居共识距离
        D_h_i_j_list = []
        
        
        for idx in range(args.num_users):
            neibor_ids = coord.get_neibors(idx, Ah)
            neibor_models = [apprs[j].model for j in neibor_ids]
            
            D_h_i_j_list.append(
                funcs.compute_consensus_distance(idx, apprs[idx].model, neibor_models, neibor_ids, 
                                                 [cid for cid in range(args.num_users)])
            )
        
        # 构建全局共识距离
        D_h = funcs.compose_consensus_matrix(D_h_i_j_list, [cid for cid in range(args.num_users)])
        
        coord.set_dh(D_h)
        
        
    end = time.time()
    print(end - start)
