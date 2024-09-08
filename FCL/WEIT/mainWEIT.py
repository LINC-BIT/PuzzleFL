import copy
import itertools
from random import shuffle

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from ClientTrain.utils.options import args_parser
from ClientTrain.utils.train_utils import get_data, get_model, read_data
from ClientTrain.models.Update import LocalUpdate,DatasetSplit
from ClientTrain.models.test import test_img_local_all
from FCL.WEIT.WEIT import Appr, LongLifeTrain

from ClientTrain.AggModel.sixcnn import SixCNN
from torch.utils.data import DataLoader
import time
from Agg.FedAvg import FedAvg
from randseed import set_rand
import os
import ClientTrain.AggModel.weitnet.consts as CONST


if __name__ == '__main__':
    # parse args
    args = args_parser()
    set_rand(args.seed)
    
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    os.environ['CUDA_VISIBLE_DEVICE'] = f'{args.gpu}'  # compatible to cuda()
    CONST.DEVICE = args.device
    
    # hyperparam for WEIT
    args.wd = 1e-4
    args.lambda_l1 = 1e-3
    args.lambda_l2 = 1
    args.lambda_mask = 0
    

    # dataset
    lens = np.ones(args.num_users)
    dataset_train, dataset_test, dict_users_train, dict_users_test = get_data(args)
    for idx in dict_users_train.keys():
        np.random.shuffle(dict_users_train[idx])
    
    # client task sequence
    client_task=[[j for j in range(args.task)] for i in range(args.num_users)]
    for i in client_task:
        shuffle(i)

    print(args.alg)
    
    # writer
    write = SummaryWriter(f'/data/zxj/projects/vscodes/Dist-79/explogs-{args.dataset}/{args.alg}/' + args.dataset+'_' + args.model +'_' +'round' + str(args.round) + "_epoch_" + str(args.local_ep) + '_frac' + str(args.frac) + '_' + str(args.seed)
                          + "_" + str(int(time.time()))[-4:])
    
    # model
    net_glob = get_model(args)
    net_glob.train()


    # training
    indd = None  # indices of embedding for sent140
    loss_train = []
    accs = []
    times = []
    accs10 = 0
    accs10_glob = 0
    start = time.time()
    task=-1
    
    apprs = [Appr(copy.deepcopy(net_glob).to(args.device), None, lr=args.lr, nepochs=args.local_ep, args=args) for i in range(args.num_users)]
    
    print(args.round)
    
    
    # serverAgg = FedAvg(copy.deepcopy(net_glob).to(args.device))
    for iter in range(args.epochs):
        all_users = [i for i in range(args.num_users)]

        # 设定每个client的邻居集合
        for ind, idx in enumerate(all_users):
            temp = copy.deepcopy(all_users)
            temp.remove(idx)
            
            idxs_users = np.random.choice(temp, args.neibour, replace=False)
            for idx_user in idxs_users:
                apprs[idx].set_neibors(idxs_users)
                
        
        if iter % (args.round) == 0:
            task+=1
            
            # 为每个client分配知识库            
            for idx in range(args.num_users):
                # build knowledge database
                from_kb = []
                for name, para in net_glob.named_parameters():
                    if 'aw' in name:
                        shape = np.concatenate([para.shape, [int(round(args.num_users * args.frac))]], axis=0)
                        from_kb_l = np.zeros(shape)
                        from_kb_l = torch.from_numpy(from_kb_l)
                        from_kb.append(from_kb_l)
                
                apprs[idx].set_kb(copy.deepcopy(from_kb))
            
            
        # client sampling
        m = max(int(args.frac * args.num_users), 1)
        if iter == args.epochs:
            m = args.num_users
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        
        times_in = []
        total_len = 0
        tr_dataloaders= None
        loss_locals = []

        shared_w_list = []
        for ind, idx in enumerate(idxs_users):
            start_in = time.time()

            tr_dataloaders = DataLoader(DatasetSplit(dataset_train[client_task[idx][task]],dict_users_train[idx][:args.m_ft],tran_task=[task,client_task[idx][task]]),batch_size=args.local_bs, shuffle=True)
            appr = apprs[idx]

            # appr.set_model(net_local.to(args.device))
            appr.set_trData(tr_dataloaders)
            last = iter == args.epochs

            # local_models,loss, indd = LongLifeTrain(args,appr,iter,None,idx)
            from_kb = appr.get_kb()
            shared_w, _, loss, indd = LongLifeTrain(args, appr, iter, from_kb, idx)
            shared_w_list.append(copy.deepcopy(shared_w))
            loss_locals.append(copy.deepcopy(loss))
          

        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train.append(loss_avg)


        acc_test, loss_test = test_img_local_all(None, args, dataset_test, dict_users_test,task,apprs=apprs,w_locals=None,return_all=False,write=write,round=iter,client_task=client_task)
        accs.append(acc_test)


        print('Round {:3d}, Train loss: {:.3f}, Test loss: {:.3f}, Test accuracy: {:.2f}'.format(
                iter, loss_avg, loss_test, acc_test))

        # DisGOSSIP
        clients_global = []
        all_users = [i for i in range(args.num_users)]
        
        for ind, idx in enumerate(all_users):
            cur_local_models = []
            
            for c_idx in apprs[idx].get_neibors():
                cur_local_models.append(shared_w_list[c_idx])              
            
            # 聚合            
            glob_sw = copy.deepcopy(apprs[idx].get_sw())  
            for sw in cur_local_models:
                for i, w in enumerate(sw):
                    glob_sw[i] = glob_sw[i] + w
            for i in range(len(glob_sw)):
                glob_sw[i] = torch.div(glob_sw[i], len(cur_local_models)+1)

            clients_global.append(glob_sw)
            
        # 更新 
        for ind,idx in enumerate(all_users):
            apprs[idx].set_sw(clients_global[idx])
        
        
    end = time.time()
    print(end - start)
    print(times)
