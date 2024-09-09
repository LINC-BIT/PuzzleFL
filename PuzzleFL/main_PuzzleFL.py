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
from PuzzleFL.LongLifeMethod.PuzzleFL_EWC import Appr,LongLifeTest,LongLifeTrain
from ClientTrain.AggModel.sixcnn import SixCNN
from torch.utils.data import DataLoader
from Agg.AggModel.sixcnn import SixCNN as KDmodel
import time
# from Agg.FedDag import FedDag
from randseed import set_rand

if __name__ == '__main__':
    # parse args
    # seed = 614
    # seed = 314
    # seed = 3040
    # seed = 508
    # seed = 505
    # set_rand(seed)
    
    # seed
    args = args_parser()
    set_rand(args.seed)
    
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    lens = np.ones(args.num_users)
    
    # dataset
    dataset_train, dataset_test, dict_users_train, dict_users_test = get_data(args)
    
    # model
    net_glob = get_model(args)
    net_glob.train()
    kd_model = copy.deepcopy(net_glob)
    
    for idx in dict_users_train.keys():
        np.random.shuffle(dict_users_train[idx])
    client_task=[[j for j in range(args.task)] for i in range(args.num_users)]
    for i in client_task:
        shuffle(i)
    client_task = [[0, 1, 5, 6, 8, 2, 9, 7, 3, 4], 
                   [0, 1, 5, 6, 8, 2, 9, 7, 3, 4], 
                   [0, 1, 5, 6, 8, 2, 9, 7, 3, 4], 
                   [9, 7, 6, 1, 0, 4, 5, 3, 2, 8],
                   [9, 7, 6, 1, 0, 4, 5, 3, 2, 8], 
                   [9, 7, 6, 1, 0, 4, 5, 3, 2, 8], 
                   [6, 8, 3, 0, 7, 5, 4, 2, 1, 9], 
                   [6, 8, 3, 0, 7, 5, 4, 2, 1, 9],
                   [6, 8, 3, 0, 7, 5, 4, 2, 1, 9], 
                   [6, 8, 3, 0, 7, 5, 4, 2, 1, 9]]
    clients_agg = [[1,2],[0,2],[0,1],[4,5],[3,5],[3,4],[7,8,9],[6,8,9],[6,7,9],[6,7,8]]

    print(args.alg)
    
    write = SummaryWriter(f'/data/zxj/projects/vscodes/Dist-79/explogs/{args.alg}/' + args.dataset+'_' + args.model +'_' +'round' + str(args.round) + "_epoch_" + str(args.local_ep) + '_frac' + str(args.frac) + '_' + str(args.seed)
                          + "_" + str(int(time.time()))[-4:])
    
    all_users = [i for i in range(args.num_users)]

    # generate list of local models for each user
    net_local_list = []
    w_locals = {}
    for user in range(args.num_users):
        w_local_dict = {}
        for key in net_glob.state_dict().keys():
            w_local_dict[key] = net_glob.state_dict()[key]
        w_locals[user] = w_local_dict

    
    # training
    indd = None  # indices of embedding for sent140
    loss_train = []
    accs = []
    times = []
    accs10 = 0
    accs10_glob = 0
    start = time.time()
    task=-1
    
    apprs = [Appr(copy.deepcopy(net_glob).to(args.device), None,lr=args.lr, nepochs=args.local_ep, args=args,kd_model=kd_model,client_task = client_task[i], idx=i) for i in range(args.num_users)]
    print(args.round)
    
    w_globals = []
    for iter in range(args.epochs):
        if iter % (args.round) == 0:
            task+=1
            for idx in all_users:
                apprs[idx].new_kd()
        elif iter % (args.round) == 1:
            for idx in all_users:
                apprs[idx].extract_task_class(task)
        
        w_glob = {}
        loss_locals = []
        m = max(int(args.frac * args.num_users), 1)
        if iter == args.epochs:
            m = args.num_users

        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        # w_keys_epoch = w_glob_keys
        times_in = []
        total_len = 0
        tr_dataloaders= None
        all_kd_models = []

        for ind, idx in enumerate(idxs_users):
            start_in = time.time()

            tr_dataloaders = DataLoader(DatasetSplit(dataset_train[client_task[idx][task]],dict_users_train[idx],tran_task=[task,client_task[idx][task]]),batch_size=args.local_bs, shuffle=True)
            appr = apprs[idx]
            appr.set_trData(tr_dataloaders)
            
            last = iter == args.epochs
            
            if iter % (args.round) == 0:
                appr.train_kd(iter//args.round)

            kd_models,loss, indd = LongLifeTrain(args,appr,iter,None,idx)
            all_kd_models.append({'client':idx, 'models': kd_models})

            loss_locals.append(copy.deepcopy(loss))

        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train.append(loss_avg)

        # Global test
        acc_test, loss_test = test_img_local_all(None, args, dataset_test, dict_users_test,task,apprs=apprs,w_locals=None,return_all=False,write=write,round=iter,client_task=client_task)
        accs.append(acc_test)

        print('Round {:3d}, Train loss: {:.3f}, Test loss: {:.3f}, Test accuracy: {:.2f}'.format(
                iter, loss_avg, loss_test, acc_test))
        
        ## DFL Aggregation
        for ind, idx in enumerate(all_users):
            cur_recive = apprs[idx].recive_class(task)
            cur_local_models = []
            idxs_users = clients_agg[idx]
            other_client_models = []
            for idx_user in idxs_users:
                other_send = apprs[idx_user].send_class(task)
                real_send = list(set(other_send) & set(cur_recive)) # knowledge intersection
                real_send_model = apprs[idx_user].get_task_class(real_send)
                for rm in real_send_model:
                    other_client_models.append(rm)
            apprs[idx].aggregation(other_client_models, task)
        

        for idx in all_users:
            apprs[idx].update_local_class_knowledge(task)