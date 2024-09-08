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
from NoAgg.LongLifeMethod.Avg_GEM import Appr,LongLifeTrain
from ClientTrain.models.Nets import RepTail
from torch.utils.data import DataLoader
import time
from Agg.FedAvg import FedAvg
from randseed import set_rand
if __name__ == '__main__':
    # seed
    args = args_parser()
    set_rand(args.seed)
    
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    lens = np.ones(args.num_users)
    dataset_train, dataset_test, dict_users_train, dict_users_test = get_data(args)
    for idx in dict_users_train.keys():
        np.random.shuffle(dict_users_train[idx])
    
    client_task = [[j for j in range(args.task)] for i in range(args.num_users)]
    for i in client_task:
        shuffle(i)

    # client_task = [[0, 1, 5, 6, 8, 2, 9, 7, 3, 4], 
    #                 [0, 1, 5, 6, 8, 2, 9, 7, 3, 4], 
    #                 [0, 1, 5, 6, 8, 2, 9, 7, 3, 4], 
    #                 [9, 7, 6, 1, 0, 4, 5, 3, 2, 8],
    #                 [9, 7, 6, 1, 0, 4, 5, 3, 2, 8], 
    #                 [9, 7, 6, 1, 0, 4, 5, 3, 2, 8], 
    #                 [6, 8, 3, 0, 7, 5, 4, 2, 1, 9], 
    #                 [6, 8, 3, 0, 7, 5, 4, 2, 1, 9],
    #                 [6, 8, 3, 0, 7, 5, 4, 2, 1, 9], 
    #                 [6, 8, 3, 0, 7, 5, 4, 2, 1, 9]]
    
    print(args.alg)
    # write = SummaryWriter('/data/lpyx/DisFed/ClientTrainAvg/log/GEM/GEM_high_20past_cd ' + args.dataset+'_'+'round' + str(args.round) + '_frac' + str(args.frac))
    write = SummaryWriter(f'/data/zxj/projects/vscodes/Dist-79/explogs-{args.dataset}/{args.alg}/' + args.dataset+'_' + args.model +'_' +'round' + str(args.round) + "_epoch_" + str(args.local_ep) + '_frac' + str(args.frac) + '_' + str(args.seed)
                          + "_" + str(int(time.time()))[-4:])
    # build model
    net_glob = get_model(args)
    net_glob.train()

    # training
    serverAgg = FedAvg(copy.deepcopy(net_glob.to(args.device)))
    indd = None  # indices of embedding for sent140
    loss_train = []
    accs = []
    times = []
    accs10 = 0
    accs10_glob = 0
    start = time.time()
    task=-1
    
    sample_shape = (3, 32, 32)
    apprs = [Appr(net_glob.to(args.device),sample_shape,100,10, args) for i in range(args.num_users)]
    
    print(args.round)
    w_globals = []
    for iter in range(args.epochs):
        if iter % (args.round) == 0:
            task+=1
            w_globals = []
        w_glob = {}
        loss_locals = []
        
        m = max(int(args.frac * args.num_users), 1)
        if iter == args.epochs:
            m = args.num_users
    
        all_users = [i for i in range(args.num_users)]
        times_in = []
        total_len = 0
        tr_dataloaders= None

        for ind, idx in enumerate(all_users):
            start_in = time.time()

            # tr_dataloaders = DataLoader(DatasetSplit(dataset_train[client_task[idx][task]],dict_users_train[idx][:args.m_ft],tran_task=[task,client_task[idx][task]]),batch_size=args.local_bs, shuffle=True)
            tr_dataloaders = DataLoader(DatasetSplit(dataset_train[client_task[idx][task]],dict_users_train[idx],tran_task=[task,client_task[idx][task]]),batch_size=args.local_bs, shuffle=True)

            appr = apprs[idx]

            # appr.set_model(net_local.to(args.device))
            last = iter == args.epochs

            local_model,loss, indd = LongLifeTrain(args,appr,tr_dataloaders,iter,idx)
            loss_locals.append(copy.deepcopy(loss))

        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train.append(loss_avg)

        acc_test, loss_test = test_img_local_all(None, args, dataset_test, dict_users_test,task,apprs=apprs,w_locals=None,return_all=False,write=write,round=iter,client_task=client_task)
        accs.append(acc_test)

        print('Round {:3d}, Train loss: {:.3f}, Test loss: {:.3f}, Test accuracy: {:.2f}'.format(
                iter, loss_avg, loss_test, acc_test))
        
        ## DisAgg
        # clients_global = []
        # for ind, idx in enumerate(all_users):
        #     temp = copy.deepcopy(all_users)
        #     temp.remove(idx)
            
        #     # 邻居模型
        #     cur_local_models = []
        #     idxs_users = np.random.choice(temp, args.neibour, replace=False)
        #     for idx_user in idxs_users:
        #         cur_local_models.append(apprs[idx_user].model.state_dict())
            
        #     # 这里需要同时把自己的模型也加进去
        #     cur_local_models.append(apprs[idx].model.state_dict())
        #     w_globals = serverAgg.update(cur_local_models)
        #     clients_global.append(copy.deepcopy(w_globals))
            
        # for ind,idx in enumerate(all_users):
        #     apprs[idx].model.load_state_dict(clients_global[idx])

    end = time.time()
    print(end - start)
