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
from NoAgg.LongLifeMethod.Avg_EWC import Appr,LongLifeTest,LongLifeTrain
from ClientTrain.AggModel.sixcnn import SixCNN
from torch.utils.data import DataLoader
import time
from Agg.FedAvg import FedAvg
from randseed import set_rand
if __name__ == '__main__':
    # parse args
    # args = args_parser()
    
    # seed
    args = args_parser()
    set_rand(args.seed)
    
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    lens = np.ones(args.num_users)
    dataset_train, dataset_test, dict_users_train, dict_users_test = get_data(args)
    for idx in dict_users_train.keys():
        np.random.shuffle(dict_users_train[idx])
    
    client_task=[[j for j in range(args.task)] for i in range(args.num_users)]
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
    
    # write = SummaryWriter('/data/lpyx/FedAgg/ClientTrainAvg/log/EWC/EWC_high20_dag_' + args.dataset+'_'+'round' + str(args.round) + '_frac' + str(args.frac))
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
    apprs = [Appr(copy.deepcopy(net_glob).to(args.device), None,lr=args.lr, nepochs=args.local_ep, args=args) for i in range(args.num_users)]
    
    print(args.round)
    serverAgg = FedAvg(copy.deepcopy(net_glob).to(args.device))
    w_globals = None
    
    for iter in range(args.epochs):
        if iter % (args.round) == 0:
            task+=1
            w_globals = []
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

        for ind, idx in enumerate(idxs_users):
            start_in = time.time()

            # tr_dataloaders = DataLoader(DatasetSplit(dataset_train[client_task[idx][task]],dict_users_train[idx][:args.m_ft],tran_task=[task,client_task[idx][task]]),batch_size=args.local_bs, shuffle=True)
            tr_dataloaders = DataLoader(DatasetSplit(dataset_train[client_task[idx][task]],dict_users_train[idx],tran_task=[task,client_task[idx][task]]),batch_size=args.local_bs, shuffle=True)
            appr = apprs[idx]


            # appr.set_model(net_local.to(args.device))
            appr.set_trData(tr_dataloaders)
            last = iter == args.epochs

            local_models,loss, indd = LongLifeTrain(args,appr,iter,None,idx)
            loss_locals.append(copy.deepcopy(loss))
          
        # Zuo: 这行注释掉，这个是把所有模型进行聚合，但DFL是只跟邻居进行聚合，因此参考PENS的更新方式
        # w_globals = serverAgg.update(all_local_models)


        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train.append(loss_avg)


        acc_test, loss_test = test_img_local_all(None, args, dataset_test, dict_users_test,task,apprs=apprs,w_locals=None,return_all=False,write=write,round=iter,client_task=client_task)
        accs.append(acc_test)


        print('Round {:3d}, Train loss: {:.3f}, Test loss: {:.3f}, Test accuracy: {:.2f}'.format(
                iter, loss_avg, loss_test, acc_test))

        # Zuo:注释掉，参考PENS的更新方式
        # if w_globals is not None:
        #     for i in range(args.num_users):
        #         apprs[i].model.load_state_dict(w_globals)
        #
        
        ## DisGOSSIP
        # clients_global = []
        # all_users = [i for i in range(args.num_users)]
        
        # for ind, idx in enumerate(all_users):
        #     temp = copy.deepcopy(all_users)
        #     temp.remove(idx)
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
        
        # if iter >= args.epochs - 10 and iter != args.epochs:
        #     accs10 += acc_test / 10
        # if iter >= args.epochs - 10 and iter != args.epochs:
        #     accs10_glob += acc_test / 10

            # model_save_path = './save/Baseline/0.4/accs_Fedavg_lambda_'+str(args.lamb) +str('_') + args.alg + '_' + args.dataset + '_' + str(args.num_users) + '_' + str(
            #     args.shard_per_user) + '_iter' + str(iter) + '_frac_'+str(args.frac)+'.pt'
            # torch.save(net_glob.state_dict(), model_save_path)

    # print('Average accuracy final 10 rounds: {}'.format(accs10))
    # if args.alg == 'fedavg' or args.alg == 'prox':
    #     print('Average global accuracy final 10 rounds: {}'.format(accs10_glob))
    end = time.time()
    print(end - start)
    print(times)
