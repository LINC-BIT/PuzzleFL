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
from baselines.FedViT.RepAGG import FedRepAVG
from baselines.FedKNOW.FedKNOW_Client import Appr, LongLifeTrain

from ClientTrain.AggModel.sixcnn import SixCNN
from torch.utils.data import DataLoader
import time


from randseed import set_rand
import os
import ClientTrain.AggModel.weitnet.consts as CONST

from baselines.FedKNOW.packnet.PacknetCNN import PackNetCNN
from baselines.FedKNOW.packnet.PacknetViT import PackNetViT


if __name__ == '__main__':
    # parse args
    args = args_parser()
    set_rand(args.seed)
    
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    os.environ['CUDA_VISIBLE_DEVICE'] = f'{args.gpu}'  # compatible to cuda()
    CONST.DEVICE = args.device
    
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

    ############### fedknow
    PackNet =  PackNetViT if args.model in ['tiny_vit', 'tiny_pit'] else PackNetCNN

    apprs = [
        Appr(
            model=copy.deepcopy(net_glob),  # RepTail
            packnet=PackNet(args.task, local_ep=int(args.local_ep / 2), local_rep_ep=args.local_rep_ep, device=args.device,  # TODO: packnet 开销降低
                            prune_instructions=0.95, net=net_glob),    # TODO: 存储率
            packmodel=copy.deepcopy(net_glob),  # RepTail
            tr_dataloader=None,
            lr=args.lr,
            nepochs=args.local_ep,
            args=args,
            cid=i
        ) for i in range(args.num_users)]
    ###############
    
    print(args.round)
    
    serverAgg = FedRepAVG(copy.deepcopy(net_glob).to(args.device))

    # serverAgg = FedAvg(copy.deepcopy(net_glob).to(args.device))
    for iter in range(args.epochs):
        all_users = [i for i in range(args.num_users)]
        
        if iter % (args.round) == 0:
            task+=1
            
        # client sampling
        m = max(int(args.frac * args.num_users), 1)
        if iter == args.epochs:
            m = args.num_users
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        
        times_in = []
        total_len = 0
        tr_dataloaders= None
        loss_locals = []

        for ind, idx in enumerate(idxs_users):
            start_in = time.time()

            tr_dataloaders = DataLoader(DatasetSplit(dataset_train[client_task[idx][task]],dict_users_train[idx][:args.m_ft],tran_task=[task,client_task[idx][task]]),batch_size=args.local_bs, shuffle=True)
            appr = apprs[idx]

            # appr.set_model(net_local.to(args.device))
            appr.set_trData(tr_dataloaders)
            last = iter == args.epochs

            local_w, _, loss, _ = LongLifeTrain(args, appr, task, write, idx)
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
            temp = copy.deepcopy(all_users)
            temp.remove(idx)
            cur_local_models = []
            idxs_users = np.random.choice(temp, args.neibour, replace=False)
            
            for idx_user in idxs_users:
                cur_local_models.append(apprs[idx_user].model.state_dict())
            
            # 这里需要同时把自己的模型也加进去
            cur_local_models.append(apprs[idx].model.state_dict())
            
            w_globals = serverAgg.update(cur_local_models)
            clients_global.append(copy.deepcopy(w_globals))
        
        for ind,idx in enumerate(all_users):
            for n, p in apprs[idx].model.named_parameters():
                if 'last' not in n:
                    p.data = clients_global[ind][n]
        
    end = time.time()
    print(end - start)
    print(times)
