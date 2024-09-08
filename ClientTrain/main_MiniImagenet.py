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
from ClientTrain.models.test import test_img_local_all, test_img_local_all_channel
from ClientTrain.LongLifeMethod.EWC import Appr as EWC_Appr
from ClientTrain.LongLifeMethod.EWC import LongLifeTrain as EWC_LongLifeTrain
from ClientTrain.LongLifeMethod.MAS import Appr as MAS_Appr
from ClientTrain.LongLifeMethod.MAS import LongLifeTrain as MAS_LongLifeTrain
from ClientTrain.LongLifeMethod.GEM import Appr as GEM_Appr
from ClientTrain.LongLifeMethod.GEM import LongLifeTrain as GEM_LongLifeTrain
from ClientTrain.LongLifeMethod.FedKNOW import Appr as FedKNOW_Appr
from ClientTrain.LongLifeMethod.MAS import LongLifeTrain as FedKNOW_LongLifeTrain
from ClientTrain.LongLifeMethod.Packnet import Appr as Packnet_Appr
from ClientTrain.LongLifeMethod.Packnet import LongLifeTrain as Packnet_LongLifeTrain
from ClientTrain.LongLifeMethod.ChannelGate import Appr as ChannelGate_Appr
from ClientTrain.LongLifeMethod.ChannelGate import LongLifeTrain as ChannelGate_LongLifeTrain
from ClientTrain.models.ChannelGatemodel.model import ChannelGatedCL

from ClientTrain.AggModel.sixcnn import SixCNN
from torch.utils.data import DataLoader
from Agg.AggModel.sixcnn import SixCNN as KDmodel
import time
from Agg.FedDag import FedDag
from ClientTrain.models.Packnet import PackNet
import copy
from ClientTrain.config import cfg
from ClientTrain.AggModel.resnet import Resnet18
if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    args.dataset = 'miniimagenet'
    lens = np.ones(args.num_users)
    dataset_train, dataset_test, dict_users_train, dict_users_test = get_data(args)
    # dict_users_test = [copy.deepcopy(dict_users_test) for i in range(2) for dict_user in dict_users_test]
    for idx in dict_users_train.keys():
        np.random.shuffle(dict_users_train[idx])
    client_task=[[j for j in range(args.task)] for i in range(args.num_users)]
    for i in client_task:
        shuffle(i)

    print(args.alg)
    write = SummaryWriter('/data/lpyx/FedAgg/ClientTrain/log/Miniimagenet/'+args.clmethod+'/server_epoch10_high20_dag_' + args.dataset+'_'+'round' + str(args.round) + '_frac' + str(args.frac))
    # build model
    # net_glob = get_model(args)
    net_glob = Resnet18([3, 32, 32], outputsize=100)
    net_glob.train()
    net_glob_cl = ChannelGatedCL(in_ch=cfg.IN_CH, out_dim=cfg.OUT_DIM,
                   conv_ch=cfg.CONV_CH,
                   sparsity_patience_epochs=cfg.SPARSITY_PATIENCE_EPOCHS,
                   lambda_sparse=cfg.LAMBDA_SPARSE,
                   freeze_fixed_proc=cfg.FREEZE_FIXED_PROC,
                   freeze_top_proc=cfg.FREEZE_TOP_PROC,
                   freeze_prob_thr=cfg.FREEZE_PROB_THR).to(cfg.DEVICE)
    total_num_layers = len(net_glob.state_dict().keys())
    print(net_glob.state_dict().keys())
    net_keys = [*net_glob.state_dict().keys()]

    # specify the representation parameters (in w_glob_keys) and head parameters (all others)
    if args.alg == 'fedrep' or args.alg == 'fedper':
        if 'cifar' in args.dataset or 'miniimagenet' in args.dataset:
            # w_glob_keys = [[k] for k,_ in net_glob.feature_net.named_parameters()]
            w_glob_keys = [net_glob.weight_keys[i] for i in [j for j in range(len(net_glob.weight_keys))]]
        elif 'mnist' in args.dataset:
            w_glob_keys = [net_glob.weight_keys[i] for i in [0, 1, 2]]
        elif 'sent140' in args.dataset:
            w_glob_keys = [net_keys[i] for i in [0, 1, 2, 3, 4, 5]]
        else:
            w_glob_keys = net_keys[:-2]
    elif args.alg == 'lg':
        if 'cifar' in args.dataset:
            w_glob_keys = [net_glob.weight_keys[i] for i in [1, 2]]
        elif 'mnist' in args.dataset:
            w_glob_keys = [net_glob.weight_keys[i] for i in [2, 3]]
        elif 'sent140' in args.dataset:
            w_glob_keys = [net_keys[i] for i in [0, 6, 7]]
        else:
            w_glob_keys = net_keys[total_num_layers - 2:]

    if args.alg == 'fedavg' or args.alg == 'prox':
        w_glob_keys = []
    if 'sent140' not in args.dataset:
        w_glob_keys = list(itertools.chain.from_iterable(w_glob_keys))

    print(total_num_layers)
    if args.alg == 'fedrep' or args.alg == 'fedper' or args.alg == 'lg':
        num_param_glob = 0
        num_param_local = 0
        for key in net_glob.state_dict().keys():
            num_param_local += net_glob.state_dict()[key].numel()
            print(num_param_local)
            if key in w_glob_keys:
                num_param_glob += net_glob.state_dict()[key].numel()
        percentage_param = 100 * float(num_param_glob) / num_param_local
        print('# Params: {} (local), {} (global); Percentage {:.2f} ({}/{})'.format(
            num_param_local, num_param_glob, percentage_param, num_param_glob, num_param_local))
    print("learning rate, batch size: {}, {}".format(args.lr, args.local_bs))

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
    kd_model = KDmodel([3,32,32],100)
    if args.clmethod == 'EWC':
        apprs = [EWC_Appr(copy.deepcopy(net_glob).to(args.device), None,lr=args.lr, nepochs=args.local_ep, args=args,kd_model=kd_model) for i in range(args.num_users)]
    elif args.clmethod == 'MAS':
        apprs = [MAS_Appr(copy.deepcopy(net_glob).to(args.device), None, lr=args.lr, nepochs=args.local_ep, args=args,
                      kd_model=kd_model) for i in range(args.num_users)]
    elif args.clmethod == 'GEM':
        apprs = [GEM_Appr(net_glob.to(args.device), kd_model, 3 * 32 * 32, 100, 10, args) for i in range(args.num_users)]
    elif args.clmethod == 'FedKNOW':
        apprs = [FedKNOW_Appr(copy.deepcopy(net_glob),PackNet(args.task,device=args.device),copy.deepcopy(net_glob), None,lr=args.lr, nepochs=args.local_ep, args=args,kd_model=kd_model) for i in range(args.num_users)]
    elif args.clmethod == 'Packnet':
        apprs = [Packnet_Appr(copy.deepcopy(net_glob).to(args.device), None, lr=args.lr, nepochs=args.local_ep, args=args,
                      kd_model=kd_model) for i in range(args.num_users)]
    elif args.clmethod == 'ChannelGate':
        apprs = [ChannelGate_Appr(copy.deepcopy(net_glob_cl).to(args.device), None,lr=args.lr, nepochs=args.local_ep, args=args,kd_model=kd_model) for i in range(args.num_users)]
    print(args.round)
    serverAgg = FedDag(int(args.frac * args.num_users),int(args.frac * args.num_users * 5),datasize=[3,32,32],dataname='miniimagenet')
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

        if iter % (args.round) == args.round - 1:
            print("*"*100)
            print("Last Train")
            idxs_users = [i for i in range(args.num_users)]
        else:
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        w_keys_epoch = w_glob_keys
        times_in = []
        total_len = 0
        tr_dataloaders= None
        all_kd_models = []

        for ind, idx in enumerate(idxs_users):
            start_in = time.time()

            tr_dataloaders = DataLoader(DatasetSplit(dataset_train[client_task[idx][task]],dict_users_train[idx],tran_task=[task,client_task[idx][task]]),batch_size=args.local_bs, shuffle=True)
                # if args.epochs == iter:
                #     local = LocalUpdate(args=args, dataset=dataset_train[task], idxs=dict_users_train[idx][:args.m_ft])
                # else:
                #     local = LocalUpdate(args=args, dataset=dataset_train[task], idxs=dict_users_train[idx][:args.m_tr])

                # appr = Appr(net, sbatch=args.batch_size, lr=args.lr, nepochs=args.nepochs, args=args, log_name=log_name)


            appr = apprs[idx]
            if len(w_globals) != 0:
                agg_client = [w['client'] for w in w_globals]
                if idx in agg_client:
                    appr.cur_kd.load_state_dict(w_globals[agg_client.index(idx)]['model'])

            # appr.set_model(net_local.to(args.device))
            appr.set_trData(tr_dataloaders)
            last = iter == args.epochs

            if args.clmethod == 'EWC':
                kd_models,loss, indd = EWC_LongLifeTrain(args,appr,iter,None,idx)
            elif args.clmethod == 'MAS':
                kd_models, loss, indd = MAS_LongLifeTrain(args, appr, iter, None, idx)
            elif args.clmethod == 'GEM':
                kd_models, loss, indd = GEM_LongLifeTrain(args, appr, tr_dataloaders, iter, idx)
            elif args.clmethod == 'FedKNOW':
                kd_models, loss, indd = FedKNOW_LongLifeTrain(args,appr,iter,None,idx)
            elif args.clmethod == 'Packnet':
                kd_models, loss, indd = Packnet_LongLifeTrain(args, appr, iter, None, idx)
            elif args.clmethod == 'ChannelGate':
                kd_models, loss, indd = ChannelGate_LongLifeTrain(args, appr, iter, None, idx)

            all_kd_models.append({'client':idx, 'models': kd_models})

            loss_locals.append(copy.deepcopy(loss))
        if iter % args.round == args.round - 1:
            w_globals = []
        else:
            w_globals = serverAgg.update(all_kd_models,task)


        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train.append(loss_avg)
        if args.clmethod == 'Packnet':
            acc_test, loss_test = test_img_local_all(None, args, dataset_test, dict_users_test,task,apprs=apprs,w_locals=None,return_all=False,write=write,round=iter,client_task=client_task,spe=True)
        elif args.clmethod == 'ChannelGate':
            acc_test, loss_test = test_img_local_all_channel(None, args, dataset_test, dict_users_test, task,
                                                             apprs=apprs, w_locals=None, return_all=False, write=write,
                                                             round=iter, client_task=client_task)
        else:
            acc_test, loss_test = test_img_local_all(None, args, dataset_test, dict_users_test,task,apprs=apprs,w_locals=None,return_all=False,write=write,round=iter,client_task=client_task)
        accs.append(acc_test)

        print('Round {:3d}, Train loss: {:.3f}, Test loss: {:.3f}, Test accuracy: {:.2f}'.format(
                iter, loss_avg, loss_test, acc_test))
        if iter % (args.round) == args.round - 1:
            for i in range(args.num_users):
                tr_dataloaders = DataLoader(
                    DatasetSplit(dataset_train[client_task[i][task]], dict_users_train[i][:args.m_ft],
                                 tran_task=[task, client_task[i][task]]), batch_size=args.local_bs, shuffle=True)
                client_state = apprs[i].prune_kd_model(tr_dataloaders,task)
                serverAgg.add_history(client_state)
