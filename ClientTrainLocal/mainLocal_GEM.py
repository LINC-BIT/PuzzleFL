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
from ClientTrainLocal.LongLifeMethod.Local_GEM import Appr,LongLifeTest,LongLifeTrain
from ClientTrain.AggModel.sixcnn import SixCNN
from torch.utils.data import DataLoader
from Agg.AggModel.sixcnn import SixCNN as KDmodel
import time
# from Agg.FedDag import FedDag
from randseed import set_rand

if __name__ == '__main__':
    # parse args
    
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
    
    # write = SummaryWriter('/data/lpyx/DisFed/ClientTrainLocal/log/EWC_our_nonepre/EWC_server_epoch10_high20_our_' + args.dataset+'_'+'round' + str(args.round) + '_frac' + str(args.frac))
    write = SummaryWriter(f'/data/zxj/projects/vscodes/Dist-79/explogs-{args.dataset}/{args.alg}/' + args.dataset+'_' + args.model +'_' +'round' + str(args.round) + "_epoch_" + str(args.local_ep) + '_frac' + str(args.frac) + '_' + str(args.seed)
                          + "_" + str(int(time.time()))[-4:])
    # build model
    # net_glob = get_model(args)
    # net_glob = SixCNN([3,32,32],outputsize=100)
    # net_glob.train()
    
    net_glob = get_model(args)
    net_glob.train()
    kd_model = copy.deepcopy(net_glob)

    # total_num_layers = len(net_glob.state_dict().keys())
    # print(net_glob.state_dict().keys())
    # net_keys = [*net_glob.state_dict().keys()]

    # specify the representation parameters (in w_glob_keys) and head parameters (all others)
 
    # if args.alg == 'fedavg' or args.alg == 'prox':
    #     w_glob_keys = []
    # if 'sent140' not in args.dataset:
    #     w_glob_keys = list(itertools.chain.from_iterable(w_glob_keys))

    # print(total_num_layers)
    # all_users = [i for i in range(args.num_users)]

    # # generate list of local models for each user
    # net_local_list = []
    # w_locals = {}
    # for user in range(args.num_users):
    #     w_local_dict = {}
    #     for key in net_glob.state_dict().keys():
    #         w_local_dict[key] = net_glob.state_dict()[key]
    #     w_locals[user] = w_local_dict

    # training
    indd = None  # indices of embedding for sent140
    loss_train = []
    accs = []
    times = []
    accs10 = 0
    accs10_glob = 0
    start = time.time()
    task=-1
    # kd_model = KDmodel([3,32,32],100)
    
    sample_shape = (3, 32, 32)
    apprs = [Appr(net_glob.to(args.device),sample_shape,100,10, args) for i in range(args.num_users)]
    
    # apprs = [Appr(copy.deepcopy(net_glob).to(args.device), None,lr=args.lr, nepochs=args.local_ep, args=args,kd_model=kd_model,client_task = client_task[i]) for i in range(args.num_users)]

    print(args.round)
    # serverAgg = FedDag(int(args.frac * args.num_users),int(args.frac * args.num_users * 5),datasize=[3,32,32],dataname='CIFAR100')
    w_globals = []
    for iter in range(args.epochs):
        
        if iter % (args.round) == 0:
            task+=1
            # for idx in all_users:
            #     apprs[idx].new_kd()
        
        w_glob = {}
        loss_locals = []
        
        m = max(int(args.frac * args.num_users), 1)
        if iter == args.epochs:
            m = args.num_users

        # if iter % (args.round) == args.round - 1:
        #     print("*"*100)
        #     print("Last Train")
        #     idxs_users = [i for i in range(args.num_users)]
        # else:
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        # w_keys_epoch = w_glob_keys
        times_in = []
        total_len = 0
        tr_dataloaders= None

        for ind, idx in enumerate(idxs_users):
            start_in = time.time()

            # tr_dataloaders = DataLoader(DatasetSplit(dataset_train[client_task[idx][task]],dict_users_train[idx][:args.m_ft],tran_task=[task,client_task[idx][task]]),batch_size=args.local_bs, shuffle=True)
            tr_dataloaders = DataLoader(DatasetSplit(dataset_train[client_task[idx][task]],dict_users_train[idx],tran_task=[task,client_task[idx][task]]),batch_size=args.local_bs, shuffle=True)
                # if args.epochs == iter:
                #     local = LocalUpdate(args=args, dataset=dataset_train[task], idxs=dict_users_train[idx][:args.m_ft])
                # else:
                #     local = LocalUpdate(args=args, dataset=dataset_train[task], idxs=dict_users_train[idx][:args.m_tr])

                # appr = Appr(net, sbatch=args.batch_size, lr=args.lr, nepochs=args.nepochs, args=args, log_name=log_name)


            appr = apprs[idx]
            # if len(w_globals) != 0:
            #     agg_client = [w['client'] for w in w_globals]
            #     if idx in agg_client:
            #         appr.cur_kd.load_state_dict(w_globals[agg_client.index(idx)]['model'])

            # appr.set_model(net_local.to(args.device))
            appr.set_trData(tr_dataloaders)
            local_model,loss, indd = LongLifeTrain(args,appr,tr_dataloaders,iter,idx)
            loss_locals.append(copy.deepcopy(loss))

        # if iter % args.round == args.round - 1:
        #     w_globals = []
        # else:
        #     w_globals = serverAgg.update(all_kd_models,task)


        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train.append(loss_avg)


        acc_test, loss_test = test_img_local_all(None, args, dataset_test, dict_users_test,task,apprs=apprs,w_locals=None,return_all=False,write=write,round=iter,client_task=client_task)
        accs.append(acc_test)

        print('Round {:3d}, Train loss: {:.3f}, Test loss: {:.3f}, Test accuracy: {:.2f}'.format(
                iter, loss_avg, loss_test, acc_test))
        # if iter % (args.round) == args.round - 1:
        #     for i in range(args.num_users):
        #         tr_dataloaders = DataLoader(
        #             DatasetSplit(dataset_train[client_task[i][task]], dict_users_train[i][:args.m_ft],
        #                          tran_task=[task, client_task[i][task]]), batch_size=args.local_bs, shuffle=True)
        #         client_state = apprs[i].prune_kd_model(tr_dataloaders,task)
        #         serverAgg.add_history(client_state)
        
        ## DisAgg
        # for ind, idx in enumerate(all_users):
        #     temp = copy.deepcopy(all_users)
        #     temp.remove(idx)
        #     cur_recive = apprs[idx].recive_class()
        #     cur_local_models = []
        #     idxs_users = np.random.choice(temp, args.neibour, replace=False)
        #     other_client_models = []
        #     for idx_user in idxs_users:
        #         other_send = apprs[idx_user].send_class()
        #         real_send = list(set(other_send) & set(cur_recive))
        #         real_send_model = apprs[idx_user].get_task_class(real_send)
        #         for rm in real_send_model:
        #             other_client_models.append(rm)
        #     apprs[idx].aggregation(other_client_models)
        
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
    # end = time.time()
    # print(end - start)
    # print(times)
    # print(accs)
    # base_dir = './save/EWC/0.4_dag_High10_ccs_Fedavg_lambda_'+str(args.lamb) +str('_') + args.alg + '_' + args.dataset + '_' + str(args.num_users) + '_' + str(
    #             args.shard_per_user) + '_iterFinal' + '_frac_'+str(args.frac)+ '.csv'
    # user_save_path = base_dir
    # accs = np.array(accs)
    # accs = pd.DataFrame(accs, columns=['accs'])
    # accs.to_csv(base_dir, index=False)
