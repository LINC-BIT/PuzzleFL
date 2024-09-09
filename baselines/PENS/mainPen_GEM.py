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
from PENS.LongLifeMethod.Pen_GEM import Appr,LongLifeTrain
from ClientTrain.models.Nets import RepTail
from torch.utils.data import DataLoader
import time
from Agg.FedAvg import FedAvg
from randseed import set_rand
if __name__ == '__main__':
    
    # parse args
    # set_rand()
    # args = args_parser()
    
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
    # write = SummaryWriter('/data/lpyx/DisFed/ClientTrainPen/log/GEM/GEM_high_20past_cd ' + args.dataset+'_'+'round' + str(args.round) + '_frac' + str(args.frac))
    write = SummaryWriter(f'/data/zxj/projects/vscodes/Dist-79/explogs-{args.dataset}/{args.alg}/' + args.dataset+'_' + args.model +'_' +'round' + str(args.round) + "_epoch_" + str(args.local_ep) + '_frac' + str(args.frac) + '_' + str(args.seed)
                          + "_" + str(int(time.time()))[-4:])
    
    # build model
    net_glob = get_model(args)
    # net_glob = RepTail([3, 32, 32])
    net_glob.train()


    # total_num_layers = len(net_glob.state_dict().keys())
    # print(net_glob.state_dict().keys())
    # net_keys = [*net_glob.state_dict().keys()]

    # # specify the representation parameters (in w_glob_keys) and head parameters (all others)
    # if args.alg == 'fedrep' or args.alg == 'fedper':
    #     if 'cifar' in args.dataset:
    #         # w_glob_keys = [[k] for k,_ in net_glob.feature_net.named_parameters()]
    #         w_glob_keys = [net_glob.weight_keys[i] for i in [j for j in range(14)]]
    #     elif 'mnist' in args.dataset:
    #         w_glob_keys = [net_glob.weight_keys[i] for i in [0, 1, 2]]
    #     elif 'sent140' in args.dataset:
    #         w_glob_keys = [net_keys[i] for i in [0, 1, 2, 3, 4, 5]]
    #     else:
    #         w_glob_keys = net_keys[:-2]
    # elif args.alg == 'lg':
    #     if 'cifar' in args.dataset:
    #         w_glob_keys = [net_glob.weight_keys[i] for i in [1, 2]]
    #     elif 'mnist' in args.dataset:
    #         w_glob_keys = [net_glob.weight_keys[i] for i in [2, 3]]
    #     elif 'sent140' in args.dataset:
    #         w_glob_keys = [net_keys[i] for i in [0, 6, 7]]
    #     else:
    #         w_glob_keys = net_keys[total_num_layers - 2:]

    # if args.alg == 'fedavg' or args.alg == 'prox':
    #     w_glob_keys = []


    # print(total_num_layers)
    # print(w_glob_keys)
    # print(net_keys)
    # if args.alg == 'fedrep' or args.alg == 'fedper' or args.alg == 'lg':
    #     num_param_glob = 0
    #     num_param_local = 0
    #     for key in net_glob.state_dict().keys():
    #         num_param_local += net_glob.state_dict()[key].numel()
    #         print(num_param_local)
    #         if key in w_glob_keys:
    #             num_param_glob += net_glob.state_dict()[key].numel()
    #     percentage_param = 100 * float(num_param_glob) / num_param_local
    #     print('# Params: {} (local), {} (global); Percentage {:.2f} ({}/{})'.format(
    #         num_param_local, num_param_glob, percentage_param, num_param_glob, num_param_local))
    # print("learning rate, batch size: {}, {}".format(args.lr, args.local_bs))

    # generate list of local models for each user
    # net_local_list = []
    # w_locals = {}
    # for user in range(args.num_users):
    #     w_local_dict = {}
    #     for key in net_glob.state_dict().keys():
    #         w_local_dict[key] = net_glob.state_dict()[key]
    #     w_locals[user] = w_local_dict

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

        # if iter % (args.round) == args.round - 1:
        #     print("*"*100)
        #     print("Last Train")
        #     idxs_users = [i for i in range(args.num_users)]
        # else:
        all_users = [i for i in range(args.num_users)]
        # w_keys_epoch = w_glob_keys
        times_in = []
        total_len = 0
        tr_dataloaders= None
        # all_local_models = []
        
        Client_tr_dataloaders = []

        for ind, idx in enumerate(all_users):
            start_in = time.time()

            tr_dataloaders = DataLoader(DatasetSplit(dataset_train[client_task[idx][task]],dict_users_train[idx],tran_task=[task,client_task[idx][task]]),batch_size=args.local_bs, shuffle=True)
            # tr_dataloaders = DataLoader(DatasetSplit(dataset_train[client_task[idx][task]],dict_users_train[idx][:args.m_ft],tran_task=[task,client_task[idx][task]]),batch_size=args.local_bs, shuffle=True)
            Client_tr_dataloaders.append(tr_dataloaders)
                # if args.epochs == iter:
                #     local = LocalUpdate(args=args, dataset=dataset_train[task], idxs=dict_users_train[idx][:args.m_ft])
                # else:
                #     local = LocalUpdate(args=args, dataset=dataset_train[task], idxs=dict_users_train[idx][:args.m_tr])

                # appr = Appr(net, sbatch=args.batch_size, lr=args.lr, nepochs=args.nepochs, args=args, log_name=log_name)


            appr = apprs[idx]


            # appr.set_model(net_local.to(args.device))
            last = iter == args.epochs
            
            local_model,loss, indd = LongLifeTrain(args,appr,tr_dataloaders,iter,idx)
            loss_locals.append(copy.deepcopy(loss))
            # all_local_models.append(local_model)

        
       
            
        # if iter % args.round == args.round - 1:
        #     w_globals = None
        # else:
        #     w_globals = serverAgg.update(all_local_models)


        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train.append(loss_avg)



        acc_test, loss_test = test_img_local_all(None, args, dataset_test, dict_users_test,task,apprs=apprs,w_locals=None,return_all=False,write=write,round=iter,client_task=client_task)
        accs.append(acc_test)

        print('Round {:3d}, Train loss: {:.3f}, Test loss: {:.3f}, Test accuracy: {:.2f}'.format(
                iter, loss_avg, loss_test, acc_test))
        
         ## DisPen
        clients_global = []
        for ind, idx in enumerate(all_users):
            temp = copy.deepcopy(all_users)
            temp.remove(idx)
            cur_local_models = []
            other_similarity = []
            for other in temp:
                _,temp_acc = apprs[idx].validTest(iter//args.round ,Client_tr_dataloaders[idx],model = apprs[other].model)
                other_similarity.append(temp_acc)
            
            
            import pandas as pd
            idxs_users = pd.Series(other_similarity).sort_values(ascending = False).index[:args.neibour]
                
            for idx_user in idxs_users:
                cur_local_models.append(apprs[idx_user].model.state_dict())
            
            # 这里需要同时把自己的模型也加进去
            cur_local_models.append(apprs[idx].model.state_dict())
            
            w_globals = serverAgg.update(cur_local_models)
            clients_global.append(copy.deepcopy(w_globals))
        
        for ind,idx in enumerate(all_users):
            apprs[idx].model.load_state_dict(clients_global[idx])

        #
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
