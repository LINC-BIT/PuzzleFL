#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import math
import random
import numpy as np
import torch


def noniid(dataset, num_users, shard_per_user, num_classes, rand_set_all=[],dataname='cifar100'):
    """
    Sample non-I.I.D client data from MNIST dataset    (单个任务数据集)
    :param dataset:
    :param num_users:
    :return:
    """

    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}     # client number

    if len(rand_set_all) > 1:
        print('Setting test dataset')

    # 构建 label-data_idxs 字典
    idxs_dict = {}      # {label: [data_index]}
    count = 0           # length of dataset
    idx_count = {}      # for
    for i in range(len(dataset)):
        if dataname == 'miniimagenet' or dataname == 'FC100' or dataname == 'tinyimagenet':
            label = torch.tensor(dataset.data[i]['label']).item()
        elif dataname == 'Corn50':
            label = torch.tensor(dataset.data['label'][i]).item()
        else:
            label = torch.tensor(dataset.data[i][1]).item()
        if label < num_classes and label not in idxs_dict.keys():
            idxs_dict[label] = []
            idx_count[label] = 0    #
        if label < num_classes:
            idxs_dict[label].append(i)
            idx_count[label] += 1   #
            count += 1
    ##
    print(f'Label vs number dict is: {idx_count}')
    shard_per_class = int(shard_per_user * num_users / num_classes)     # 每个class多少个shard
    print('Shard number per class is: {}'.format(shard_per_class))
    samples_per_user = int(count/num_users)
    print('Sample number per user is: {}'.format(samples_per_user))

    # whether to sample more test samples per user
    if (samples_per_user < 10):     # changed by Zuo on 04-07; origin is 20
        double = True
    else:
        double = False

    # 生成每个类别对应的shard数据，比如 label 1 : [ shard1, shard2 ...]   每个shard中对应了样本的索引位置
    for label in idxs_dict.keys():
        x = idxs_dict[label]        # label 对应的所有数据的索引
        num_leftover = len(x) % shard_per_class
        leftover = x[-num_leftover:] if num_leftover > 0 else []
        x = np.array(x[:-num_leftover]) if num_leftover > 0 else np.array(x)    # 如果不能被shard_per_class整除，则从左边舍去余处的样本;   x.shape为 （500， ）
        x = x.reshape((shard_per_class, -1))    # shard_per_class x sample_per_shard
        x = list(x)

        for i, idx in enumerate(leftover):
            x[i] = np.concatenate([x[i], [idx]])
        idxs_dict[label] = x    # 每个label中，包含了N个shard的样本

    if len(rand_set_all) == 0:
        rand_set_all = list(range(num_classes)) * shard_per_class
        random.shuffle(rand_set_all)

        # Corn50在 50个任务，20个user，5个shard的情况下，只有 99 个shard，因此扩充至100
        # if dataname == 'Corn50':
        #     rand_set_all.append(rand_set_all[-1])

        rand_set_all = np.array(rand_set_all).reshape((num_users, -1))

    # divide and assign
    testb = False
    for i in range(num_users):
        # Corn50 的最后一个client不够分配
        # if dataname == 'Corn50' and i==19:
        if dataname == 'Corn50' and num_users == 20 and i == 19:
            dict_users[i] = np.copy(dict_users[i-1])
            break
            #

        if double:
            rand_set_label = list(rand_set_all[i]) * 50
        else:
            rand_set_label = rand_set_all[i]
        rand_set = []
        for label in rand_set_label:
            try:
                idx = np.random.choice(len(idxs_dict[label]), replace=False)
            except Exception as e:
                print(e)
            if (samples_per_user < 100 and testb):
                rand_set.append(idxs_dict[label][idx])
            else:
                rand_set.append(idxs_dict[label].pop(idx))
        dict_users[i] = np.concatenate(rand_set)

    print('Top 10 label per user dict is : {}'.format(rand_set_all[:10, ]))

    return dict_users, rand_set_all
