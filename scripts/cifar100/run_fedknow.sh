#!/usr/bin/env bash

pwd

# DEBUG
# python FCL/FedKNOW/mainFedKNOW.py --alg=FedKNOW --model=resnet --dataset=cifar100 --gpu=0 --seed=614 --m_ft=100 --round=1 --local_ep=2
# python FCL/FedKNOW/mainFedKNOW.py --alg=FedKNOW --model=tiny_vit --dataset=tinyimagenet --gpu=0 --seed=614 --m_ft=100 --round=1 --local_ep=2 --task=20
# python FCL/FedKNOW/mainFedKNOW.py --alg=FedKNOW_debug --model=cnn --dataset=cifar100 --gpu=0 --seed=614 --m_ft=100 --round=1 --local_ep=5 --shard_per_user=8 --local_rep_ep=3
python FCL/FedKNOW/mainFedKNOW.py --alg=FedKNOW_k2 --model=cnn --dataset=cifar100 --gpu=0 --seed=614 --m_ft=20 --round=1 --local_ep=4 --shard_per_user=4 --local_rep_ep=2 --num_users=100


# CNN
# python FCL/FedKNOW/mainFedKNOW.py --alg=FedKNOW --model=cnn --dataset=cifar100 --gpu=0 --seed=614 --m_ft=500 --round=5 --local_ep=5 --shard_per_user=8

# ResNEt
# python FCL/FedKNOW/mainFedKNOW.py --alg=FedKNOW --model=resnet --dataset=cifar100 --gpu=1 --seed=614 --m_ft=500 --round=5 --local_ep=5 --shard_per_user=8
# python FCL/FedKNOW/mainFedKNOW.py --alg=FedKNOW_halfpack_k2 --model=resnet --dataset=cifar100 --gpu=0 --seed=614 --m_ft=500 --round=5 --local_ep=5 --shard_per_user=8 --local_rep_ep=3

# DENSE
# python FCL/FedKNOW/mainFedKNOW.py --alg=FedKNOW_halfpack_k2 --model=mobinet --dataset=miniimagenet --gpu=0 --seed=614 --m_ft=500 --round=5 --local_ep=5 --shard_per_user=8 --local_rep_ep=3

# Tiny VIT
# python FCL/FedKNOW/mainFedKNOW.py --alg=FedKNOW --model=tiny_pit --dataset=tinyimagenet --gpu=1 --seed=614 --m_ft=500 --round=5 --local_ep=5 --shard_per_user=8 --task=20  --epochs=120 

# Tiny PIT
# python FCL/FedKNOW/mainFedKNOW.py --alg=FedKNOW --model=tiny_pit --dataset=tinyimagenet --gpu=1 --seed=614 --m_ft=500 --round=5 --local_ep=5 --shard_per_user=8 --task=20  --epochs=120 
