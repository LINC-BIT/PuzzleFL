#!/usr/bin/env bash

pwd

# baseline

# cifar100
# python baselines/WEIT/mainWEIT.py --alg=FedWEIT --model=weit_resnet --dataset=cifar100 --gpu=0 --seed=614 --m_ft=500

# python baselines/WEIT/mainWEIT.py --alg=FedWEIT --model=weit_resnet --dataset=cifar100 --gpu=0 --seed=614 --m_ft=500
python baselines/WEIT/mainWEIT.py --alg=FedWEIT --model=weit_cnn --dataset=cifar100 --gpu=0 --seed=614 --m_ft=500 --round=5 --local_ep=5 --shard_per_user=4 --epochs=50 --num_users=100 


# CNN
# python baselines/WEIT/mainWEIT.py --alg=FedWEIT --model=weit_cnn --dataset=cifar100 --gpu=0 --seed=614 --m_ft=500
# python baselines/WEIT/mainWEIT.py --alg=FedWEIT --model=weit_cnn --dataset=cifar100 --gpu=1 --seed=614 --m_ft=500 --round=5 --local_ep=5 --shard_per_user=8

# RESN
# python baselines/WEIT/mainWEIT.py --alg=FedWEIT --model=weit_resnet --dataset=cifar100 --gpu=0 --seed=614 --m_ft=500
# python baselines/WEIT/mainWEIT.py --alg=FedWEIT --model=weit_resnet --dataset=cifar100 --gpu=1 --seed=614 --m_ft=500 --round=5 --local_ep=5 --shard_per_user=8

# MOBI
# python baselines/WEIT/mainWEIT.py --alg=FedWEIT --model=weit_mobinet --dataset=miniimagenet --gpu=0 --seed=614 --m_ft=300
# python baselines/WEIT/mainWEIT.py --alg=FedWEIT --model=weit_mobinet --dataset=miniimagenet --gpu=0 --seed=614 --m_ft=500 --round=5 --local_ep=5 --shard_per_user=8


# DENSE
# python baselines/WEIT/mainWEIT.py --alg=FedWEIT --model=weit_densenet --dataset=miniimagenet --gpu=0 --seed=614 --m_ft=300
# python baselines/WEIT/mainWEIT.py --alg=FedWEIT --model=weit_densenet --dataset=miniimagenet --gpu=1 --seed=614 --m_ft=500 --round=5 --local_ep=5 --shard_per_user=8

# PIT
# python baselines/WEIT/mainWEIT.py --alg=FedWEIT --model=weit_tiny_pit --dataset=tinyimagenet --seed=614 --epochs=120 --task=20 --m_ft=330 --gpu=0 
# python baselines/WEIT/mainWEIT.py --alg=FedWEIT --model=weit_tiny_pit --dataset=tinyimagenet --gpu=1 --seed=614 --m_ft=500 --round=5 --local_ep=5 --shard_per_user=8 --epochs=120 --task=20
# 
# VIT
# python baselines/WEIT/mainWEIT.py --alg=FedWEIT --model=weit_tiny_vit --dataset=tinyimagenet --seed=614 --epochs=120 --task=20 --m_ft=330 --gpu=0 
# python baselines/WEIT/mainWEIT.py --alg=FedWEIT --model=weit_tiny_vit --dataset=tinyimagenet --gpu=1 --seed=614 --m_ft=500 --round=5 --local_ep=5 --shard_per_user=8 --epochs=120 --task=20