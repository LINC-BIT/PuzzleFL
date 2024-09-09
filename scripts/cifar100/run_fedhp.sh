#!/usr/bin/env bash

pwd

# baseline
# python ClientTrainOur/mainOur_EWC.py --alg=EWC_Our --model=tiny_vit --dataset=cifar100 --gpu=1 --seed=614
# python baselines/FedHP/mainFedHP.py --alg=GEM_FedHP --model=resnet --dataset=cifar100 --gpu=0 --seed=614
# python baselines/FedHP/mainFedHP.py --alg=GEM_FedHP --model=cnn --dataset=cifar100 --gpu=0 --seed=614


# res NET
# python baselines/FedHP/mainFedHP.py --alg=EWC_FedHP --model=resnet --dataset=cifar100 --gpu=0 --seed=614 --m_ft=300


# DENCE NET
# python baselines/FedHP/mainFedHP.py --alg=EWC_FedHP --model=dense --dataset=miniimagenet --gpu=0 --seed=614 --m_ft=300

# MOBI NET
# python baselines/FedHP/mainFedHP.py --alg=EWC_FedHP --model=mobinet --dataset=miniimagenet --gpu=0 --seed=614 --m_ft=300

# VIT NET
# python baselines/FedHP/mainFedHP.py --alg=EWC_FedHP --model=tiny_vit --dataset=tinyimagenet --seed=614 --epochs=120 --task=20 --m_ft=330 --gpu=0

# PIT NET
# python baselines/FedHP/mainFedHP.py --alg=EWC_FedHP --model=tiny_pit --dataset=tinyimagenet --seed=614 --epochs=120 --task=20 --m_ft=330 --gpu=0


# multi clients
 python baselines/FedHP/mainFedHP.py --alg=EWC_FedHP --model=cnn --dataset=cifar100 --seed=614 --gpu=1 --m_ft=500 --round=5 --local_ep=5 --shard_per_user=4 --epochs=50 --num_users=50 