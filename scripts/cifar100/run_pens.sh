#!/usr/bin/env bash

pwd

# baseline
# python ClientTrainAvg/mainAvg_EWC.py --alg=EWC_GOSSIP --model=tiny_vit --dataset=cifar100 --gpu=0 --seed=614
# python single/main_GEM.py --alg=GEM --model=tiny_vit --dataset=cifar100 --optim=AdamW --wd=0.05 --gpu=1 --n_memories=17 --log_dir==logs/fgcs/log-fgcs-cifar100 --log_ep=2 --seed=0 --comment=g_test_seed0


# Resnet
# python ClientTrainPen/mainPen_EWC.py --alg=EWC_PENS --model=resnet --dataset=cifar100 --gpu=0 --seed=614 --m_ft=300

# DENSE NET
# python ClientTrainPen/mainPen_EWC.py --alg=EWC_PENS --model=dense --dataset=miniimagenet --gpu=0 --seed=614 --m_ft=300


# DENSE NET
# python ClientTrainPen/mainPen_EWC.py --alg=EWC_PENS --model=mobinet --dataset=miniimagenet --gpu=0 --seed=614 --m_ft=300

# DENSE NET
# python ClientTrainPen/mainPen_EWC.py --alg=EWC_PENS --model=tiny_vit --dataset=tinyimagenet --gpu=0 --seed=614 --m_ft=300


################################
# TINY IMAGENET VIT
# python ClientTrainPen/mainPen_EWC.py --alg=EWC_PENS --model=tiny_vit --dataset=tinyimagenet --seed=614 --epochs=120 --task=20 --m_ft=300 --gpu=1

# TINY IMAGENET PIT
# python ClientTrainPen/mainPen_EWC.py --alg=EWC_PENS --model=tiny_pit --dataset=tinyimagenet --seed=614 --epochs=120 --task=20 --m_ft=300 --gpu=1 

# multi clients
python ClientTrainPen/mainPen_EWC.py --alg=EWC_PENS --model=cnn --dataset=cifar100 --seed=614 --gpu=1 --m_ft=500 --round=5 --local_ep=5 --shard_per_user=4 --epochs=50 --num_users=100 