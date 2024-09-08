#!/usr/bin/env bash

pwd

# baseline
# python ClientTrainAvg/mainAvg_EWC.py --alg=EWC_GOSSIP --model=tiny_vit --dataset=cifar100 --gpu=0 --seed=614

# python ClientTrainAvg/mainAvg_GEM.py --alg=GEM_GOSSIP_RandTask --model=cnn --dataset=cifar100 --gpu=1 --seed=614
# python ClientTrainAvg/mainAvg_EWC.py --alg=EWC_GOSSIP_lr5 --model=resnet --dataset=cifar100 --gpu=1 --seed=614 --lr=0.0005

# python ClientTrainAvg/mainAvg_GEM.py --alg=GEM_GOSSIP_lr5 --model=resnet --dataset=cifar100 --gpu=1 --seed=614


# python single/main_GEM.py --alg=GEM --model=tiny_vit --dataset=cifar100 --optim=AdamW --wd=0.05 --gpu=1 --n_memories=17 --log_dir==logs/fgcs/log-fgcs-cifar100 --log_ep=2 --seed=0 --comment=g_test_seed0


# mini EWC
# python ClientTrainAvg/mainAvg_EWC.py --alg=EWC_GOSSIP_raw_img_lr5 --model=mobinet --dataset=miniimagenet --gpu=1 --seed=614 --lr=0.0005
# python ClientTrainAvg/mainAvg_EWC.py --alg=EWC_GOSSIP_raw_img --model=dense --dataset=miniimagenet --gpu=1 --seed=614

# mini GEM
# python ClientTrainAvg/mainAvg_GEM.py --alg=GEM_GOSSIP --model=mobinet --dataset=miniimagenet --gpu=0 --seed=614
# python ClientTrainAvg/mainAvg_GEM.py --alg=GEM_GOSSIP --model=dense --dataset=miniimagenet --gpu=0 --seed=614

# TINY IMAGENET VIT
# python ClientTrainAvg/mainAvg_EWC.py --alg=EWC_GOSSIP --model=tiny_vit --dataset=tinyimagenet --seed=614 --epochs=120 --task=20 --m_ft=300 --gpu=0 

# TINY IMAGENET PIT
# python ClientTrainAvg/mainAvg_EWC.py --alg=EWC_GOSSIP --model=tiny_pit --dataset=tinyimagenet --seed=614 --epochs=120 --task=20 --m_ft=300 --gpu=0 

# multi clients
python ClientTrainAvg/mainAvg_EWC.py --alg=EWC_GOSSIP --model=cnn --dataset=cifar100 --seed=614 --gpu=1 --m_ft=500 --round=5 --local_ep=5 --shard_per_user=4 --epochs=50 --num_users=100 