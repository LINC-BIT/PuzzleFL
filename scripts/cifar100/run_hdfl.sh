#!/usr/bin/env bash

pwd

# baseline
# python ClientTrainOur/mainOur_EWC.py --alg=EWC_Our --model=tiny_vit --dataset=cifar100 --gpu=1 --seed=614
# python Infocome/HDFL/mainHDFL.py --alg=EWC_HDFL --model=cnn --dataset=cifar100 --gpu=1 --seed=614 --epochs=10  --round=1 --local_ep=1
# python Infocome/HDFL/mainHDFL.py --alg=GEM_HDFL_RT --model=cnn --dataset=cifar100 --gpu=0 --seed=614
# python Infocome/HDFL/mainHDFL.py --alg=GEM_HDFL --model=cnn --dataset=cifar100 --gpu=1 --seed=614


# python single/main_GEM.py --alg=GEM --model=tiny_vit --dataset=cifar100 --optim=AdamW --wd=0.05 --gpu=1 --n_memories=17 --log_dir==logs/fgcs/log-fgcs-cifar100 --log_ep=2 --seed=0 --comment=g_test_seed0



# mini EWC
# python Infocome/HDFL/mainHDFL.py --alg=EWC_HDFL --model=dense --dataset=miniimagenet --gpu=0 --seed=614 --m_ft=300

# mini EWC
# python Infocome/HDFL/mainHDFL.py --alg=EWC_HDFL --model=mobinet --dataset=miniimagenet --gpu=0 --seed=614 --m_ft=300

# mini EWC
# python Infocome/HDFL/mainHDFL.py --alg=EWC_HDFL --model=dense --dataset=miniimagenet --gpu=0 --seed=614 --m_ft=300

# Tiny PIT
# python Infocome/HDFL/mainHDFL.py --alg=EWC_HDFL --model=tiny_pit --dataset=tinyimagenet --seed=614 --epochs=120 --task=20 --m_ft=330 --gpu=0 

# Tiny vit
# python Infocome/HDFL/mainHDFL.py --alg=EWC_HDFL --model=tiny_vit --dataset=tinyimagenet --seed=614 --epochs=120 --task=20 --m_ft=330 --gpu=0 

# multi clients
 python Infocome/HDFL/mainHDFL.py --alg=EWC_HDFL --model=cnn --dataset=cifar100 --seed=614 --gpu=0 --m_ft=500 --round=5 --local_ep=5 --shard_per_user=4 --epochs=50 --num_users=50 