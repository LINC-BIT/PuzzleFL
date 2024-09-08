#!/usr/bin/env bash

pwd

# baseline
# TINY IMAGENET VIT
python FCL/FedViT/mainFedViT.py --alg=FedViT_sr0.5_k2_rep3_mem4 --model=tiny_vit --dataset=tinyimagenet --seed=614 --epochs=120 --task=20 --m_ft=500 --gpu=0 --n_memories=2 --round=5 --local_ep=5 --shard_per_user=8 --local_rep_ep=3

# TINY IMAGENET PIT
# python FCL/FedViT/mainFedViT.py --alg=FedViT --model=tiny_pit --dataset=tinyimagenet --seed=614 --epochs=120 --task=20 --m_ft=300 --gpu=1 --n_memories=10
# python FCL/FedViT/mainFedViT.py --alg=FedViT --model=tiny_pit --dataset=tinyimagenet --seed=614 --epochs=120 --task=20 --m_ft=500 --gpu=0 --n_memories=40 --round=5 --local_ep=5 --shard_per_user=8
