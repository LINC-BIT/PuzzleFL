#!/usr/bin/env bash

pwd

# baseline
# python PuzzleFL/main_PuzzleFL.py --alg=EWC_Our --model=tiny_vit --dataset=cifar100 --gpu=1 --seed=614

python PuzzleFL/main_PuzzleFL.py --alg=EWC_Our --model=cnn --dataset=cifar100 --gpu=0 --seed=614
