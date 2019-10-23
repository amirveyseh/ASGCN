#!/usr/bin/env bash

SAVE_ID=$1
CUDA_VISIBLE_DEVICES=1 python train.py --model_name asgcn --dataset rest14 --save True --save_name $SAVE_ID