#!/usr/bin/env bash

SAVE_ID=$1
CUDA_VISIBLE_DEVICES=0 python train.py --model_name asgcn --dataset lap14 --save True --save_name $SAVE_ID