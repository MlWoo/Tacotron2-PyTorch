#!/bin/bash
CUDA_VISIBLE_DEVICES=$1 python train.py --data-root=/home/wumenglin/repo/deepvoice3_pytorch/data/training_data_mandarin/ --run-name="Tacotron2_beta"

