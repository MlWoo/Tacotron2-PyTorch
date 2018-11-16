#!/bin/bash
CUDA_VISIBLE_DEVICES=$1 python train.py --data-root=dataset_mandarin/training_data/ --run-name="Tacotron2_beta_profile"

