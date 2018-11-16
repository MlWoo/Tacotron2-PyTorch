#!/bin/bash
#CUDA_VISIBLE_DEVICES=
python preprocess.py --dataset=mandarin_pre \
                     --input_txt='train_without_space.txt' \
	             --base_dir=./ \
		     --output_dir='dataset_mandarin'
