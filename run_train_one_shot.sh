#!/bin/sh
python -u train.py --c_way 5 --k_shot 1 --query_num_per_class 15 \
--episode 100000 --test_episode 600 --test_query_num_per_class 3 \
--gpu 1 --train_path /home/aistudio/work/mini_Imagenet/dataset/train \
--val_path /home/aistudio/work/mini_Imagenet/dataset/val --model_path ./model/best_accuracy