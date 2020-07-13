#!/bin/sh
python -u train.py --c_way 5 --k_shot 5 --query_num_per_class 10 \
--episode 100000 --test_episode 600 --test_query_num_per_class 5 \
--gpu 1 --train_path /home/aistudio/work/mini_Imagenet/dataset/train \
--val_path /home/aistudio/work/mini_Imagenet/dataset/val