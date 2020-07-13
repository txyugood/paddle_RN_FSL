#!/bin/sh
python -u test.py --c_way 5 --k_shot 1 \
--episode 10 --test_episode 600 --test_query_num_per_class 3 \
--gpu 1 --test_path /home/aistudio/work/mini_Imagenet/dataset/test