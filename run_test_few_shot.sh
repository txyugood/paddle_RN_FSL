#!/bin/sh
python -u test.py --c_way 5 --k_shot 5 \
--episode 10 --test_episode 600 --test_query_num_per_class 5 \
--gpu 1 --test_path /home/aistudio/work/mini_Imagenet/dataset/test \
--model_path ./model_few_shot/best_accuracy