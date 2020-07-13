import numpy as np
import paddle.fluid as fluid
from reader import episode_reader
import program
from utils import mean_confidence_interval

import argparse

parser = argparse.ArgumentParser(description="Few Shot Visual Recognition")
parser.add_argument("--c_way", type=int, default=5)
parser.add_argument("--k_shot", type=int, default=1)
parser.add_argument("--episode", type=int, default=10)
parser.add_argument("--test_episode", type=int, default=600)
parser.add_argument("--test_query_num_per_class", type=int, default=3)
parser.add_argument("--gpu", type=int, default=0)
# parser.add_argument("--test_path",type=str, default='/home/aistudio/work/mini_Imagenet/dataset/test')
parser.add_argument("--test_path", type=str, default='/Users/alex/baidu/mini_Imagenet/dataset/test')
parser.add_argument("--model_path", type=str, default='./model/best_accuracy')

args = parser.parse_args()


C_WAY = args.c_way
K_SHOT = args.k_shot
QUERY_NUM = args.test_query_num_per_class
TEST_PATH = args.test_path
MODEL_PATH = args.model_path
EPISODE_NUM = args.episode
TEST_EPISODE_NUM = args.test_episode
use_gpu = args.gpu

def test():
    main_program = fluid.Program()
    startup_program = fluid.Program()

    test_build_outputs = program.build(main_program, startup_program, mode='eval', c_way=C_WAY, k_shot=K_SHOT)
    test_feed_list = test_build_outputs[0]
    test_fetch_list = test_build_outputs[1]

    place = fluid.CUDAPlace(0) if use_gpu == 1 else fluid.CPUPlace()
    exe = fluid.Executor(place=place)
    exe.run(startup_program)
    fluid.load(main_program, MODEL_PATH, exe)

    total_accuracy = 0.0
    for episode in range(EPISODE_NUM):
        print("Testing...")
        accuracies = []
        test_reader = episode_reader(TEST_PATH,
                                     TEST_EPISODE_NUM, C_WAY, K_SHOT, QUERY_NUM)
        for data in test_reader():
            total_rewards = 0
            counter = 0
            sample_images = np.array(data[0]).astype('float32')
            query_images = np.array(data[2]).astype('float32')
            test_labels = np.array(data[3]).astype('int64')
            feed_dic = dict(zip(test_feed_list, [sample_images, query_images]))
            res = exe.run(main_program,
                          feed=feed_dic,
                          fetch_list=test_fetch_list)
            predict_labels = np.argmax(res[0],axis=1)

            rewards = [1 if predict_labels[j] == test_labels[j] else 0 for j in range(query_images.shape[0])]

            total_rewards += np.sum(rewards)
            counter += query_images.shape[0]
            accuracy = total_rewards / 1.0 / counter
            accuracies.append(accuracy)

        test_accuracy, h = mean_confidence_interval(accuracies)

        print("test accuracy:", test_accuracy, "h:", h)
        total_accuracy += test_accuracy
    print("aver_accuracy:",total_accuracy/EPISODE_NUM)

if __name__ == '__main__':
    test()