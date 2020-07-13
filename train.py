import numpy as np
import paddle.fluid as fluid
from reader import episode_reader
import os
import program
from utils import mean_confidence_interval
import argparse

parser = argparse.ArgumentParser(description="Few Shot Visual Recognition")
parser.add_argument("--c_way", type=int, default=5)
parser.add_argument("--k_shot", type=int, default=1)
parser.add_argument("--query_num_per_class", type=int, default=15)
parser.add_argument("--episode", type=int, default=500000)
parser.add_argument("--test_episode", type=int, default=600)
parser.add_argument("--test_query_num_per_class", type=int, default=3)
parser.add_argument("--gpu", type=int, default=0)
# parser.add_argument("--train_path",type=str, default='/home/aistudio/work/mini_Imagenet/dataset/train')
# parser.add_argument("--val_path",type=str, default='/home/aistudio/work/mini_Imagenet/dataset/val')
parser.add_argument("--train_path", type=str, default='/Users/alex/baidu/mini_Imagenet/dataset/train')
parser.add_argument("--val_path", type=str, default='/Users/alex/baidu/mini_Imagenet/dataset/val')
args = parser.parse_args()

use_gpu = args.gpu
EPISODE_NUM = args.episode
TEST_EPISODE_NUM = args.test_episode
C_WAY = args.c_way
K_SHOT = args.k_shot
QUERY_NUM = args.query_num_per_class
TEST_NUM = args.test_query_num_per_class

TRAIN_PATH = args.train_path
TEST_PATH = args.val_path


def train():
    train_reader = episode_reader(TRAIN_PATH, EPISODE_NUM, C_WAY, K_SHOT, QUERY_NUM, augment=True)

    main_program = fluid.Program()
    startup_program = fluid.Program()
    train_build_outputs = program.build(main_program, startup_program, mode='train', c_way=C_WAY, k_shot=K_SHOT)
    train_feed_list = train_build_outputs[0]
    train_fetch_list = train_build_outputs[1]

    eval_program = fluid.Program()
    eval_build_outputs = program.build(eval_program, startup_program, mode='eval', c_way=C_WAY, k_shot=K_SHOT)
    eval_feed_list = eval_build_outputs[0]
    eval_fetch_list = eval_build_outputs[1]

    place = fluid.CUDAPlace(0) if use_gpu == 1 else fluid.CPUPlace()
    exe = fluid.Executor(place=place)
    exe.run(startup_program)

    episode = 0
    last_accuracy = 0.0
    accs = []
    for data in train_reader():
        sample_images = np.array(data[0]).astype('float32')
        query_images = np.array(data[2]).astype('float32')
        labels = np.array(data[3]).astype('int64')[:, np.newaxis]
        feed_dic = dict(zip(train_feed_list, [sample_images, query_images, labels]))
        res = exe.run(main_program,
                      feed=feed_dic,
                      fetch_list=train_fetch_list)
        predict_labels = np.argmax(res[0], axis=1)
        rewards = [1 if predict_labels[j] == labels[j] else 0 for j in range(len(labels))]
        acc = np.sum(rewards) / 1.0 / len(rewards)
        accs.append(acc)
        loss = res[1]
        if (episode + 1) % 100 == 0:
            print("episode:", episode + 1, "loss:", loss, 'acc:', sum(accs) / len(accs))

        if episode % 5000 == 0:
            print("Testing...")
            accuracies = []
            test_reader = episode_reader(TEST_PATH,
                                         TEST_EPISODE_NUM, C_WAY, K_SHOT, TEST_NUM)
            for data in test_reader():
                total_rewards = 0
                counter = 0
                sample_images = np.array(data[0]).astype('float32')
                query_images = np.array(data[2]).astype('float32')
                test_labels = np.array(data[3]).astype('int64')
                feed_dic = dict(zip(eval_feed_list, [sample_images, query_images]))
                res = exe.run(eval_program,
                              feed=feed_dic,
                              fetch_list=eval_fetch_list)
                predict_labels = np.argmax(res[0], axis=1)

                rewards = [1 if predict_labels[j] == test_labels[j] else 0 for j in range(query_images.shape[0])]

                total_rewards += np.sum(rewards)
                counter += query_images.shape[0]
                accuracy = total_rewards / 1.0 / counter
                accuracies.append(accuracy)

            test_accuracy, h = mean_confidence_interval(accuracies)

            print("test accuracy:", test_accuracy, "h:", h)

            if test_accuracy > last_accuracy:
                # save networks
                if not os.path.exists('model'):
                    os.makedirs('model')
                fluid.save(main_program, './model/best_accuracy')
                print("save networks for episode:", episode)
                last_accuracy = test_accuracy

        episode += 1


if __name__ == '__main__':
    train()
