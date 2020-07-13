import os
import numpy as np
import random
from PIL import Image
from autoaugment import ImageNetPolicy
import glob

def get_class(sample):
    return sample.split('/')[-2]
def episode_reader(path, episode_num, support_class_num, support_shot_num, query_num, augment=False):
    # class_list = os.listdir(path)
    class_list = glob.glob(path + '/n*')
    class_list = [os.path.join(path, c) for c in class_list]
    if augment:
        policy = ImageNetPolicy()
    def reader():
        for i in range(episode_num):
            class_folders = random.sample(class_list, support_class_num)
            labels = np.array(range(len(class_folders)))
            labels = dict(zip([f.split('/')[-1] for f in class_folders], labels))
            samples = dict()

            sample_roots = []
            query_roots = []
            for c in class_folders:
                temp = [os.path.join(c, x) for x in os.listdir(c)]
                samples[c] = random.sample(temp, len(temp))
                random.shuffle(samples[c])

                sample_roots += samples[c][:support_shot_num]
                query_roots += samples[c][support_shot_num:support_shot_num + query_num]

            # np.random.shuffle(sample_roots)
            np.random.shuffle(query_roots)
            sample_labels = [labels[get_class(x)] for x in sample_roots]
            query_labels = [labels[get_class(x)] for x in query_roots]
            sample_img = []
            query_img = []
            for im_file in sample_roots:
                img = Image.open(im_file)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                if augment:
                    img = policy(img)
                image = np.array(img)

                image = np.transpose(image, [2, 0, 1])
                image = image / 255.0
                mean = [0.485, 0.456, 0.406]
                std = [0.229, 0.224, 0.225]
                img_mean = np.array(mean).reshape((3, 1, 1))
                img_std = np.array(std).reshape((3, 1, 1))
                image -= img_mean
                image /= img_std
                sample_img.append(image.astype('float32'))

            for im_file in query_roots:
                img = Image.open(im_file)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                if augment:
                    img = policy(img)
                image = np.array(img)

                image = np.transpose(image, [2, 0, 1])
                image = image / 255.0
                mean = [0.485, 0.456, 0.406]
                std = [0.229, 0.224, 0.225]
                img_mean = np.array(mean).reshape((3, 1, 1))
                img_std = np.array(std).reshape((3, 1, 1))
                image -= img_mean
                image /= img_std
                query_img.append(image.astype('float32'))
            yield sample_img, sample_labels, query_img, query_labels

    return reader