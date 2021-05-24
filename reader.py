import os
import numpy as np
import random
from PIL import Image
from autoaugment import ImageNetPolicy
import glob


def get_class(label_map, sample):
    for c in label_map.keys():
        if sample in label_map[c]:
            return c
def episode_reader(path, episode_num, support_class_num, support_shot_num, query_num, augment=False):
    # class_list = os.listdir(path)
    with open(os.path.join(path,'train_labels.csv'), 'r') as f:
        lines = f.readlines()[1:]
    label_map = {}
    for line in lines:
        image_path, label = line.strip().split(',')
        if label in label_map.keys():
            label_map[label] += [image_path]
        else:
            label_map[label] = [image_path]
    class_list = label_map.keys()

    if augment:
        policy = ImageNetPolicy()
    def reader():
        for i in range(episode_num):
            labels = np.array(range(len(class_list)))
            labels = dict(zip(class_list, labels))

            sample_roots = []
            query_roots = []
            for c in class_list:
                random.shuffle(label_map[c])

                sample_roots += label_map[c][:support_shot_num]
                query_roots += label_map[c][support_shot_num:support_shot_num + query_num]

            # np.random.shuffle(sample_roots)
            np.random.shuffle(query_roots)
            sample_labels = [labels[get_class(label_map, x)] for x in sample_roots]
            query_labels = [labels[get_class(label_map, x)] for x in query_roots]
            sample_img = []
            query_img = []
            for im_file in sample_roots:
                im_file = os.path.join(path, 'images',im_file)
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
                im_file = os.path.join(path, 'images', im_file)
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

if __name__ == '__main__':
    reader = episode_reader("/Users/alex/Downloads/train", 10000, 4, 5, 10)
    for d in reader():
        pass
