import tensorflow as tf
import numpy as np
import os
import cv2
from utils.aug_utils import *

def load_list(list_path, image_root_path):
    images = []
    labels = []
    with open(list_path, 'r') as f:
        for line in f:
            line = line.replace('\n', '').split(' ')
            images.append(os.path.join(image_root_path, line[0]))
            labels.append(int(line[1]))
    return images, labels

def load_image(image_path, label, augment=False, crop_10=False):
    """
    In training, it is highly recommended to set the augment to true.
    In test, the standard 10-crop test [1] is provided for fair comparison.
    [1] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016.
    """
    image = cv2.imread(image_path.numpy().decode()).astype(np.float32)

    if augment:
        image = random_aspect(image)
        image = random_size(image)
        image = random_crop(image)
        image = random_flip(image)
        image = random_hsv(image)
        image = random_pca(image)
    else:
        image = random_size(image, target_size=256)
        if crop_10:
            image = test_10_crop(image)
        else:
            image = center_crop(image)

    image = normalize(image)

    label_one_hot = np.zeros(c.category_num)
    label_one_hot[label] = 1.0

    return image, label_one_hot

def train_iterator(list_path=c.train_list_path):
    images, labels = load_list(list_path, c.train_data_path)
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.shuffle(len(images))
    dataset = dataset.repeat()
    dataset = dataset.map(lambda x, y: tf.py_function(load_image, inp=[x, y, True, False], Tout=[tf.float32, tf.float32]),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(c.batch_size)
    it = dataset.__iter__()
    return it

def test_iterator(list_path=c.test_list_path):
    images, labels = load_list(list_path, c.test_data_path)
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.map(lambda x, y: tf.py_function(load_image, inp=[x, y, False, False], Tout=[tf.float32, tf.float32]),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(c.batch_size)
    it = dataset.__iter__()
    return it

def test_10_crop_iterator(list_path=c.test_list_path):
    images, labels = load_list(list_path, c.test_data_path)
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.map(lambda x, y: tf.py_function(load_image, inp=[x, y, False, True], Tout=[tf.float32, tf.float32]),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    it = dataset.__iter__()
    return it

if __name__ == '__main__':
    # Annotate the 'normalize' function for visualization
    # it = train_iterator()
    it = test_10_crop_iterator('../data/validation_label.txt')
    images, labels = it.next()
    # print(np.shape(images), np.shape(labels))
    for i in range(10):
        print(np.where(labels[i].numpy() != 0))
        cv2.imshow('show', images[i].numpy().astype(np.uint8))
        cv2.waitKey(0)
