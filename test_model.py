#!/usr/bin/env python
# -*- coding_utf-8 -*-
"""
    Author: Zhenghan Lee
    Date: 2018/07/21
"""
import os
import numpy as np
import random
import tensorflow as tf
from datetime import datetime
from itertools import combinations, chain

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def Prex_img(cls,img_name):
    """
    Add root_path to each img

    e.g. root_path = '/Users/lees/Desktop/img_folder/cls_name
         img = 1.png
         then return: /Users/lees/Desktop/img_folder/cls_name/1.png

    """
    return os.path.join(cls,img_name)

def sample_test_data(test_data_path):
    """
    sample test data and return combinations of sampled_test_data and label of each pair of img,
    so, len(sampled_test_data) is twice more than len(label)

    """
    test_data =[]
    test_label = []
    num = 0
    root_path = os.path.expanduser(test_data_path)
    cls_dir = os.listdir(root_path)
    for cls in cls_dir:
        # walk through the cls_dir
        if cls != '.DS_Store':
            img_list = os.listdir(os.path.join(root_path, cls))
            img_data = list(map(Prex_img, [os.path.join(root_path, cls)] * len(img_list), img_list))
            test_data += img_data

    test_data = np.array(test_data)
    np.random.shuffle(test_data)
    sampled_test_data = list(chain(*combinations(test_data, 2)))

    while num != len(sampled_test_data):
        img1_label = sampled_test_data[num].split('/')[-2]
        img2_label = sampled_test_data[num+1].split('/')[-2]
        if img1_label == img2_label:
            test_label.append(1)
        else:
            test_label.append(0)
        num += 2

    return sampled_test_data, test_label

def calcu_accuracy(dist, label, threshold):
    correct = 0
    for idx, i in enumerate(dist):
        if i > threshold:
            label_ = 0
        else:
            label_ = 1
        if label[idx] == label_:
            correct += 1
    return correct / len(dist)

def preprocessing_image_data(img_path):
    """ Preprocess image data with per_image_standardization and resize image to (160,160) """
    file = tf.read_file(img_path)
    image = tf.image.decode_image(file, channels=3)
    image.set_shape([None, None, None])
    image = tf.image.resize_images(image, (160, 160), method=1)
    img_pre = tf.image.per_image_standardization(image)
    return img_pre

def find_threshold(sess1, ckpt, embeddings, images_placeholder, phase_train_placeholder, images_emb, test_label):
    """ Find threshold that makes the accuracy higher"""
    img_emb = sess1.run(embeddings, feed_dict={images_placeholder: images_emb, phase_train_placeholder: False})
    dist_list = []
    num = 0
    accu = 0
    threshold = 0

    while num < 2 * 561:
        img1 = img_emb[num]
        img2 = img_emb[num + 1]
        dist = np.sum(np.square(np.subtract(img1, img2)))
        dist_list.append(dist)
        num += 2

    for i in np.arange(0, 3, 0.01):
        temp = calcu_accuracy(dist_list, test_label, i)
        if accu < temp:
            accu = temp
            threshold = i
    return accu, threshold, ckpt


def main():
    g = tf.Graph()
    with g.as_default():
        ckpt = tf.train.latest_checkpoint('/Users/lees/Desktop/model')
        saver = tf.train.import_meta_graph(os.path.join('/Users/lees/Desktop/model', '{}.meta'.format(ckpt)))

        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")

        with tf.Session() as sess1:
            saver.restore(sess1, tf.train.latest_checkpoint('/Users/lees/Desktop/model'))
            sampled_test_data, test_label = sample_test_data('~/Desktop/siamese_network/img2')

            images = tf.map_fn(lambda i: preprocessing_image_data(i), np.array(sampled_test_data),dtype=tf.float32)
            images_emb = sess1.run(images)

            accu, threshold, ckpt = find_threshold(sess1, ckpt, embeddings, images_placeholder, phase_train_placeholder, images_emb,
                                         test_label)
            print('-' * 10)
            print('Accuracy: %.3f' % accu)
            print('Threshold: %.2f ' % threshold)
            print('Model saved in path: %s' % ckpt)
            print('-' * 10)


if __name__ == '__main__':
    main()