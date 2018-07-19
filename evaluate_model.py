#!/usr/bin/env python
# -*- coding_utf-8 -*-
"""
    Author: Zhenghan Lee
    Date: 2018.7.17
"""
import os
import numpy as np
import random
from six.moves import xrange
import tensorflow as tf
import argparse
import sys
import itertools


def sample_val_data(val_data_path):
    """ sample data from val_dataset, which including 50 pairs of same person and 50 pairs of different person"""
    temp_container = []
    classes_num = 0
    pick_num = 0
    sampled_val_data = []
    n_classes = len(os.listdir(val_data_path))-1
    img_num_same = 50
    img_num_diff = 50
    for i in os.listdir(val_data_path): # walk through all the dir except for .DS_Store(Mac Os), Win doesn't inclue this file
        if i == '.DS_Store':
            pass
        else:
            img_num_per_classes = 0
            for j in os.listdir(os.path.join(val_data_path, i)):
                if j == '.DS_Store':
                    pass
                else:
                    data = (os.path.join(val_data_path,i,j), classes_num)
                    temp_container.append(data)
                    img_num_per_classes += 1
                    if img_num_per_classes >= 40:
                        break
        classes_num += 1
    # pick same classes
    while pick_num < img_num_same:
        pick_class = np.ceil(np.random.uniform(0, n_classes-1))
        start_idx = pick_class * 40
        end_idx = start_idx + 39
        img = random.sample(list(np.arange(start_idx, end_idx)),2)
        sampled_val_data.append((temp_container[int(img[0])][0],temp_container[int(img[1])][0],1))
        pick_num += 1
    pick_num = 0
    np.random.shuffle(temp_container)
    # pick diff classes
    while pick_num < img_num_diff:
        img1 = temp_container.pop(0)
        img2 = temp_container.pop(0)
        if img1[1] == img2[1]:
            sampled_val_data.append((img1[0], img2[0], 1))
        else:
            sampled_val_data.append((img1[0], img2[0], 0))
        pick_num += 1
    np.random.shuffle(sampled_val_data)
    val_data_array = []
    val_label_array = []
    for i in sampled_val_data:
        val_data_array.append(i[0])
        val_data_array.append(i[1])
        val_label_array.append(i[2])

    return val_data_array, val_label_array, img_num_same, img_num_diff

def calcu_accu(dist, label, threshold):
    correct = 0
    for idx, i in enumerate(dist):
        if i > threshold:
            label_ = 0
        else:
            label_ = 1
        if label[idx] == label_:
            correct += 1
    return correct / len(dist)

def accu(model_version):
    val_data, val_label ,img_num_same, img_num_diff= sample_val_data('/Users/lees/Desktop/siamese_network/img1')
    print('sample val_data done')
    with tf.Session() as sess:
        print('a')
        saver = tf.train.import_meta_graph(os.path.join('/Users/lees/Desktop/model', 'model-{}.meta'.format(model_version)))
        print('b')
        saver.restore(sess, tf.train.latest_checkpoint('/Users/lees/Desktop/model'))
        print('Model restored')
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")

        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")

        image_array = []
        for i in val_data:
            file = tf.read_file(i)
            image = tf.image.decode_image(file, channels=3)
            image.set_shape([None, None, None])
            image = tf.image.resize_images(image, (160, 160), method=1)
            img_pre = tf.image.per_image_standardization(image)
            img1 = np.array([sess.run(img_pre)])
            img_emb = sess.run(embeddings, feed_dict={images_placeholder: img1, phase_train_placeholder: False})
            image_array.append(img_emb)
        dist_list = []
        num = 0
        print('1')
        while num < 2*(img_num_same + img_num_diff):
            img1 = image_array[num]
            img2 = image_array[num+1]
            dist = np.sum(np.square(np.subtract(img1, img2)))
            dist_list.append(dist)
            num += 2
        accu = 0
        threshold = 0
        for i in np.arange(0,3,0.01):
            temp = calcu_accu(dist_list, val_label, i)
            if accu < temp:
                accu = temp
                threshold = i
            else:
                pass
        print(accu, threshold,model_version)

if __name__ == '__main__':
    accu('20180719-181429')

