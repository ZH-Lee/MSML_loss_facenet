#!/usr/bin/env python
# -*- coding_utf-8 -*-
"""
    Author: Zhenghan Lee
    Date: 2018.7.17
"""
import os
import numpy as np
import random
import tensorflow as tf
from datetime import datetime
from itertools import combinations, chain
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def sample_val_data(val_data_path):
    """ sample data from val_dataset, which including 50 pairs of same person and 50 pairs of different person"""
    temp_container = []
    classes_num = 0
    pick_num = 0
    sampled_val_data = []
    n_classes = len(os.listdir(val_data_path))-1
    img_num_same =50
    img_num_diff = 110
    for i in os.listdir(val_data_path): # walk through all the dir except for .DS_Store(Mac Os), Win doesn't inclue this file
        if i != '.DS_Store':
            img_num_per_classes = 0
            for j in os.listdir(os.path.join(val_data_path, i)):
                if j != '.DS_Store':
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

def calcu_accuracy(dist, label, threshold):
    """Calculate accuracy based on diffrenet threshold"""
    correct = 0 # currect img num
    for idx, i in enumerate(dist):
        if i > threshold:
            label_ = 0
        else:
            label_ = 1
        if label[idx] == label_:
            correct += 1
    return correct / len(dist)

def preprocessing_image_data(img_path):
    file = tf.read_file(img_path)
    image = tf.image.decode_image(file, channels=3)
    image.set_shape([None, None, None])
    image = tf.image.resize_images(image, (160, 160), method=1)
    img_pre = tf.image.per_image_standardization(image)
    return img_pre

def find_threshold(sess1, ckpt, embeddings, images_placeholder, phase_train_placeholder, images_emb,val_label,img_num_same,img_num_diff):
    img_emb = sess1.run(embeddings, feed_dict={images_placeholder: images_emb, phase_train_placeholder: False})
    dist_list = []
    num = 0

    while num < 2 * (img_num_same + img_num_diff):
        img1 = img_emb[num]
        img2 = img_emb[num + 1]
        dist = np.sum(np.square(np.subtract(img1, img2)))
        dist_list.append(dist)
        num += 2
    accu = 0
    threshold = 0
    for i in np.arange(0, 3, 0.01):
        temp = calcu_accuracy(dist_list, val_label, i)
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
            val_data, val_label, img_num_same, img_num_diff = sample_val_data('~/Desktop/siamese_network/img1')

            images = tf.map_fn(lambda i: preprocessing_image_data(i), np.array(val_data),dtype=tf.float32)
            images_emb = sess1.run(images)
            accu, threshold, ckpt = find_threshold(sess1, ckpt, embeddings, images_placeholder, phase_train_placeholder, images_emb,
                                          val_label, img_num_same,img_num_diff)
            print('*' * 10)
            print('Accuracy: %.3f' % accu )
            print('Threshold: %.2f ' % threshold)
            print('Model saved in path: %s' % ckpt)
            print('*' * 10)

if __name__ == '__main__':
    new_accuracy()
    start = datetime.now()
    sample_test_data('~/Desktop/siamese_network/img2')
    print(datetime.now() - start)