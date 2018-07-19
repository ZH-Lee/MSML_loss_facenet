#!/usr/bin/env python
# -*- coding_utf-8 -*-
"""
    Author: Zhenghan Lee
    Date: 2018/07/10
"""
import tensorflow as tf
import detect_face
import numpy as np
import cv2
import os
import shutil


def gene_face(img,save_path):
    with tf.Graph().as_default():
        sess1 = tf.Session()
        with sess1.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess1, None)
    minsize = 20
    threshold = [0.6, 0.7, 0.7]
    factor = 0.709
    bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    if len(bounding_boxes) < 1:
        return False
    else:
        for face_position in bounding_boxes:
            face_position = face_position.astype(int)

            face = img[face_position[1]:face_position[3],
                   face_position[0]:face_position[2]]
            cv2.imwrite(save_path,face)
# if __name__ == '__main__':
#     root_dir = '/Users/lees/Desktop/siamese_network/img'
#     num = 0
#     for i in os.listdir(root_dir):
#         if i == '.DS_Store':
#             pass
#         else:
#             if len([img for img in os.listdir(os.path.join(root_dir,i))]) >= :
#                 #shutil.rmtree(os.path.join(root_dir, i))
#                 num +=1
#     print(num)

