#!/usr/bin/env python
# -*- coding_utf-8 -*-
"""
    Author: Zhenghan Lee
    Date:
"""
import cv2
import os
import numpy as np

def flip_img(img):
    return cv2.flip(img, 1) # 1 represent flip horizontal

def img_adjust_brightness(img,brightness):

    return np.uint8(np.clip((1.5 * img + brightness), 0, 255))

if __name__ == '__main__':
    root = '/Users/lees/Desktop/siamese_network/img/lfw'
    num = 1
    for i in os.listdir(root):
        if i == '.DS_Store':
            pass
        else:
            for j in os.listdir(os.path.join(root,i)):
                if j =='.DS_Store':
                    pass
                else:
                    img_path = os.path.join(root,i,j)
                    img = cv2.imread(img_path)
                    #for k in [0.2,-0.1]:
                        #new_img = img_adjust_brightness(img, k)
                    new_img = flip_img(img)
                    new_name = j.split('.')[0] + '_new'  +'.png'
                    cv2.imwrite(os.path.join(root,i,new_name),new_img)
