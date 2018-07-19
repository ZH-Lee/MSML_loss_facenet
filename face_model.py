#!/usr/bin/env python
# -*- coding_utf-8 -*-
"""
    Author: Zhenghan Lee
    Date:
"""

import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim

def inception_block(net, activation_fn=tf.nn.relu,scope=None, reuse=None):
    """Builds the inception with dimension reductions block"""
    with tf.variable_scope(scope, 'inception_block',[net], reuse=reuse):
        with tf.variable_scope('Branch_0'):
            conv1_0 = slim.conv2d(net,
                                  num_outputs=64,
                                  kernel_size=1,
                                  scope='conv1_1',
                                  padding='SAME',
                                  activation_fn=activation_fn,
                                  reuse=reuse)

        with tf.variable_scope('Branch_1'):
            conv2_0 = slim.conv2d(net,
                                  num_outputs=32,
                                  kernel_size=1,
                                  scope='conv_a_1x1',
                                  activation_fn=None,reuse=reuse)
            conv2_1 = slim.conv2d(conv2_0,
                                  num_outputs=32,
                                  kernel_size=3,
                                  scope='conv_b_3x3',
                                  padding='SAME',
                                  activation_fn=activation_fn,reuse=reuse)

        with tf.variable_scope('Branch_2'):
            conv3_0 = slim.conv2d(net,
                                  num_outputs=96,
                                  kernel_size=1,
                                  scope='conv_a_1x1',
                                  padding='SAME',
                                  activation_fn=None,reuse=reuse)
            conv3_1 = slim.conv2d(conv3_0,
                                  num_outputs=128,
                                  kernel_size=5,
                                  scope='conv_b_5x5',
                                  padding='SAME',
                                  activation_fn=activation_fn,reuse=reuse)
        with tf.variable_scope('Branch_3'):
            pool4_0 = slim.max_pool2d(net,
                                     kernel_size=3,
                                     padding='SAME')
            conv4_1 = slim.conv2d(pool4_0,
                                  num_outputs=32,
                                  kernel_size=1,
                                  scope='conv_a_1x1',
                                  padding='SAME',
                                  activation_fn=activation_fn,reuse=reuse)

        mixed = tf.concat([conv1_0, conv2_1, conv3_1, conv4_1],3)

    return mixed




def inference(images, keep_probability, phase_train=True,
              bottleneck_layer_size=128, weight_decay=0.0, reuse=None):
    batch_norm_params = {
        # Decay for the moving averages.
        'decay': 0.995,
        # epsilon to prevent 0s in variance.
        'epsilon': 0.001,
        # force in-place updates of mean and variance estimates
        'updates_collections': None,
        # Moving averages ends up in the trainable variables collection
        'variables_collections': [tf.GraphKeys.TRAINABLE_VARIABLES],
    }
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_initializer=slim.initializers.xavier_initializer(),
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
        # return inception_resnet_v1(images, is_training=phase_train,
        #                            dropout_keep_prob=keep_probability, bottleneck_layer_size=bottleneck_layer_size,
        #                            reuse=reuse)
        return inception_v1(images, is_training=phase_train,
                            bottleneck_layer_size=bottleneck_layer_size,
                            reuse=reuse,normalizer_fn=slim.batch_norm, normalizer_param=batch_norm_params,
                            dropout_keep_prob=keep_probability,)

def inception_v1(inputs,
                normalizer_fn,
                 normalizer_param,
                dropout_keep_prob,
                 is_training=True,
                 bottleneck_layer_size=128,
                 reuse=None,

                 scope='InceptionV1'
                 ):
    with tf.variable_scope(scope, 'InceptionV1', [inputs], reuse=reuse):
        with slim.arg_scope([slim.batch_norm, slim.dropout],is_training=is_training):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],stride=1, padding='SAME'):
                net = slim.conv2d(inputs, num_outputs=64, kernel_size=5,
                                  stride=2, padding='SAME',
                                  scope='conv_1a_7x7')

                net = slim.max_pool2d(net,
                                     kernel_size=3,
                                     stride=2,
                                     padding='SAME')
                net = slim.conv2d(net, num_outputs=192,kernel_size=3,
                                  stride=1,padding='SAME',scope='conv_2a_3x3')

                net = slim.max_pool2d(net,
                                      kernel_size=3,
                                      stride=2,
                                      padding='SAME')
                with tf.variable_scope('mixed_1a'):
                    net = inception_block(net)

                with tf.variable_scope('mixed_2a'):
                    net = inception_block(net)

                with tf.variable_scope('mixed_3a'):
                    net = inception_block(net)

                with tf.variable_scope('Logits'):
                    net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='SAME',
                                          scope='AvgPool_1a_8x8')
                    net = slim.flatten(net)
                net  = slim.dropout(net, keep_prob=dropout_keep_prob)
                net = slim.fully_connected(net, bottleneck_layer_size, activation_fn=None,
                                           scope='Bottleneck', reuse=False)
    return net

