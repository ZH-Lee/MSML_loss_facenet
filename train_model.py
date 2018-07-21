#!/usr/bin/env python
# -*- coding_utf-8 -*-
"""
    Author: Zhenghan Lee
    Date:
"""
import os
import numpy as np
import tensorflow as tf
import importlib
from six.moves import xrange
import itertools
import time
import argparse
import sys
import random
from evaluate_model import new_accuracy,sample_val_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def get_image_path(facedir):
    image_paths = []
    if os.path.isdir(facedir):
        images = os.listdir(facedir)
        if '.DS_Store' in images:
            images.remove('.DS_Store')
        image_paths = [os.path.join(facedir, img) for img in images]
    return image_paths

def get_dataset(path):
    dataset = []
    path_exp = os.path.expanduser(path)
    classes = [path for idx,path in enumerate(os.listdir(path_exp)) if os .path.isdir(os.path.join(path_exp, path))]
    classes.sort()
    n_classes = len(classes)
    for i in range(n_classes):
        class_name = classes[i]
        face_dir = os.path.join(path_exp, str(class_name))
        image_paths = get_image_path(face_dir)
        dataset.append(image_paths)
    return dataset

def sample_person(dataset, person_per_batch, images_per_person):
    n_images = person_per_batch * images_per_person
    n_classes = len(dataset)
    class_indices = np.arange(n_classes)
    np.random.shuffle(class_indices)
    i = 0
    image_paths = []
    num_per_class = []

    # Sample images from these classes until we have enough
    while len(image_paths) < n_images:
        class_index = class_indices[i]
        n_images_in_class = len(dataset[class_index]) # 每个类的图像数
        image_indices = np.arange(n_images_in_class) #
        np.random.shuffle(image_indices)
        n_images_from_class = min(n_images_in_class, images_per_person, n_images-len(image_paths))
        # n_images_in_class : 每个人在数据库里面有多少张，images_oer_person: 每个人需要抽取的张数
        # 如果每个人数据库里面的图像张数小于要抽取的张数，就将数据库的数据全部抽取。n_images-len(image_paths)：总共还需要抽取多少张
        idx = image_indices[0:n_images_from_class]
        image_path_for_class = [dataset[class_index][j] for j in idx]
        image_paths += image_path_for_class # 把采样好的图像的路径加入list
        i+=1
        num_per_class.append(n_images_from_class)

    return image_paths, num_per_class

def triplet_loss(anchor, positive, negative, alpha):
    """Calculate the triplet loss according to the FaceNet paper"""

    with tf.variable_scope('triplet_loss'):
        pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
        neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)

        basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
        loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)

    return loss

def save_variables_and_metagraph(sess, saver, model_dir, model_name, step):
    # Save the model checkpoint
    checkpoint_path = os.path.join(model_dir, 'model-%s.ckpt' % model_name)
    saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)
    metagraph_filename = os.path.join(model_dir, 'model-%s.meta' % model_name)

def msml_loss(anchor ,pos, neg1 , neg2, alpha):

    with tf.variable_scope('msml_loss'):
        pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, pos)), 1)
        neg_dist = tf.reduce_sum(tf.square(tf.subtract(neg1, neg2)), 1)
        basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
        loss = tf.reduce_mean(tf.maximum(basic_loss,0.0),0)

    return loss

        
def main(args):
    network = importlib.import_module(args.model_dir)
    model_save_dir = os.path.join(os.path.expanduser('/Users/lees/Desktop/model'))
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False,name='global_step')
        batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
        image_paths_placeholder = tf.placeholder(tf.string, shape=(None,4), name='image_paths')
        phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
        learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')

        input_queue = tf.FIFOQueue(capacity=100000,
                                  dtypes=[tf.string],
                                  shapes=[(4,)])

        enqueue_op = input_queue.enqueue_many([image_paths_placeholder])

        n_preprocess_threads = 6
        image_and_labels = []

        for _ in range(n_preprocess_threads):
            filenames = input_queue.dequeue()
            images = []
            for filename in tf.unstack(filenames):
                file_contents = tf.read_file(filename)
                image = tf.image.decode_image(file_contents, channels=3)
                image.set_shape([None, None, None])
                image = tf.image.resize_images(image, (args.image_size,args.image_size),method=1)

                images.append(tf.image.per_image_standardization(image))

            image_and_labels.append([images])

        image_batch = tf.train.batch_join(image_and_labels, batch_size=batch_size_placeholder,
                                                        shapes=[(args.image_size, args.image_size, args.channel)],
                                                        enqueue_many=True,
                                                        capacity=10000,
                                                        allow_smaller_final_batch=True)  # 获取batch数据

        image_batch = tf.identity(image_batch, 'image_batch')
        image_batch = tf.identity(image_batch, 'input')

        prelogits= network.inference(image_batch, args.keep_probability,
                                         phase_train=phase_train_placeholder,
                                         bottleneck_layer_size=args.embedding_size,
                                         weight_decay=args.weight_decay)

        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings') #l2归一化
        anchor, pos, neg1, neg2 = tf.unstack(tf.reshape(embeddings, [-1,4,args.embedding_size]),4,1) # 获取apn

        msml_loss_ = msml_loss(anchor, pos, neg1, neg2, args.alpha)

        learning_rate = tf.train.exponential_decay(learning_rate_placeholder, global_step,
                                                   args.decay_steps, args.decay_rate, staircase=True)

        regularization_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n([msml_loss_] + regularization_loss, name='total_loss') # 加上正则项的loss

        train_op = train_(total_loss, global_step, learning_rate,
                          args.moving_average_decay, tf.global_variables())

        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=5)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        with sess.as_default():
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord, sess=sess)
            epoch = 0
            val_data, val_label, img_num_same, img_num_diff = sample_val_data('/Users/lees/Desktop/siamese_network/img1')
            ckpt = tf.train.get_checkpoint_state('/Users/lees/Desktop/model')
            if ckpt and ckpt.model_checkpoint_path:  # 判断ckpt是否为空，若不为空，才进行模型的加载，否则从头开始训练
                saver.restore(sess, tf.train.latest_checkpoint('/Users/lees/Desktop/model'))
                print('exits, now reload model: %s' % ckpt)
            while epoch < args.max_epoch:
                print('epoch: %d' %epoch)
                train2(args,sess, args.batch_size,args.person_per_batch, args.images_per_person,enqueue_op,
                        image_paths_placeholder,args.embedding_size,embeddings,
                        batch_size_placeholder, phase_train_placeholder,train_op,args.epoch_size,args.alpha,total_loss,global_step,
                          learning_rate_placeholder,learning_rate)
                epoch += 1
                subdir = time.strftime('%Y%m%d-%H%M%S', time.localtime())
                saver.save(sess, '/Users/lees/Desktop/model/{}'.format(subdir))
                new_accuracy(val_data, val_label,img_num_same,img_num_diff)
                print('Model saved at %s' % model_save_dir)
            coord.request_stop()
            coord.join(threads)


def _add_loss_summaries(total_loss):
    """Add summaries for losses.

    Generates moving average for all losses

    Args:
      total_loss: Total loss from loss().
    Returns:
      loss_averages_op: op for generating moving averages of losses.
    """
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    return loss_averages_op

def train_(total_loss, global_step, learning_rate,moving_average_decay, update_gradient_vars):

    loss_average_op = _add_loss_summaries(total_loss)
    # 计算梯度
    with tf.control_dependencies([loss_average_op]):
        opt = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=0.1)
        grads = opt.compute_gradients(total_loss, update_gradient_vars)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    variable_averages = tf.train.ExponentialMovingAverage(
        moving_average_decay, global_step) #计算参数滑动平均
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op


def train1(args,sess, batch_size, person_per_batch,images_per_person,enqueue_op,
          image_paths_placeholder,embedding_size,embeddings,
            batch_size_placeholder, phase_train_placeholder,train_op,epoch_size,alpha,
          loss,global_step,learning_rate_placeholder,learning_rate):

    batch_number = 0
    lr = args.learning_rate
    while batch_number < epoch_size:  #一个epoch有多少组subepoch
        print('batch_number:%d' %batch_number)
        train_set = get_dataset('~/Desktop/siamese_network/img/lfw')

        image_paths, num_per_class= sample_person(train_set, person_per_batch, images_per_person) #随机选取一些图片用以下次训练 90个
        print('sample dataset done')
        class_per_img = [i.split('/')[-2] for i in image_paths]
        n_example = person_per_batch * images_per_person # 40*39

        image_paths_array = np.reshape(np.expand_dims(np.array(image_paths), 1), (-1, 3))
        class_per_img_array = np.reshape(np.expand_dims(np.array(class_per_img), 1), (-1, 3))

        sess.run(enqueue_op, {image_paths_placeholder: image_paths_array})
        emb_array_for_select = np.zeros((n_example, embedding_size)) # 40 x 39 x 128 embedding_size是一幅图像映射出来的特征数，用来放一个n_example的所有图片经过network后的feature map
        nrof_batches = int(np.ceil(n_example / batch_size)) #1560张图片，batch大小为15，一共有 40 x 39 / 15 个batch

        for i in xrange(nrof_batches):
            n_batch_size = min(n_example-i*batch_size, batch_size)
            # emb是一个triplets组经过神经网络，此时的神经网络是上一次训练的结果，得到的128维feature map，维数是(3,160,160,3),分别是3个图像，160x160的图像大小，3通道（RGB
            emb= sess.run([embeddings], feed_dict={phase_train_placeholder:False, batch_size_placeholder:n_batch_size})
            emb_array_for_select[[j+15*i for j in range(15)],:] = emb # 这个feature map矩阵将用来选择hard-mining 的triplets tuple images

        # 通过文中介绍的on-line选择一些图像用来训练这一次的network
        triplets, nrof_random_negs, nrof_triplets = select_triplets(emb_array_for_select, num_per_class,
                                                                    image_paths, person_per_batch,alpha)
        print('select triplets done')
        nrof_batches1 = int(np.round(nrof_triplets/ batch_size)) # batch_size = 9
        triplet_paths = list(itertools.chain(*triplets)) #将每个triplets的图片路径链接起来

        triplet_paths_array = np.reshape(np.expand_dims(np.array(triplet_paths), 1), (-1, 3)) #组成一个个triplet，再构成一个整个列表
        sess.run(enqueue_op, {image_paths_placeholder: triplet_paths_array}) #triplet入队
        i=0
        print('start train...')
        while i < nrof_batches1:

            feed_dict = {batch_size_placeholder: 9,  phase_train_placeholder: True, learning_rate_placeholder:lr}
            err, _, step, emb,l = sess.run([loss, train_op, global_step, embeddings,learning_rate], feed_dict=feed_dict)
            i+=1
            # if err < 2.0:
            #     return
            print(err,step,l)
        batch_number += 1
        print('done')
    return

def train2(args,sess, batch_size, person_per_batch,images_per_person,enqueue_op,
          image_paths_placeholder,embedding_size,embeddings,
            batch_size_placeholder, phase_train_placeholder,train_op,epoch_size,alpha,
          loss,global_step,learning_rate_placeholder,learning_rate):

    batch_number = 0
    err = 0.0
    step = 0
    l = 0.0
    lr = args.learning_rate
    while batch_number < epoch_size:  #一个epoch有多少组subepoch
        print('batch_number:%d' %batch_number)
        train_set = get_dataset('~/Desktop/siamese_network/img/lfw')
        image_paths, num_per_class= sample_person(train_set, person_per_batch, images_per_person) #随机选取一些图片用以下次训练 3

        #evaluate_model(sess, enqueue_op, image_paths_placeholder, embeddings,phase_train_placeholder,batch_size_placeholder)
        n_example = person_per_batch * images_per_person # 50 x 40

        image_paths_array = np.reshape(np.expand_dims(np.array(image_paths), 1), (-1, 4))
        sess.run(enqueue_op, {image_paths_placeholder: image_paths_array})
        emb_array_for_select = np.zeros((n_example, embedding_size)) # 146 x 8 x 128 embedding_size是一幅图像映射出来的特征数，用来放一个n_example的所有图片经过network后的feature map
        nrof_batches = int(np.round(n_example / batch_size))-1 # 1152张图片，8，一共有 144 x 8 / 8 个batch 144

        for i in xrange(nrof_batches):
            n_batch_size = min(n_example-i*batch_size, batch_size)
            # emb是一个triplets组经过神经网络，此时的神经网络是上一次训练的结果，得到的128维feature map，维数是(3,160,160,3),分别是3个图像，160x160的图像大小，3通道（RGB
            emb= sess.run([embeddings], feed_dict={phase_train_placeholder:False, batch_size_placeholder:n_batch_size})
            emb_array_for_select[[j+15*i for j in range(15)], :] = emb # 这个feature map矩阵将用来选择hard-mining

        # 通过文中介绍的on-line选择一些图像用来训练这一次的network
        # triplets, nrof_random_negs, nrof_triplets = select_triplets(emb_array_for_select, num_per_class,class_per_img_array,
        #                                                             image_paths, person_per_batch,alpha)
        msml, n_msml = select_msml(emb_array_for_select, num_per_class,image_paths, person_per_batch,alpha)
        nrof_batches1 = int(np.round(n_msml/ batch_size)) # batch_size = 15
        msml_paths = list(itertools.chain(*msml)) #将每个triplets的图片路径链接起来

        msml_paths_array = np.reshape(np.expand_dims(np.array(msml_paths), 1), (-1, 4)) #组成一个个triplet，再构成一个整个列表
        sess.run(enqueue_op, {image_paths_placeholder: msml_paths_array}) #triplet入队
        i=0

        while i < nrof_batches1:

            feed_dict = {batch_size_placeholder: 16,  phase_train_placeholder: True, learning_rate_placeholder:lr}
            err, _, step, emb,l = sess.run([loss, train_op, global_step, embeddings,learning_rate], feed_dict=feed_dict)
            i+=1

        print('loss: %.9f, step: %d, learning_rate: %.9f' % (err,step,l))

        batch_number += 1
    return
def conti_train():
    pass
def select_msml(embeddings, nrof_images_per_class,image_paths, person_per_batch, alpha):
    emb_start_idx = 0
    num_ = 0
    num_1 = 0
    msml = []
    neg_tuple_idx = []
    ap_tuple_idx = []
    for i in xrange(person_per_batch):  # i代表第几个人
        n_img_from_this_person = int(nrof_images_per_class[i])
        num_ +=n_img_from_this_person #neg类的开始坐标
        for m in xrange(40, sum(nrof_images_per_class)):
            neg1_idx = m
            new_embeddings = embeddings[m:]
            temp_neg_dist = np.sum(np.square(embeddings[m] - new_embeddings),1)[1:]
            if temp_neg_dist.shape[0]==0:
                pass
            else:
                temp_neg_dist = temp_neg_dist.tolist()
                neg2_idx = temp_neg_dist.index(min(temp_neg_dist))
                if neg2_idx+m - neg1_idx < 40:
                    pass
                else:
                    neg_tuple_idx.append((neg1_idx,neg2_idx+m))
        for j in xrange(1,n_img_from_this_person): #从某个类中开始选择A，P
            a_idx = emb_start_idx +j -1
            new_embedding = embeddings[a_idx: num_]

            temp_dists_sqr = np.sum(np.square(embeddings[a_idx] - new_embedding),1)
            temp = temp_dists_sqr.tolist()# dist(anchor - embeddings)
            pos_idx = temp.index(max(temp))
            ap_tuple_idx.append((a_idx, pos_idx+num_1))

        sample_ap_list = random.sample(ap_tuple_idx,7)
        sample_neg_list = random.sample(neg_tuple_idx, 7)

        for k in sample_neg_list:
            for q in sample_ap_list:
                d1 = np.sum(np.square(embeddings[q[0]]- embeddings[q[1]]))
                d2 = np.sum(np.square(embeddings[k[0]]- embeddings[k[1]]))
                if d2-d1<alpha:
                    msml.append((image_paths[q[0]], image_paths[q[1]], image_paths[k[0]],image_paths[k[1]]))

        neg_tuple_idx = []
        num_1 += n_img_from_this_person
        emb_start_idx += n_img_from_this_person

    np.random.shuffle(msml)
    print(len(msml))
    return msml, len(msml)


def select_triplets(embeddings, nrof_images_per_class,image_paths, people_per_batch, alpha):
    """ Select the triplets for training
    """
    emb_start_idx = 0
    num_trips = 0
    triplets = []
    for i in xrange(people_per_batch):
        nrof_images = int(nrof_images_per_class[i])
        for j in xrange(1, nrof_images):
            a_idx = emb_start_idx + j - 1
            neg_dists_sqr = np.sum(np.square(embeddings[a_idx] - embeddings), 1)
            print(neg_dists_sqr)
            for pair in xrange(j, nrof_images):  # For every possible positive pair.
                p_idx = emb_start_idx + pair
                pos_dist_sqr = np.sum(np.square(embeddings[a_idx] - embeddings[p_idx]))
                print(pos_dist_sqr)
                neg_dists_sqr[emb_start_idx:emb_start_idx + nrof_images] = np.NaN
                all_neg = np.where(neg_dists_sqr - pos_dist_sqr < alpha)[0]
                nrof_random_negs = all_neg.shape[0]

                if nrof_random_negs > 0:
                    rnd_idx = np.random.randint(nrof_random_negs)
                    n_idx = all_neg[rnd_idx]
                    triplets.append((image_paths[a_idx], image_paths[p_idx], image_paths[n_idx]))
                num_trips += 1

        emb_start_idx += nrof_images

    np.random.shuffle(triplets)
    return triplets, num_trips, len(triplets)



def parse(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str,
                        help='Directory where model in.',default='face_model')
    parser.add_argument('--max_epoch',type=int,
                        help='max_epoch',default=50)
    parser.add_argument('--epoch_size', type=int,
                        help='Number of batches per epoch', default=10)
    parser.add_argument('--keep_probability', type=int,
                        help='params keep_probability', default=0.5)
    parser.add_argument('--batch_size', type=int,
                        help='Number of images to process in a batch', default=15)
    parser.add_argument('--embedding_size', type=int,
                        help='Number of feature map', default=128)
    parser.add_argument('--weight_decay', type=float,
                        help='For L2 weight regularization', default=0.05)
    parser.add_argument('--alpha', type=float,
                        help='Margin', default=0.3)
    parser.add_argument('--moving_average_decay', type=float,
                        help='Exponential decay for tracking of training parameters.', default=0.88)
    parser.add_argument('--person_per_batch', type=int,
                        help='Number of people per batch.', default=50)
    parser.add_argument('--images_per_person', type=int,
                        help='Number of images per person.', default=40)
    parser.add_argument('--image_size', type=int,
                        help='Input images size.', default=160)
    parser.add_argument('--channel', type=int,
                        help='Input images channel.', default=3)
    parser.add_argument('--learning_rate',type=float,
                         help='learning_rate', default=9e-5)
    parser.add_argument('--decay_rate', type=float,
                        help='decay_rate', default=0.8)
    parser.add_argument('--decay_steps', type=int,
                        help='decay_steps', default=6000)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse(sys.argv[1:]))
    #sample_val_data('/Users/lees/Desktop/siamese_network/img1')
    #evaluate_model()