# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 23:06:31 2017

@author: Kel
"""

import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import graph
import params
import util
import matplotlib.pyplot as plt

train_dir = 'saved_model/'

data_record = ["dataset/fly_train.tfrecords", "dataset/fly_test.tfrecords"]

p = params.Params()

batch_train = util.read_and_decode(p, data_record[0], my_data=False)
batch_test = util.read_and_decode(p, data_record[1], my_data=False)

img_L = tf.placeholder(tf.float32, [p.batch_size, p.target_h, p.target_w, 3])
img_R = tf.placeholder(tf.float32, [p.batch_size, p.target_h, p.target_w, 3])
disp = tf.placeholder(tf.float32, [p.batch_size, p.target_h, p.target_w, 1])
phase = tf.placeholder(tf.bool)

pred = graph.GCNet(img_L, img_R, phase, p.max_disparity)

# loss = tf.reduce_mean(tf.losses.mean_squared_error(pred, gt))
loss = tf.losses.absolute_difference(pred, disp)

learning_rate = 0.001
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)

global_step = tf.Variable(0, name='global_step', trainable=False)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_op = optimizer.minimize(loss, global_step=global_step)

init = tf.group(tf.global_variables_initializer(),
                tf.local_variables_initializer())

loss_summary = tf.summary.scalar("loss", loss)
train_loss_ = []
test_loss_ = []
saver = tf.train.Saver()
with tf.Session() as sess:
    summary_writer_train = tf.summary.FileWriter("./log/fly/train", graph=tf.get_default_graph())
    summary_writer_test = tf.summary.FileWriter("./log/fly/test", graph=tf.get_default_graph())
    restore_dir = tf.train.latest_checkpoint(train_dir)
    if restore_dir:
        saver.restore(sess, restore_dir)
        print('restore succeed')
    else:
        sess.run(init)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for step in range(150001):
        batch = sess.run(batch_train)
        feed_dict = {img_L: batch[0], img_R: batch[1], disp: batch[2], phase: True}

        #        _, loss_value, sample_dis, sample_gt = sess.run([train_op, loss,
        #        pred[0, 100, 100, :], disp[0, 100, 100,:]], feed_dict=feed_dict)
        _, loss_value, glb_step, loss_s = sess.run([train_op, loss, global_step, loss_summary], feed_dict=feed_dict)
        summary_writer_train.add_summary(loss_s, glb_step)
        if glb_step % 2 == 0 and step > 0:
            #            print('Step %d: training loss = %.2f | sample disparity: %.2f |
            #            ground truth: %.2f' % (step, loss_value, sample_dis, sample_gt))
            print('Step %d: training loss = %.2f' % (glb_step, loss_value))
            # train_loss_.append([loss_value, glb_step])

        if glb_step % 1000 == 0 and step > 0:
            test_total_loss = 0
            for j in range(10):
                batch = sess.run(batch_test)
                feed_dict = {img_L: batch[0], img_R: batch[1], disp: batch[2], phase: False}
                test_loss, loss_s = sess.run([loss, loss_summary], feed_dict=feed_dict)
                test_total_loss += test_loss
                summary_writer_test.add_summary(loss_s, glb_step + j)
            test_total_loss = test_total_loss / 10
            print('------------------  Step %d: test loss = %.2f ------------------' % (glb_step, test_total_loss))
            # test_loss_.append([test_total_loss, glb_step])

            if glb_step % (1000 * 5) == 0 and step > 0:
                # fig, axes = plt.subplots(nrows=2, ncols=2)
                # pd.DataFrame(train_loss_, columns=["loss", "step"]).plot(ax=axes[0, 0], x="step", y="loss",
                #                                                          title="Training loss")
                # pd.DataFrame(test_loss_, columns=["loss", "step"]).plot(ax=axes[0, 1], x="step", y="loss",
                #                                                         title="Testing loss")
                # # print(pd.DataFrame(train_loss_, columns=["loss", "step"]))
                # plt.show()
                key = input("Do you want to save model?[y/n] ").capitalize()
                if key == "Y":
                    saver.save(sess, train_dir, global_step=global_step)
                key = input("Do you want to continue training?[y/n] ").capitalize()
                if key == "N":
                    break
            merged_summary_op = tf.summary.merge_all()

    coord.request_stop()
    coord.join(threads)
