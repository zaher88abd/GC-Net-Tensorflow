# -*- coding: utf-8 -*-
"""
Created on Tue May 15 22:24:05 2018

@author: Kel
"""
from params import Params
import tensorflow as tf


def _parse_function(proto):
    params = Params()
    width, height = params.original_w, params.original_h
    batch_size = params.batch_size
    target_w, target_h = params.target_w, params.target_h

    # define your tfrecord again. Remember that you saved your image as a string.
    features = {
        'img_left': tf.FixedLenFeature([], tf.string),
        'img_right': tf.FixedLenFeature([], tf.string),
        'disparity': tf.FixedLenFeature([], tf.string)
    }

    # Load one example
    parsed_features = tf.parse_single_example(proto, features)

    # Turn your saved image string into an array
    # parsed_features['image'] = tf.decode_raw(
    #     parsed_features['image'], tf.uint8)

    image_left = tf.decode_raw(parsed_features['img_left'], tf.uint8)
    image_right = tf.decode_raw(parsed_features['img_right'], tf.uint8)
    disparity = tf.decode_raw(parsed_features['disparity'], tf.float32)

    return image_left, image_right, disparity


def read_and_decode(params, filename, my_data=False):
    if my_data:
        width, height = params.my_original_w, params.my_original_h
    else:
        width, height = params.original_w, params.original_h
    batch_size = params.batch_size
    target_w, target_h = params.target_w, params.target_h

    # filename_queue = tf.train.string_input_producer([filename])
    files = tf.data.Dataset.list_files(filename)

    dataset = files.interleave(tf.data.TFRecordDataset, cycle_length=6,
                               num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    # Maps the parser on every filepath in the array. You can set the number of parallel loaders here
    dataset = dataset.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.repeat()

    dataset = dataset.shuffle(1)

    # Set the batchsize
    dataset = dataset.batch(batch_size)

    # Create an iterator
    iterator = dataset.make_one_shot_iterator()

    # Create your tf representation of the iterator
    image_left_batch, image_right_batch, disparity_batch = iterator.get_next()

    image_left = tf.reshape(image_left_batch, [batch_size, height, width, 3])

    image_right = tf.reshape(image_right_batch, [batch_size, height, width, 3])

    disparity = tf.reshape(disparity_batch, [batch_size, height, width, 1])

    # Convert from [0, 255] -> [-0.5, 0.5] floats.
    image_left = tf.cast(image_left, tf.float32) * (1. / 255) - 0.5
    image_right = tf.cast(image_right, tf.float32) * (1. / 255) - 0.5
    concat = tf.concat([image_left, image_right, disparity], 3)
    img_crop = tf.random_crop(concat, [batch_size, target_h, target_w, 7])

    return img_crop[:, :, :, 0:3], img_crop[:, :, :, 3:6], img_crop[:, :, :, 6:]
