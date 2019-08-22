# -*- coding: utf-8 -*-
"""
Created on Tue May 15 22:24:05 2018

@author: Zaher
"""
import time
from tqdm import tqdm
import os
import tensorflow as tf
from PIL import Image
import re
import numpy as np
import cv2
import params
import pandas as pd

# tf.enable_eager_execution()


def readPFM(file):
    file = open(file, 'r', encoding='ISO-8859-1')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale


def rgba_to_rgb(img):
    '''
    change image from rgba to rgb
    [height, width, 4] -> [height, width, 3]
    '''
    img.load()
    img_temp = Image.new("RGB", img.size, (255, 255, 255))
    img_temp.paste(img, mask=img.split()[3])
    return img_temp


def preprocess_image(image, depth):
    if not depth:
        image = tf.image.decode_png(image, channels=3)
        image = tf.cast(image, tf.float32)
        image = image / 255.0 - 0.5  # normalize to [-0.5,0.5] range
    else:
        image = tf.image.decode_png(image, channels=1)
        image = tf.cast(image, tf.float32)
    return image


def load_and_preprocess_image(path, depth=False):
    image = tf.read_file(path)
    return preprocess_image(image, depth=depth)


def read_fly_db(monkaa_dataset=False):
    main_path = './dataset/'
    param = params.Params()

    l_img_path_train = []
    r_img_path_train = []
    d_img_path_train = []

    l_img_path_test = []
    r_img_path_test = []
    d_img_path_test = []

    count_train = 0
    count_test = 0
    if not os.path.exists(main_path + "train_dataset.csv") and not os.path.exists(main_path + "test_dataset.csv"):
        dirs_fly = ['./dataset/' + 'flyingthings3d_frames_cleanpass/',
                    './dataset/' + 'flyingthings3d__disparity/disparity/']
        for phase in tqdm(['TRAIN', 'TEST']):
            for group in tqdm(['A', 'B', 'C', 'D']):
                dir_group = dirs_fly[0] + phase + '/' + group
                dir_group2 = dirs_fly[1] + phase + '/' + group
                for img_group in tqdm(os.listdir(dir_group)):
                    dir_img_group = dir_group + '/' + img_group
                    dir_dis_group = dir_group2 + '/' + img_group
                    for img_name in tqdm(os.listdir(dir_img_group + '/left')):
                        img_path_l = dir_img_group + '/left/' + img_name
                        img_path_r = dir_img_group + '/right/' + img_name
                        if not os.path.exists(dir_dis_group + '/left/' + img_name):
                            disparity_path = dir_dis_group + '/left/' + img_name.split('.')[0] + '.pfm'
                            data, scale = readPFM(disparity_path)
                            cv2.imwrite(dir_dis_group + '/left/' + img_name, data)
                        disparity_path = dir_dis_group + '/left/' + img_name
                        if phase == 'TRAIN':
                            l_img_path_train.append(img_path_l)
                            r_img_path_train.append(img_path_r)
                            d_img_path_train.append(disparity_path)
                            count_train += 1
                        else:
                            l_img_path_test.append(img_path_l)
                            r_img_path_test.append(img_path_r)
                            d_img_path_test.append(disparity_path)
                            count_test += 1
        if monkaa_dataset == True:
            dirs_monkaa = [main_path + 'monkaa_frames_cleanpass',
                           main_path + 'monkaa_disparity']
            for folder in tqdm(os.listdir(dirs_monkaa[0])):
                for img in tqdm(os.listdir(os.path.join(dirs_monkaa[0], folder, 'left'))):
                    img_path_l = os.path.join(dirs_monkaa[0], folder, 'left', img)
                    img_path_r = os.path.join(dirs_monkaa[0], folder, 'right', img)
                    if not os.path.exists(os.path.join(dirs_monkaa[1], folder, 'left', img)):
                        disparity_path = os.path.join(dirs_monkaa[1], folder, 'left', img.split('.')[0] + '.pfm')
                        data, scale = readPFM(disparity_path)
                        cv2.imwrite(os.path.join(dirs_monkaa[1], folder, 'left', img), data)
                    disparity_path = os.path.join(dirs_monkaa[1], folder, 'left', img)

                    l_img_path_train.append(img_path_l)
                    r_img_path_train.append(img_path_r)
                    d_img_path_train.append(disparity_path)
                    count_train += 1

        dist = {'img_l': l_img_path_train, 'img_r': r_img_path_train, "img_d": d_img_path_train}
        pd.DataFrame(dist).to_csv(main_path + "train_dataset.csv")
        dist = {'img_l': l_img_path_test, 'img_r': r_img_path_test, "img_d": d_img_path_test}
        pd.DataFrame(dist).to_csv(main_path + "test_dataset.csv")
    else:
        train_dp = pd.read_csv(main_path + "train_dataset.csv")
        l_img_path_train = train_dp['img_l'].to_list()
        r_img_path_train = train_dp['img_r'].to_list()
        d_img_path_train = train_dp['img_d'].to_list()
        count_train = len(l_img_path_train)

        test_dp = pd.read_csv(main_path + "test_dataset.csv")
        l_img_path_test = test_dp['img_l'].to_list()
        r_img_path_test = test_dp['img_r'].to_list()
        d_img_path_test = test_dp['img_d'].to_list()
        count_test = len(l_img_path_test)

    train_ds = gen_dataset(d_img_path_train, l_img_path_train, r_img_path_train, param)
    test_ds = gen_dataset(d_img_path_test, l_img_path_test, r_img_path_test, param)
    return train_ds, test_ds, count_train, count_test


def random_crop(image_left, image_right, disparity):
    param = params.Params()
    batch_size = param.batch_size
    target_w, target_h = param.target_w, param.target_h
    concat = tf.concat([image_left, image_right, disparity], 3)
    img_crop = tf.random_crop(concat, [batch_size, target_h, target_w, 7])
    return img_crop[:, :, :, 0:3], img_crop[:, :, :, 3:6], img_crop[:, :, :, 6:]


def read_db(main_path, testing=10):
    param = params.Params()
    l_img_path = os.path.join(main_path, 'left')
    r_img_path = os.path.join(main_path, 'right')
    d_img_path = os.path.join(main_path, 'depth')
    images_ = os.listdir(l_img_path)
    all_l_image_paths = [os.path.join(l_img_path, x) for x in images_]
    all_r_image_paths = [os.path.join(r_img_path, x) for x in images_]
    all_d_image_paths = [os.path.join(d_img_path, x) for x in images_]

    l_img_path_train = all_l_image_paths[testing:]
    r_img_path_train = all_r_image_paths[testing:]
    d_img_path_train = all_d_image_paths[testing:]

    l_img_path_test = all_l_image_paths[:testing]
    r_img_path_test = all_r_image_paths[:testing]
    d_img_path_test = all_d_image_paths[:testing]
    count_train = len(l_img_path_train)
    count_test = len(l_img_path_test)
    train_ds = gen_dataset(d_img_path_train, l_img_path_train, r_img_path_train, param)
    test_ds = gen_dataset(d_img_path_test, l_img_path_test, r_img_path_test, param)
    return train_ds, test_ds, count_train, count_test


def gen_dataset(d_img_path, l_img_path, r_img_path, param):
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    path_l_ds = tf.data.Dataset.from_tensor_slices(l_img_path)
    path_r_ds = tf.data.Dataset.from_tensor_slices(r_img_path)
    path_d_ds = tf.data.Dataset.from_tensor_slices(d_img_path)
    image_l_ds = path_l_ds.map(lambda x: load_and_preprocess_image(x, depth=False), num_parallel_calls=AUTOTUNE)
    image_r_ds = path_r_ds.map(lambda x: load_and_preprocess_image(x, depth=False), num_parallel_calls=AUTOTUNE)
    image_d_ds = path_d_ds.map(lambda x: load_and_preprocess_image(x, depth=True), num_parallel_calls=AUTOTUNE)
    images_ds = tf.data.Dataset.zip((image_l_ds, image_r_ds, image_d_ds))
    ds = images_ds.apply(
        tf.data.experimental.shuffle_and_repeat(buffer_size=param.batch_size * 2))
    ds = ds.batch(param.batch_size).prefetch(buffer_size=AUTOTUNE).repeat()
    ds = ds.map(random_crop)
    return ds


def timeit(ds, batches):
    param = params.Params()

    overall_start = time.time()
    # Fetch a single batch to prime the pipeline (fill the shuffle buffer),
    # before starting the timer
    it = iter(ds.take(batches + 1))
    next(it)

    start = time.time()
    for i, (_, _, _) in enumerate(it):
        if i % 10 == 0:
            print('.', end='')

    print()
    end = time.time()

    duration = end - start
    print("{} batches: {} s".format(batches, duration))
    print("{:0.5f} Images/s".format(param.batch_size * batches / duration))
    print("Total time: {}s".format(end - overall_start))


if __name__ == '__main__':
    param = params.Params()
    train_ds, test_ds, count_train, count_test = read_fly_db()
    steps_per_epoch = tf.ceil(count_train / param.batch_size).numpy()
    print(steps_per_epoch)
    batches = 2 * steps_per_epoch + 1
    timeit(train_ds, batches)
    timeit(train_ds, batches)
