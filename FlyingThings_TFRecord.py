from tqdm import tqdm
import os
import tensorflow as tf
from PIL import Image
import re
import numpy as np
import cv2
import params


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


def read_fly_db():
    cwd = os.getcwd()
    dirs = [cwd + '/' + 'flyingthings3d_frames_cleanpass/',
            cwd + '/' + 'flyingthings3d__disparity/disparity/']

    writer_tr = tf.python_io.TFRecordWriter("dataset/fly_train.tfrecords")
    writer_ts = tf.python_io.TFRecordWriter("dataset/fly_test.tfrecords")

    count = 0
    for phase in tqdm(['TRAIN', 'TEST']):
        for group in tqdm(['A', 'B', 'C']):
            dir_group = dirs[0] + phase + '/' + group
            dir_group2 = dirs[1] + phase + '/' + group
            for img_group in tqdm(os.listdir(dir_group)):
                dir_img_group = dir_group + '/' + img_group
                dir_dis_group = dir_group2 + '/' + img_group
                for img_name in tqdm(os.listdir(dir_img_group + '/left')):
                    img_path_1 = dir_img_group + '/left/' + img_name
                    img_1 = Image.open(img_path_1)
                    # img_1 = img_1.resize((width, height))
                    # img_1 = rgba_to_rgb(img_1)
                    img_1 = np.array(img_1)
                    img_1_raw = img_1.tobytes()

                    img_path_2 = dir_img_group + '/right/' + img_name
                    img_2 = Image.open(img_path_2)
                    # img_2 = img_2.resize((width, height))
                    # img_2 = rgba_to_rgb(img_2)
                    img_2 = np.array(img_2)
                    img_2_raw = img_2.tobytes()

                    disparity_path = dir_dis_group + '/left/' + img_name.split('.')[0] + '.pfm'
                    disparity = readPFM(disparity_path)[0]

                    disparity_raw = disparity.tobytes()

                    example = tf.train.Example(features=tf.train.Features(feature={
                        "img_left": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_1_raw])),
                        'img_right': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_2_raw])),
                        'disparity': tf.train.Feature(bytes_list=tf.train.BytesList(value=[disparity_raw]))}))

                    count += 1
                    if phase == 'TRAIN':
                        writer_tr.write(example.SerializeToString())
                    else:
                        writer_ts.write(example.SerializeToString())

    writer_tr.close()
    writer_ts.close()


def read_db(main_path, scaling=False):
    param = params.Params()
    l_img_path = os.path.join(main_path, 'left')
    r_img_path = os.path.join(main_path, 'right')
    d_img_path = os.path.join(main_path, 'depth')

    writer_tr = tf.python_io.TFRecordWriter("dataset/my_train.tfrecords")
    writer_ts = tf.python_io.TFRecordWriter("dataset/my_test.tfrecords")

    count = 0
    train_counter = 0
    test_counter = 0
    images_ = os.listdir(l_img_path)
    np.random.shuffle(images_)
    p_test = 21
    for img_name in tqdm(images_):
        img_path_1 = os.path.join(l_img_path, img_name)
        img_1 = cv2.imread(img_path_1, cv2.IMREAD_COLOR)
        # img_1 = cv2.resize(img_1, (param.original_w, param.original_h))
        img_1 = np.array(img_1)

        img_1_raw = img_1.tobytes()

        img_path_2 = os.path.join(r_img_path, img_name)
        img_2 = cv2.imread(img_path_2, cv2.IMREAD_COLOR)
        # img_2 = cv2.resize(img_2, (param.original_w, param.original_h))
        img_2 = np.array(img_2)

        img_2_raw = img_2.tobytes()

        disparity_path = os.path.join(d_img_path, img_name)
        disparity = cv2.imread(disparity_path, cv2.IMREAD_GRAYSCALE)
        # disparity = cv2.resize(disparity, (param.original_w, param.original_h))
        disparity = np.array(disparity, dtype=np.float32)
        if scaling:
            disparity = (param.max_disparity / disparity.max()) * (disparity - disparity.max()) + param.max_disparity
        disparity_raw = disparity.tobytes()
        # print(disparity_raw.shape)

        example = tf.train.Example(features=tf.train.Features(feature={
            "img_left": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_1_raw])),
            'img_right': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_2_raw])),
            'disparity': tf.train.Feature(bytes_list=tf.train.BytesList(value=[disparity_raw]))}))

        count += 1
        if p_test > count:
            test_images = True
            test_counter += 1
        else:
            test_images = False
            train_counter += 1

        if test_images:
            writer_ts.write(example.SerializeToString())
        else:
            writer_tr.write(example.SerializeToString())

    writer_tr.close()
    writer_ts.close()
    print("Train dataset = ", train_counter)
    print("Test dataset = ", test_counter)


if __name__ == '__main__':
    read_db("./stereo_dataset")
    # read_fly_db()
