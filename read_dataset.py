import pandas as pd
import os
from tqdm import tqdm
import os.path as osp
import re
from PIL import Image
import numpy as np
import cv2


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


def dataset_csv(main_path):
    dirs = [osp.join(main_path, 'flyingthings3d_frames_cleanpass/'),
            osp.join(main_path, 'flyingthings3d__disparity/disparity/')]

    count = 0
    data_set = list()
    for phase in tqdm(['TRAIN', 'TEST']):
        for group in tqdm(['A', 'B', 'C']):
            dir_group = dirs[0] + phase + '/' + group
            dir_group2 = dirs[1] + phase + '/' + group
            for img_group in tqdm(os.listdir(dir_group)):
                dir_img_group = dir_group + '/' + img_group
                dir_dis_group = dir_group2 + '/' + img_group
                for img_name in tqdm(os.listdir(dir_img_group + '/left')):
                    img_path_1 = dir_img_group + '/left/' + img_name
                    img_path_2 = dir_img_group + '/right/' + img_name
                    disparity_path = dir_dis_group + '/left/' + img_name.split('.')[0] + '.pfm'
                    disparity = readPFM(disparity_path)[0]
                    cv2.imwrite(dir_dis_group + '/left/' + img_name.split('.')[0] + ".png", disparity)
                    data_set.append(
                        [img_path_1, img_path_2, dir_dis_group + '/left/' + img_name.split('.')[0] + ".png"])
                    count += 1

        pd.DataFrame(data_set, columns=["left", "right", "disp"]).to_csv(osp.join(main_path, phase+"data_set.csv"))


if __name__ == '__main__':
    dataset_csv("/home/zaher/Documents")
