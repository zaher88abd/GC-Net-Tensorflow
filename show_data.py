import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
from tqdm import tqdm


def plot_data(main_path):
    images = os.listdir(main_path)
    img_list = []
    for im in tqdm(images):
        img_list.append(cv2.imread(os.path.join(main_path, im), cv2.IMREAD_GRAYSCALE))
    arr_img=np.array(img_list)
    print(arr_img.shape)
    plt.hist(arr_img.flatten(), 256, [0, 256])
    plt.show()


if __name__ == '__main__':
    plot_data("/mnt/ExtStorge/Git/projects/GC-Net-Tensorflow/stereo_dataset/depth")
