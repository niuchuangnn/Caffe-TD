import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread, imsave
import os

def array_in_list(array, list):
    idx = None
    for i in range(len(list)):
        a = list[i]
        if (array == a).all():
            idx = i
            break
    return idx

if __name__ == "__main__":
    seg_folder = '/home/amax/NiuChuang/fcn.berkeleyvision.org/data/pascal/2012/VOCdevkit/VOC2012/SegmentationObject'
    save_folder = '/home/amax/NiuChuang/DecoupleNet/DecoupledNet/data/VOC2012/2012_seg_instance'

    all_files = os.listdir(save_folder)
    color_map = np.random.randint(0, 256, [256, 3]).astype(np.uint8)
    for name in all_files:
        save_path = "{}/{}".format(save_folder, name)
        im_test = imread(save_path)
        print set(im_test.flatten())

        im_color = color_map[im_test]
        plt.figure(1)
        plt.imshow(im_color)
        plt.show()