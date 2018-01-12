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

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    all_files = os.listdir(seg_folder)
    color_map = np.random.randint(0, 256, [256, 3]).astype(np.uint8)
    ignore_label = np.array([224, 224, 192])
    for name in all_files:
        all_diff_colors = []
        all_diff_colors.append(np.zeros((3,)))

        im_path = "{}/{}".format(seg_folder, name)
        im = imread(im_path)
        labels = np.zeros((im.shape[0], im.shape[1])).astype(np.uint8)
        for h in range(im.shape[0]):
            for w in range(im.shape[1]):
                p = im[h, w, :]

                if (p == ignore_label).all():
                    labels[h, w] = 255
                    continue

                idx = array_in_list(p, all_diff_colors)
                if idx is None:
                    all_diff_colors.append(p)
                labels[h, w] = array_in_list(p, all_diff_colors)

        save_path = "{}/{}".format(save_folder, name)
        imsave(save_path, labels)
        # im_test = imread(save_path)
        print save_path + " saved."

        if False:
            color_labels = color_map[labels]
            print all_diff_colors
            plt.figure(1)
            plt.imshow(im)
            plt.figure(2)
            plt.imshow(color_labels)
            plt.show()
        pass