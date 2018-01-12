import string
import os

ssd_train_file = '/home/amax/NiuChuang/SSD/caffe/data/VOC0712/trainval.txt'
ssd_test_file = '/home/amax/NiuChuang/SSD/caffe/data/VOC0712/test.txt'

save_file = '/home/amax/NiuChuang/SSD/caffe/data/VOC0712/bbox_seg_trainval.txt'
f_w = open(save_file, 'w')

seg_folder = '/home/amax/NiuChuang/DecoupleNet/DecoupledNet/data/VOC2012/VOC2012_SEG_AUG/segmentations/'

seg_files = os.listdir(seg_folder)
seg_type = '.png'

f = open(ssd_train_file, 'r')
lines = f.readlines()

for line in lines:
    [fn_img, fn_bbox] = line.split(' ')
    img_name = fn_img.split('/')[-1][0:-4]
    seg_name = img_name + seg_type
    if seg_name in seg_files:
        f_w.write(fn_img + ' ' + seg_name + ' ' + fn_bbox)
    else:
        f_w.write(fn_img + ' -1 ' + fn_bbox)
    pass