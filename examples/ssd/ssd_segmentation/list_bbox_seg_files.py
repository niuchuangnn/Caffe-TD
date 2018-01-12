import string
import os

selected_file = '/home/amax/NiuChuang/data/VOCdevkit/VOC2012/ImageSets/Segmentation/train_5_per_cls.txt'

save_file = '/home/amax/NiuChuang/SSD/caffe-ssd/data/VOC0712/bbox_seg_trainval_5.txt'
f_w = open(save_file, 'w')

seg_folder = '/home/amax/NiuChuang/DecoupleNet/DecoupledNet/data/VOC2012/VOC2012_SEG_AUG/segmentations/'
image_subfolder = 'VOC2012/JPEGImages/'
bbox_subfolder = 'VOC2012/Annotations/'
seg_files = os.listdir(seg_folder)
seg_type = '.png'
img_type = '.jpg'
bbox_type = '.xml'

f = open(selected_file, 'r')
lines = f.readlines()

for line in lines:
    img_name = line.split('\n')[0]
    seg_name = img_name+seg_type
    if seg_name in seg_files:
        f_w.write(image_subfolder+img_name+img_type + ' ' + seg_name + ' ' + bbox_subfolder+img_name+bbox_type+'\n')
    else:
        f_w.write(image_subfolder+img_name+img_type + ' ' + '-1' + ' ' + bbox_subfolder+img_name+bbox_type+'\n')
    pass