import string
import os

ssd_train_file = '/home/amax/NiuChuang/SSD/caffe-ssd/data/VOC0712/trainval.txt'
ssd_test_file = '/home/amax/NiuChuang/SSD/caffe-ssd/data/VOC0712/test.txt'

save_file = '/home/amax/NiuChuang/SSD/caffe-ssd/data/VOC0712/bbox_seg_instance_trainval.txt'
f_w = open(save_file, 'w')

# seg_folder = '/home/amax/NiuChuang/DecoupleNet/DecoupledNet/data/VOC2012/VOC2012_SEG_AUG/segmentations/'
instance_folder = '/home/amax/NiuChuang/data/VOCdevkit/VOC2012/SegmentationObject/'

# seg_files = os.listdir(seg_folder)
# seg_type = '.png'

instance_files = os.listdir(instance_folder)
instance_type = '.png'

only_instance = True

f = open(ssd_train_file, 'r')
lines = f.readlines()

for line in lines:
    [fn_img, fn_bbox] = line.split(' ')
    img_name = fn_img.split('/')[-1][0:-4]
    # seg_name = img_name + seg_type
    # is_seg = seg_name in seg_files
    instance_name = img_name + instance_type
    is_instance = instance_name in instance_files

    if only_instance:
        if is_instance:
            f_w.write(fn_img + ' ' + instance_name + ' ' + fn_bbox)
    else:
        if is_instance:
            f_w.write(fn_img + ' ' + instance_name + ' ' + fn_bbox)
        else:
            f_w.write(fn_img + ' ' + '-1' + ' ' + fn_bbox)
    pass