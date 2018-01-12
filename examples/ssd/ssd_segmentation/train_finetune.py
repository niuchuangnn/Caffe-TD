import sys
sys.path.insert(0, '../../../python')

import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2

from caffe.model_libs import *
from google.protobuf import text_format

import math
import os
import shutil
import stat
import subprocess
import sys

if __name__ == "__main__":
    ssd_model = '/media/amax/data2/ssd/SSD_320x320/VGG_VOC0712_SSD_320x320_iter_120000.caffemodel'
    ssd_seg_model = '/media/amax/data2/ssd_seg/vgg16_iter_20000.caffemodel'
    ssd_detect_seg_pretrain = '/media/amax/data2/ssd_seg/vgg16_detect_seg_full_voc_pretrain.caffemodel'

    # caffe.set_device(1)
    # caffe.set_mode_gpu()
    #
    # model_def = 'vgg16_ssd_detect_seg.prototxt'
    # net = caffe.Net(model_def, caffe.TRAIN)
    # net.copy_from(ssd_model)
    # net.copy_from(ssd_seg_model)
    # net.save(ssd_detect_seg_pretrain)

    pwd = '/home/amax/NiuChuang/SSD/caffe-ssd/examples/ssd/ssd_segmentation/'
    job_file = 'train_finetune.sh'
    solver_file = 'solver_finetune.prototxt'

    train_src_param = '--weights="{}" \\\n'.format(ssd_detect_seg_pretrain)
    with open(job_file, 'w') as f:
        f.write('cd {}\n'.format(pwd))
        f.write('./../../../build/tools/caffe train \\\n')
        f.write('--solver="{}" \\\n'.format(solver_file))
        f.write(train_src_param)
        f.write('--gpu {} 2>&1 | tee {}.log\n'.format('0,1', './vgg16'))

    os.system(pwd+job_file)