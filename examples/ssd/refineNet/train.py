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
    # caffe.set_device(1)
    # caffe.set_mode_gpu()
    #
    # solver_def = 'solver.prototxt'
    # solver = caffe.SGDSolver(solver_def)
    #
    # solver.solve()
    resume = False
    pwd = '/home/amax/NiuChuang/SSD/caffe-ssd/examples/ssd/refineNet/'
    job_file = 'train.sh'
    solver_file = 'solver.prototxt'
    refineDet_model = '/media/amax/data2/RefineDet/models/VGGNet/VOC0712/refinedet_vgg16_320x320/VOC0712_refinedet_vgg16_320x320_final.caffemodel'
    refineDet_solver_state = ''
    if resume:
        train_src_param = '--snapshot="{}" \\\n'.format(refineDet_solver_state)
    else:
        train_src_param = '--weights="{}" \\\n'.format(refineDet_model)
    with open(job_file, 'w') as f:
        f.write('cd {}\n'.format(pwd))
        f.write('./../../../build/tools/caffe train \\\n')
        f.write('--solver="{}" \\\n'.format(solver_file))
        f.write(train_src_param)
        f.write('--gpu {} 2>&1 | tee {}.log\n'.format('1', './vgg16'))

    os.system(pwd+job_file)