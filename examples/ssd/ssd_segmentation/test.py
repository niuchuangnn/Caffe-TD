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
    caffe.set_device(1)
    caffe.set_mode_gpu()

    model_def = 'vgg16_ssd_seg.prototxt'
    net = caffe.Net(model_def, caffe.TRAIN)

    net.forward()

    # print [(k, v[0].data.shape) for k, v in net.params.items()]
    #
    # print [(k, v.data.shape) for k, v in net.blobs.items()]

    for k, v in net.params.items():
        print (k, v[0].data.shape)

    for k, v in net.blobs.items():
        print (k, v.data.shape)