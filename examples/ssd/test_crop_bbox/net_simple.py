import sys
sys.path.insert(0, '../../../python')

import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2
from caffe.model_libs import *
import numpy as np

n = caffe.NetSpec()

n.data = L.DummyData(shape=dict(dim=[2, 2, 6, 6]), data_filler=dict(type='constant', value=1))

n.data_conv = L.Convolution(n.data, kernel_size=3, pad=1, stride=1, num_output=2)

n.bbox = L.DummyData(shape=dict(dim=[1, 1, 2, 8]), data_filler=dict(type='constant', value=0))

n.data_crop = L.CropBBox(n.data_conv, n.bbox)

n.dummy_loss = L.EuclideanLoss(n.data_conv, n.data_crop)

net_def = "test_crop_bbox_simple.prototxt"
with open(net_def, 'w') as f:
    f.write(str(n.to_proto()))

net = caffe.Net(net_def, caffe.TEST)

img_labels = [[1], [2]]

bbox = np.zeros(net.blobs['bbox'].data.shape).astype(np.float)
bbox[0,0,0,0] = 0
bbox[0,0,0,1] = 1
bbox[0,0,0,2] = 0
bbox[0,0,0,3] = 0
bbox[0,0,0,4] = 0.2
bbox[0,0,0,5] = 0.8
bbox[0,0,0,6] = 0.8
bbox[0,0,0,7] = 0

bbox[0,0,1,0] = 1
bbox[0,0,1,1] = 2
bbox[0,0,1,2] = 0
bbox[0,0,1,3] = 0.4
bbox[0,0,1,4] = 0
bbox[0,0,1,5] = 0.6
bbox[0,0,1,6] = 1
bbox[0,0,1,7] = 0

net.blobs['bbox'].data[...] = bbox

data = np.zeros(net.blobs['data'].data.shape)
data[0, :, :, :] = 1
data[1, :, :, :] = 2

net.params['data_conv'][0].data[...] = 1

net.blobs['data'].data[...] = data

net.forward()

data_conv = net.blobs['data_conv'].data
bboxes = net.blobs['bbox'].data[0, 0]

data_crop = net.blobs['data_crop'].data
data_crop_test = np.zeros(data_crop.shape)

height = data_conv.shape[2]
width = data_conv.shape[3]
num_img = data_conv.shape[0]
num_bbox = bbox.shape[2]

idx = 0
for i in range(num_img):
    for l in range(len(img_labels[i])):

        for b in range(num_bbox):
            bbox_b = bboxes[b]
            im_id = bbox_b[0]
            label = bbox_b[1]
            if im_id == i and label == img_labels[i][l]:
                xmin_l = int(np.floor(bbox_b[3] * width))
                ymin_l = int(np.floor(bbox_b[4] * height))
                xmax_l = int(np.ceil(bbox_b[5] * width))
                ymax_l = int(np.ceil(bbox_b[6] * height))

                data_crop_test[idx, :, ymin_l:ymax_l, xmin_l:xmax_l] = data_conv[i, :, ymin_l:ymax_l, xmin_l:xmax_l]
        idx += 1
print np.abs(data_crop-data_crop_test).sum()


# test backward
net.blobs['data_crop'].diff[...] = np.random.randint(1, 10, net.blobs['data_crop'].diff.shape)
net.backward(start='data_crop')

top_diff = net.blobs['data_crop'].diff
data_diff = net.blobs['data_conv'].diff

data_diff_test = np.zeros(data_diff.shape)

idx = 0
for i in range(num_img):
    for l in range(len(img_labels[i])):

        for b in range(num_bbox):
            bbox_b = bboxes[b]
            im_id = bbox_b[0]
            label = bbox_b[1]
            if im_id == i and label == img_labels[i][l]:
                xmin_l = int(np.floor(bbox_b[3] * width))
                ymin_l = int(np.floor(bbox_b[4] * height))
                xmax_l = int(np.ceil(bbox_b[5] * width))
                ymax_l = int(np.ceil(bbox_b[6] * height))

                data_diff_test[idx, :, ymin_l:ymax_l, xmin_l:xmax_l] = top_diff[i, :, ymin_l:ymax_l, xmin_l:xmax_l]
        idx += 1

print (data_diff - data_diff_test).sum()

pass
