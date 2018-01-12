from __future__ import print_function
import sys
sys.path.insert(0, '../../../python')
import caffe
from caffe.model_libs import *
from google.protobuf import text_format

import math
import os
import shutil
import stat
import subprocess

def AddExtraTopDownLayers(net, use_batchnorm=True, lr_mult=1):
    # mbox_source_layers = ['conv4_3', 'fc7', 'conv6_2', 'conv7_2', 'conv8_2', 'conv9_2']
    bbox = "cls_specific_bbox"

    use_relu = True
    # 1 x 1
    from_layer = net.keys()[-1]

    out_layer = "decls"
    DeConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 1, 0, 1, lr_mult=lr_mult)

    # crop feature form bottom-up net
    from_layer = "conv9_2"
    out_layer = "conv9_2_crop"
    net[out_layer] = L.CropBBox(net[from_layer], net[bbox])

    # concatenate the cropped feature and the class-specific top-down signals
    out_layer = "deconv9_2_concat"
    net[out_layer] = L.Concat(net["conv9_2_crop"], net["decls"])

    from_layer = out_layer
    out_layer = "deconv9_2"
    DeConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 128, 3, 0, 1,
                lr_mult=lr_mult)

    from_layer = out_layer
    out_layer = "deconv9_1"
    DeConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 1, 0, 1,
                lr_mult=lr_mult)

    # crop feature form bottom-up net
    from_layer = "conv8_2"
    out_layer = "conv8_2_crop"
    net[out_layer] = L.CropBBox(net[from_layer], net[bbox])

    # concatenate the cropped feature and the class-specific top-down signals
    out_layer = "deconv8_2_concat"
    net[out_layer] = L.Concat(net["conv8_2_crop"], net["deconv9_1"])

    # 3 x 3
    from_layer = out_layer
    out_layer = "deconv8_2"
    DeConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 128, 3, 0, 1,
      lr_mult=lr_mult)

    from_layer = out_layer
    out_layer = "deconv8_1"
    DeConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 1, 0, 1,
                lr_mult=lr_mult)

    # crop feature form bottom-up net
    from_layer = "conv7_2"
    out_layer = "conv7_2_crop"
    net[out_layer] = L.CropBBox(net[from_layer], net[bbox])

    # concatenate the cropped feature and the class-specific top-down signals
    out_layer = "deconv7_2_concat"
    net[out_layer] = L.Concat(net["conv7_2_crop"], net["deconv8_1"])

    # 5 x 5
    from_layer = out_layer
    out_layer = "deconv7_2"
    DeConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 128, 3, 1, 2,
                lr_mult=lr_mult)

    from_layer = out_layer
    out_layer = "deconv7_1"
    DeConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 512, 1, 0, 1,
      lr_mult=lr_mult)

    # crop feature form bottom-up net
    from_layer = "conv6_2"
    out_layer = "conv6_2_crop"
    net[out_layer] = L.CropBBox(net[from_layer], net[bbox])

    # concatenate the cropped feature and the class-specific top-down signals
    out_layer = "deconv6_2_concat"
    net[out_layer] = L.Concat(net["conv6_2_crop"], net["deconv7_1"])

    # 10 x 10
    from_layer = out_layer
    out_layer = "deconv6_2"
    DeConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 3, 1, 2,
                lr_mult=lr_mult)

    from_layer = out_layer
    out_layer = "deconv6_1"
    DeConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 1024, 1, 0, 1,
        lr_mult=lr_mult)

    return net

# Add extra layers on top of a "base" network (e.g. VGGNet or Inception).
def AddExtraLayers(net, use_batchnorm=True, lr_mult=1):
    use_relu = True

    # Add additional convolutional layers.
    # 19 x 19
    from_layer = net.keys()[-1]

    # TODO(weiliu89): Construct the name using the last layer to avoid duplication.
    # 10 x 10
    out_layer = "conv6_1"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 1, 0, 1,
        lr_mult=lr_mult)

    from_layer = out_layer
    out_layer = "conv6_2"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 512, 3, 1, 2,
        lr_mult=lr_mult)

    # 5 x 5
    from_layer = out_layer
    out_layer = "conv7_1"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 128, 1, 0, 1,
      lr_mult=lr_mult)

    from_layer = out_layer
    out_layer = "conv7_2"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 3, 1, 2,
      lr_mult=lr_mult)

    # 3 x 3
    from_layer = out_layer
    out_layer = "conv8_1"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 128, 1, 0, 1,
      lr_mult=lr_mult)

    from_layer = out_layer
    out_layer = "conv8_2"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 3, 0, 1,
      lr_mult=lr_mult)

    # 1 x 1
    from_layer = out_layer
    out_layer = "conv9_1"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 128, 1, 0, 1,
      lr_mult=lr_mult)

    from_layer = out_layer
    out_layer = "conv9_2"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 3, 0, 1,
      lr_mult=lr_mult)

    return net

# parameters
resize_height = 320
resize_width = 320

num_classes = 21
share_location = True
background_label_id=0
train_on_diff_gt = True
normalization_mode = P.Loss.VALID
code_type = P.PriorBox.CENTER_SIZE
ignore_cross_boundary_bbox = False
mining_type = P.MultiBoxLoss.MAX_NEGATIVE
neg_pos_ratio = 3.
loc_weight = (neg_pos_ratio + 1.) / 4.

# parameters for generating priors.
# minimum dimension of input image
min_dim = 320
# conv4_3 ==> 38 x 38
# fc7 ==> 19 x 19
# conv6_2 ==> 10 x 10
# conv7_2 ==> 5 x 5
# conv8_2 ==> 3 x 3
# conv9_2 ==> 1 x 1
mbox_source_layers = ['conv4_3', 'fc7', 'conv6_2', 'conv7_2', 'conv8_2', 'conv9_2']
# in percent %
min_ratio = 20
max_ratio = 90
step = int(math.floor((max_ratio - min_ratio) / (len(mbox_source_layers) - 2)))
min_sizes = []
max_sizes = []
for ratio in xrange(min_ratio, max_ratio + 1, step):
  min_sizes.append(min_dim * ratio / 100.)
  max_sizes.append(min_dim * (ratio + step) / 100.)
min_sizes = [min_dim * 10 / 100.] + min_sizes
max_sizes = [min_dim * 20 / 100.] + max_sizes
steps = [8, 16, 32, 64, 100, 300]
aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
# L2 normalize conv4_3.
normalizations = [20, -1, -1, -1, -1, -1]
# variance used to encode/decode prior bboxes.
if code_type == P.PriorBox.CENTER_SIZE:
  prior_variance = [0.1, 0.1, 0.2, 0.2]
else:
  prior_variance = [0.1]
flip = True
clip = False
use_batchnorm = False

# parameters for generating detection output.
det_out_param = {
    'num_classes': num_classes,
    'share_location': share_location,
    'background_label_id': background_label_id,
    'nms_param': {'nms_threshold': 0.45, 'top_k': 400},
    'keep_top_k': 200,
    'confidence_threshold': 0.01,
    'code_type': code_type,
    }


# Create test net.
net = caffe.NetSpec()
net.data = L.Input(shape=dict(dim=[1, 3, resize_height, resize_width]))

VGGNetBody(net, from_layer='data', fully_conv=True, reduced=True, dilated=True,
    dropout=False, pool_mask=True, freeze_all=True)

AddExtraLayers(net, use_batchnorm, lr_mult=0)

mbox_layers = CreateMultiBoxHead(net, data_layer='data', from_layers=mbox_source_layers,
        use_batchnorm=use_batchnorm, min_sizes=min_sizes, max_sizes=max_sizes,
        aspect_ratios=aspect_ratios, steps=steps, normalizations=normalizations,
        num_classes=num_classes, share_location=share_location, flip=flip, clip=clip,
        prior_variance=prior_variance, kernel_size=3, pad=1, lr_mult=0)

conf_name = "mbox_conf"
if P.MultiBoxLoss.SOFTMAX == P.MultiBoxLoss.SOFTMAX:
  reshape_name = "{}_reshape".format(conf_name)
  net[reshape_name] = L.Reshape(net[conf_name], shape=dict(dim=[0, -1, num_classes]))
  softmax_name = "{}_softmax".format(conf_name)
  net[softmax_name] = L.Softmax(net[reshape_name], axis=2)
  flatten_name = "{}_flatten".format(conf_name)
  net[flatten_name] = L.Flatten(net[softmax_name], axis=1)
  mbox_layers[1] = net[flatten_name]
elif P.MultiBoxLoss.SOFTMAX == P.MultiBoxLoss.LOGISTIC:
  sigmoid_name = "{}_sigmoid".format(conf_name)
  net[sigmoid_name] = L.Sigmoid(net[conf_name])
  mbox_layers[1] = net[sigmoid_name]

net.detection_out = L.DetectionOutput(*mbox_layers,
    detection_output_param=det_out_param,
    include=dict(phase=caffe_pb2.Phase.Value('TEST')))

net.cls = L.Input(shape=dict(dim=[1, 20]))
net.cls_specific_bbox = L.Input(shape=dict(dim=[1, 1, 1, 8]))

# class vector embedding deconvolution net for class-specific semantic segmentation
net.cls_reshape = L.Reshape(net.cls, shape=dict(dim=[0, 0, 1, 1]))

# add top-down deconvolution net
# mbox_source_layers = ['conv4_3', 'fc7', 'conv6_2', 'conv7_2', 'conv8_2', 'conv9_2']
AddExtraTopDownLayers(net, use_batchnorm=True, lr_mult=0)

DeVGGNetBody(net, from_layer='deconv6_1', fully_conv=True, reduced=True, dilated=True,
                dropout=False, pool_mask=True)

dekwargs = {
    'weight_filler': dict(type='xavier'),
    'bias_filler': dict(type='constant', value=0)}
deparam = {'param': [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)]}
net.seg_score = L.Deconvolution(net.derelu1_1, convolution_param=dict(num_output=2, pad=1, kernel_size=3, **dekwargs), **deparam)

net.seg_prob = L.Softmax(net.seg_score)


# Create deploy net.
# Remove the first and last layer from test net.
deploy_net_file = 'deploy.prototxt'
deploy_net = net
with open(deploy_net_file, 'w') as f:
    net_param = deploy_net.to_proto()
    print(net_param, file=f)