from __future__ import print_function
import sys
sys.path.append("./python")
import caffe
from caffe.model_libs import *
from google.protobuf import text_format

import math
import os
import shutil
import stat
import subprocess

# Add extra layers on top of a "base" network (e.g. VGGNet or ResNet).
def AddExtraLayers(net, use_batchnorm=True, arm_source_layers=[], normalizations=[], lr_mult=1):
    use_relu = True

    # Add additional convolutional layers.
    # 320/32: 10 x 10
    from_layer = net.keys()[-1]

    # 320/64: 5 x 5
    out_layer = "conv6_1"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 1, 0, 1, lr_mult=lr_mult)

    from_layer = out_layer
    out_layer = "conv6_2"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 512, 3, 1, 2, lr_mult=lr_mult)

    arm_source_layers.reverse()
    normalizations.reverse()
    num_p = 6
    for index, layer in enumerate(arm_source_layers):
        out_layer = layer
        if normalizations:
            if normalizations[index] != -1:
                norm_name = "{}_norm".format(layer)
                net[norm_name] = L.Normalize(net[layer], scale_filler=dict(type="constant", value=normalizations[index]),
                    across_spatial=False, channel_shared=False)
                out_layer = norm_name
                arm_source_layers[index] = norm_name
        from_layer = out_layer
        out_layer = "TL{}_{}".format(num_p, 1)
        ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 3, 1, 1, lr_mult=lr_mult)

        if num_p == 6:
            from_layer = out_layer
            out_layer = "TL{}_{}".format(num_p, 2)
            ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 3, 1, 1, lr_mult=lr_mult)

            from_layer = out_layer
            out_layer = "P{}".format(num_p)
            ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 3, 1, 1, lr_mult=lr_mult)
        else:
            from_layer = out_layer
            out_layer = "TL{}_{}".format(num_p, 2)
            ConvBNLayer(net, from_layer, out_layer, use_batchnorm, False, 256, 3, 1, 1, lr_mult=lr_mult)

            from_layer = "P{}".format(num_p+1)
            out_layer = "P{}-up".format(num_p+1)
            DeconvBNLayerRef(net, from_layer, out_layer, use_batchnorm, False, 256, 2, 0, 2, lr_mult=lr_mult)

            from_layer = ["TL{}_{}".format(num_p, 2), "P{}-up".format(num_p+1)]
            out_layer = "Elt{}".format(num_p)
            EltwiseLayer(net, from_layer, out_layer)
            relu_name = '{}_relu'.format(out_layer)
            net[relu_name] = L.ReLU(net[out_layer], in_place=True)
            out_layer = relu_name

            from_layer = out_layer
            out_layer = "P{}".format(num_p)
            ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 3, 1, 1, lr_mult=lr_mult)

        num_p = num_p - 1

    return net

def AddExtraTopDownLayers(net, use_batchnorm=True, lr_mult=1):
    # odm_source_layers = ['P3', 'P4', 'P5', 'P6']
    bbox = "cls_specific_bbox"

    use_relu = True
    # 5 x 5
    # crop feature form bottom-up net
    from_layer = "P6"
    out_layer = "conv6_2_crop"
    net[out_layer] = L.CropBBox(net[from_layer], net[bbox])

    from_layer = out_layer
    out_layer = "deconv6_2"
    DeconvBNLayerRef(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 3, 1, 2,
                lr_mult=lr_mult)

    from_layer = out_layer
    out_layer = "deconv6_1"
    DeconvBNLayerRef(net, from_layer, out_layer, use_batchnorm, use_relu, 1024, 1, 0, 1,
                lr_mult=lr_mult)

    return net

if __name__ == "__main__":

    # The database file for training data. Created by data/coco/create_data.sh
    train_data = "/home/amax/NiuChuang/data/VOCdevkit/VOC0712/lmdb/VOC0712_trainval_lmdb_test2"
    # The database file for testing data. Created by data/coco/create_data.sh
    test_data = "examples/coco/coco_minival_lmdb"
    # Specify the batch sampler.
    resize_width = 320
    resize_height = 320
    resize = "{}x{}".format(resize_width, resize_height)
    batch_sampler = [
            {
                    'sampler': {
                            },
                    'max_trials': 1,
                    'max_sample': 1,
            },
            {
                    'sampler': {
                            'min_scale': 0.3,
                            'max_scale': 1.0,
                            'min_aspect_ratio': 0.5,
                            'max_aspect_ratio': 2.0,
                            },
                    'sample_constraint': {
                            'min_jaccard_overlap': 0.1,
                            },
                    'max_trials': 50,
                    'max_sample': 1,
            },
            {
                    'sampler': {
                            'min_scale': 0.3,
                            'max_scale': 1.0,
                            'min_aspect_ratio': 0.5,
                            'max_aspect_ratio': 2.0,
                            },
                    'sample_constraint': {
                            'min_jaccard_overlap': 0.3,
                            },
                    'max_trials': 50,
                    'max_sample': 1,
            },
            {
                    'sampler': {
                            'min_scale': 0.3,
                            'max_scale': 1.0,
                            'min_aspect_ratio': 0.5,
                            'max_aspect_ratio': 2.0,
                            },
                    'sample_constraint': {
                            'min_jaccard_overlap': 0.5,
                            },
                    'max_trials': 50,
                    'max_sample': 1,
            },
            {
                    'sampler': {
                            'min_scale': 0.3,
                            'max_scale': 1.0,
                            'min_aspect_ratio': 0.5,
                            'max_aspect_ratio': 2.0,
                            },
                    'sample_constraint': {
                            'min_jaccard_overlap': 0.7,
                            },
                    'max_trials': 50,
                    'max_sample': 1,
            },
            {
                    'sampler': {
                            'min_scale': 0.3,
                            'max_scale': 1.0,
                            'min_aspect_ratio': 0.5,
                            'max_aspect_ratio': 2.0,
                            },
                    'sample_constraint': {
                            'min_jaccard_overlap': 0.9,
                            },
                    'max_trials': 50,
                    'max_sample': 1,
            },
            {
                    'sampler': {
                            'min_scale': 0.3,
                            'max_scale': 1.0,
                            'min_aspect_ratio': 0.5,
                            'max_aspect_ratio': 2.0,
                            },
                    'sample_constraint': {
                            'max_jaccard_overlap': 1.0,
                            },
                    'max_trials': 50,
                    'max_sample': 1,
            },
            ]
    train_transform_param = {
            'mirror': True,
            'mean_value': [104, 117, 123],
            'force_color': True,
            'resize_param': {
                    'prob': 1,
                    'resize_mode': P.Resize.WARP,
                    'height': resize_height,
                    'width': resize_width,
                    'interp_mode': [
                            P.Resize.LINEAR,
                            P.Resize.AREA,
                            P.Resize.NEAREST,
                            P.Resize.CUBIC,
                            P.Resize.LANCZOS4,
                            ],
                    },
            'distort_param': {
                    'brightness_prob': 0.5,
                    'brightness_delta': 32,
                    'contrast_prob': 0.5,
                    'contrast_lower': 0.5,
                    'contrast_upper': 1.5,
                    'hue_prob': 0.5,
                    'hue_delta': 18,
                    'saturation_prob': 0.5,
                    'saturation_lower': 0.5,
                    'saturation_upper': 1.5,
                    'random_order_prob': 0.0,
                    },
            'expand_param': {
                    'prob': 0.5,
                    'max_expand_ratio': 4.0,
                    },
            'emit_constraint': {
                'emit_type': caffe_pb2.EmitConstraint.CENTER,
                }
            }
    test_transform_param = {
            'mean_value': [104, 117, 123],
            'force_color': True,
            'resize_param': {
                    'prob': 1,
                    'resize_mode': P.Resize.WARP,
                    'height': resize_height,
                    'width': resize_width,
                    'interp_mode': [P.Resize.LINEAR],
                    },
            }

    # If true, use batch norm for all newly added layers.
    # Currently only the non batch norm version has been tested.
    use_batchnorm = False
    lr_mult = 0

    # Stores LabelMapItem.
    label_map_file = "data/VOC0712/labelmap_voc.prototxt"

    # parameters for generating priors.
    # minimum dimension of input image
    # min_dim = 320
    # conv4_3 ==> 40 x 40
    # conv5_3 ==> 20 x 20
    # fc7 ==> 10 x 10
    # conv6_2 ==> 5 x 5
    arm_source_layers = ['conv4_3', 'conv5_3', 'fc7', 'conv6_2']
    odm_source_layers = ['P3', 'P4', 'P5', 'P6']
    # L2 normalize conv4_3 and conv5_3.
    normalizations = [10, 8, -1, -1]

    # Create train net.
    net = caffe.NetSpec()

    bbox_seg_data_param = {
                    'label_map_file': label_map_file,
                    'batch_sampler': batch_sampler,
            }
    kwargs = {'include': dict(phase=caffe_pb2.Phase.Value('TRAIN')),
              'transform_param': train_transform_param}

    net.data, net.bbox, net.seg = L.BBoxSegData(name="data", bbox_seg_data_param=bbox_seg_data_param,
                    data_param=dict(batch_size=8, backend=P.Data.LMDB, source=train_data),
                    ntop=3, **kwargs)

    net.cls_specific_bbox, net.binary_mask, net.cls = L.SelectBinary(net.bbox, net.seg, random_select=True, num_class=20, ntop=3)
    net.__setattr__('cls_silence', L.Silence(net.cls, ntop=0))

    VGGNetBody(net, from_layer='data', fully_conv=True, reduced=True, dilated=False, dropout=False, pool_mask=True, freeze_all=True)

    AddExtraLayers(net, use_batchnorm, arm_source_layers, normalizations, lr_mult=0)

    AddExtraTopDownLayers(net, use_batchnorm=True, lr_mult=1)

    DeVGGNetBodyRef(net, from_layer='deconv6_1', fully_conv=True, reduced=True, dilated=False,
                         dropout=False, pool_mask=True, extra_crop_layers=[])

    dekwargs = {'weight_filler': dict(type='xavier'),
                'bias_filler': dict(type='constant', value=0)}
    deparam = {'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)]}
    net.seg_score = L.Deconvolution(net.derelu1_1, convolution_param=dict(num_output=2, pad=1, kernel_size=3, **dekwargs), **deparam)

    net.seg_loss = L.SoftmaxWithLoss(net.seg_score, net.binary_mask, loss_param=dict(ignore_label=255))

    with open('examples/ssd/refineNet/vgg16_refnet_seg_voc.prototxt', 'w') as f:
        f.write(str(net.to_proto()))