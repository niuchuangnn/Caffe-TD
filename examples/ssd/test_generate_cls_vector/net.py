import sys
sys.path.insert(0, '../../../python')

import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2

def vgg16_ssd_seg(source, bbox_seg_data_param, kwargs):

        net = caffe.NetSpec()

        net.data, net.bbox, net.seg = L.BBoxSegData(name="data", bbox_seg_data_param=bbox_seg_data_param,
                data_param=dict(batch_size=48, backend=P.Data.LMDB, source=source),
                ntop=3, **kwargs)

        net.cls = L.GenerateClsVector(net.bbox, batch_size=48, num_class=80, background_label_id=0)
        return net.to_proto()

if __name__ == "__main__":
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
                        'max_trials': 200,
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
                        'max_trials': 200,
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
                        'max_trials': 200,
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
                        'max_trials': 200,
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
                        'max_trials': 200,
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
                        'max_trials': 200,
                        'max_sample': 1,
                },
        ]
        transform_param = {
                'mirror': True,
                'mean_value': [104, 117, 123],
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

        kwargs = {'include': dict(phase=caffe_pb2.Phase.Value('TRAIN')),
                  'transform_param': transform_param}

        label_map_file = "data/VOC0712/labelmap_voc.prototxt"
        bbox_seg_data_param = {
                'label_map_file': label_map_file,
                'batch_sampler': batch_sampler,
                'is_object_mask': True,
        }
        # source = "/home/amax/NiuChuang/data/VOCdevkit/VOC0712/lmdb/VOC0712_trainval_lmdb_test2"
        source = "/media/amax/data2/MSCOCO/lmdb/coco2014_train_object_mask_lmdb"

        with open("test_generate_cls_vector.prototxt", 'w') as f:
            f.write(str(vgg16_ssd_seg(source, bbox_seg_data_param, kwargs)))