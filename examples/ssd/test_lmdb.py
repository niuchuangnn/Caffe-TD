import numpy as np
import lmdb
import sys
sys.path.insert(0, './python')
import caffe

import matplotlib.pyplot as plt
import Image as img
import cv2

save_folder = '/media/amax/data2/test_lmdb'
img_list = ''

voc_classes = np.asarray(
        ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
         'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'])

lmdb_path = '/home/amax/NiuChuang/data/VOCdevkit/VOC0712/lmdb/VOC0712_trainval_lmdb'
env = lmdb.open(lmdb_path, readonly=True)

txn = env.begin()
cursor = txn.cursor()
for key, value in cursor:
    print 'key:', key
    annotated_datum = caffe.proto.caffe_pb2.AnnotatedDatum()
    annotated_datum.ParseFromString(value)
    datum = caffe.proto.caffe_pb2.Datum()

    # note that if images have benn encoded and then they need to be decoded
    flat_x = np.fromstring(annotated_datum.datum.data, dtype=np.uint8)
    flat_x_decode = cv2.imdecode(flat_x, 1)
    # x = flat_x.reshape(annotated_datum.datum.channels, annotated_datum.datum.height, annotated_datum.datum.width)
    y = annotated_datum.datum.label
    plt.figure(1)
    plt.imshow(flat_x_decode)

    channels = annotated_datum.datum.channels
    height = annotated_datum.datum.height
    width = annotated_datum.datum.width

    annotation_group = annotated_datum.annotation_group._values

    group_num = len(annotation_group)

    ax = plt.gca()
    for i in range(group_num):
        group = annotation_group[i]
        group_label = group.group_label

        annotations = group.annotation._values
        instance_num = len(annotations)
        for j in range(instance_num):
            instance_id = annotations[j].instance_id

            bbox = annotations[j].bbox

            difficult = bbox.difficult
            label = bbox.label
            score = bbox.score
            size = bbox.size
            xmin_norm = bbox.xmin
            xmax_norm = bbox.xmax
            ymin_norm = bbox.ymin
            ymax_norm = bbox.ymax

            xmin = int(xmin_norm * width)
            xmax = int(xmax_norm * width)
            ymin = int(ymin_norm * height)
            ymax = int(ymax_norm * height)

            bbox_weight = xmax - xmin
            bbox_height = ymax - ymin

            coords = (xmin, ymin), bbox_weight, bbox_height
            ax.add_patch(plt.Rectangle(*coords, fill=False, edgecolor='g', linewidth=3))

            ax.text(xmin, ymin, voc_classes[group_label-1], bbox={'facecolor': 'b', 'alpha': 0.5})

    plt.show()