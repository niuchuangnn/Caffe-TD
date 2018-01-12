import numpy as np
import lmdb
import sys
sys.path.insert(0, './python')
import caffe

import matplotlib.pyplot as plt
import Image as img
import cv2
import json

coco_id_to_name_path = 'cocodata_preprocessing/coco_id_to_name.json'
coco_id_to_name = json.load(file(coco_id_to_name_path))

lmdb_path = '/media/amax/data2/MSCOCO/lmdb/coco2014_train_object_mask_lmdb'
env = lmdb.open(lmdb_path, readonly=True)

color_map = np.random.random_integers(0, 256, [256, 3]).astype(np.uint8)

txn = env.begin()
cursor = txn.cursor()
group_num_0 = 0
for key, value in cursor:
    print 'key:', key
    bbox_seg_datum = caffe.proto.caffe_pb2.BBoxSegDatum()
    bbox_seg_datum.ParseFromString(value)

    # note that if images have benn encoded and then they need to be decoded
    flat_x = np.fromstring(bbox_seg_datum.seg_datum.data, dtype=np.uint8)
    flat_x_decode = cv2.imdecode(flat_x, 1)
    # x = flat_x.reshape(bbox_seg_datum.datum.channels, bbox_seg_datum.datum.height, bbox_seg_datum.datum.width)
    y = bbox_seg_datum.seg_datum.label
    plt.figure(1)
    plt.imshow(flat_x_decode)
    ax = plt.gca()

    channels = bbox_seg_datum.seg_datum.channels
    height = bbox_seg_datum.seg_datum.height
    width = bbox_seg_datum.seg_datum.width

    annotation_group = bbox_seg_datum.annotation_group._values

    group_num = len(annotation_group)

    if group_num == 0:
        group_num_0 += 1
print group_num_0
        # raise AssertionError("group_num == 0")
    # fig_id = 0
    # for i in range(group_num):
    #     group = annotation_group[i]
    #     group_label = group.group_label
    #
    #     annotations = group.annotation._values
    #     instance_num = len(annotations)
    #     for j in range(instance_num):
    #         instance_id = annotations[j].instance_id
    #
    #         plt.figure(2+fig_id)
    #         fig_id += 1
    #         mask_i = np.fromstring(annotations[j].mask, dtype=np.uint8)
    #         mask_i_decode = cv2.imdecode(mask_i, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    #         plt.imshow(color_map[mask_i_decode])
    #         plt.title(coco_id_to_name[str(group_label)])
    #
    #         bbox = annotations[j].bbox
    #
    #         difficult = bbox.difficult
    #         label = bbox.label
    #         score = bbox.score
    #         size = bbox.size
    #         xmin_norm = bbox.xmin
    #         xmax_norm = bbox.xmax
    #         ymin_norm = bbox.ymin
    #         ymax_norm = bbox.ymax
    #
    #         xmin = int(xmin_norm * width)
    #         xmax = int(xmax_norm * width)
    #         ymin = int(ymin_norm * height)
    #         ymax = int(ymax_norm * height)
    #
    #         bbox_weight = xmax - xmin
    #         bbox_height = ymax - ymin
    #
    #         coords = (xmin, ymin), bbox_weight, bbox_height
    #         ax.add_patch(plt.Rectangle(*coords, fill=False, edgecolor='g', linewidth=3))
    #
    #         ax.text(xmin, ymin, coco_id_to_name[str(group_label)], bbox={'facecolor': 'b', 'alpha': 0.5})
    #
    # plt.show()