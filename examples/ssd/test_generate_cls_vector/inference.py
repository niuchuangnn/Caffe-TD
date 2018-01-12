import numpy as np
import lmdb
# import sys
# sys.path.insert(0, './python')
import caffe

from caffe.proto import caffe_pb2
from google.protobuf import text_format
import matplotlib.pyplot as plt
import numpy as np

def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in xrange(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found == True
    return labelnames


# take an array of shape (n, height, width) or (n, height, width, channels)
# and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
def vis_square(data, padsize=1, padval=0):
    data -= data.min()
    data /= data.max()

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    plt.imshow(data)


if __name__ == "__main__":
    caffe.set_device(1)
    caffe.set_mode_gpu()
    caffe.set_random_seed(1000)
    net_def = 'test_generate_cls_vector.prototxt'
    net = caffe.Net(net_def, caffe.TRAIN)

    net.forward()

    # print net.blobs['data'].data.shape
    # print net.blobs['bbox'].data.shape
    # print net.blobs['seg'].data.shape

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', np.array([104, 117, 123]))  # mean pixel
    transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2, 1, 0))  # the reference model has channels in BGR order instead of RGB

    labelmap_file = '../../../data/VOC0712/labelmap_voc.prototxt'
    file = open(labelmap_file, 'r')
    labelmap = caffe_pb2.LabelMap()
    text_format.Merge(str(file.read()), labelmap)

    plt.rcParams['figure.figsize'] = (10, 10)
    plt.rcParams['image.interpolation'] = 'nearest'

    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()

    img_blob = net.blobs['data'].data
    num_imgs = img_blob.shape[0]
    img_height = img_blob.shape[2]
    img_width = img_blob.shape[3]
    seg_blob = net.blobs['seg'].data
    label_blob = net.blobs['bbox'].data[0, 0, :, :]
    num_labels = label_blob.shape[0]
    # cls = net.blobs['cls'].data

    color_map = np.random.random_integers(0, 256, [256, 3]).astype(np.uint8)

    select_cls = net.blobs['cls'].data
    cls_specific_bbox = net.blobs['cls_specific_bbox'].data[0, 0, :, :]
    binary_mask = net.blobs['binary_mask'].data
    num_selected_bbox = cls_specific_bbox.shape[0]

    # visualize class specific bboxes, binary mask
    for i in xrange(num_imgs):
        img = transformer.deprocess('data', img_blob[i])
        binary_seg = binary_mask[i, 0].astype(np.uint8)
        binary_seg_labels = set(list(binary_seg.flatten()))
        binary_seg_color = color_map[binary_seg]
        plt.figure(1)
        plt.imshow(img)
        currentAxis = plt.gca()
        for j in xrange(num_selected_bbox):
            gt_bbox = cls_specific_bbox[j, :]
            if gt_bbox[0] == i:
                xmin = gt_bbox[3] * img_width
                ymin = gt_bbox[4] * img_height
                xmax = gt_bbox[5] * img_width
                ymax = gt_bbox[6] * img_height
                gt_label = int(gt_bbox[1])
                coords = (xmin, ymin), xmax - xmin + 1, ymax - ymin + 1
                color = colors[gt_label]
                currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
                label = get_labelname(labelmap, gt_bbox[1])[0]
                currentAxis.text(xmin, ymin, label, bbox={'facecolor': color, 'alpha': 0.5})

        plt.figure(2)
        plt.imshow(binary_seg_color)

        label_vec = select_cls[i]
        if label_vec.sum() > 0:
            label_id = np.argwhere(label_vec==1)[0][0] + 1
            label_name = get_labelname(labelmap, label_id)[0]
        else:
            label_name = "No bbox"
        plt.title(str(binary_seg_labels) + " calss: " + label_name)

        plt.show()

    # visualize the input data and label
    for i in xrange(num_imgs):
        img = transformer.deprocess('data', img_blob[i])
        seg = seg_blob[i, 0].astype(np.uint8)
        mask_labels = set(list(seg.flatten()))
        seg_color = color_map[seg]
        # plt.subplot(1, num_imgs, i + 1)
        plt.figure(1)
        plt.imshow(img)
        currentAxis = plt.gca()
        for j in xrange(num_labels):
            gt_bbox = label_blob[j, :]
            if gt_bbox[0] == i:
                xmin = gt_bbox[3] * img_width
                ymin = gt_bbox[4] * img_height
                xmax = gt_bbox[5] * img_width
                ymax = gt_bbox[6] * img_height
                gt_label = int(gt_bbox[1])
                coords = (xmin, ymin), xmax - xmin + 1, ymax - ymin + 1
                color = colors[gt_label]
                currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
                label = get_labelname(labelmap, gt_bbox[1])[0]
                currentAxis.text(xmin, ymin, label, bbox={'facecolor': color, 'alpha': 0.5})

        plt.figure(2)
        plt.imshow(seg_color)
        plt.title(str(mask_labels))

        plt.show()

