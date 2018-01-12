import sys
sys.path.insert(0, '../../../python')
import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format
import matplotlib.pyplot as plt
import numpy as np
import json

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

if __name__ == '__main__':
    model_def = 'vgg16_ssd_seg.prototxt'
    model_weight = '/media/amax/data2/ssd_coco/models/VGGNet/coco/SSD_300x300/VGG_coco_SSD_300x300_iter_400000.caffemodel'
    caffe.set_device(0)
    caffe.set_mode_gpu()
    net = caffe.Net(model_def, model_weight, caffe.TRAIN)

    while True:
        net.forward()
        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2, 0, 1))
        transformer.set_mean('data', np.array([104, 117, 123]))  # mean pixel
        transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
        transformer.set_channel_swap('data', (2, 1, 0))  # the reference model has channels in BGR order instead of RGB

        plt.rcParams['figure.figsize'] = (10, 10)
        plt.rcParams['image.interpolation'] = 'nearest'

        colors = plt.cm.hsv(np.linspace(0, 1, 81)).tolist()

        img_blob = net.blobs['data'].data
        num_imgs = img_blob.shape[0]
        img_height = img_blob.shape[2]
        img_width = img_blob.shape[3]
        seg_blob = net.blobs['seg'].data
        label_blob = net.blobs['bbox'].data[0, 0, :, :]
        # cls = net.blobs['cls'].data
        num_labels = label_blob.shape[0]

        color_map = np.random.random_integers(0, 256, [256, 3]).astype(np.uint8)
        coco_id_to_name_path = '../cocodata_preprocessing/coco_id_to_name.json'
        coco_id_to_name = json.load(file(coco_id_to_name_path))
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
                    label = str(int(gt_bbox[1]))
                    currentAxis.text(xmin, ymin, coco_id_to_name[label], bbox={'facecolor': color, 'alpha': 0.5})

            plt.figure(2)
            plt.imshow(seg_color)
            label_name = 'no cls'
            # if cls[i].sum() > 0:
            #     label_id = np.argwhere(cls[i]==1)[0][0]
            #     label_name = coco_id_to_name[str(label_id + 1)]
            # else:
            #     label_name = "no bbox"
            plt.title(str(mask_labels) + label_name)

            plt.show()
