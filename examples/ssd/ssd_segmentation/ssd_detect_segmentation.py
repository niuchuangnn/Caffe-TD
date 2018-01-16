#encoding=utf8
'''
Detection with SSD
In this example, we will load a SSD model and use it to detect objects.
'''

import os
import sys
import argparse
import numpy as np
from PIL import Image, ImageDraw
# Make sure that caffe is on the python path:
caffe_root = '../../../'
os.chdir(caffe_root)
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe

from google.protobuf import text_format
from caffe.proto import caffe_pb2

import matplotlib.pyplot as plt
from scipy.misc import imresize, imread
import xml.dom.minidom

voc_classes = np.asarray(
        ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
         'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'])

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

def parseXmlVoc(xmlFile):
    dom = xml.dom.minidom.parse(xmlFile)
    root = dom.documentElement
    objects = root.getElementsByTagName('object')
    object_names = []
    objects_attr = []
    for i in range(len(objects)):
        object = objects[i]
        cls_name = object.getElementsByTagName('name')[0]
        object_names.append(cls_name.childNodes[0].data)
        difficult = object.getElementsByTagName('difficult')[0].childNodes[0].data
        truncated = object.getElementsByTagName('truncated')[0].childNodes[0].data
        pose = object.getElementsByTagName('pose')[0].childNodes[0].data
        bndbox = object.getElementsByTagName('bndbox')[0]
        xmin = bndbox.getElementsByTagName('xmin')[0].childNodes[0].data
        xmax = bndbox.getElementsByTagName('xmax')[0].childNodes[0].data
        ymin = bndbox.getElementsByTagName('ymin')[0].childNodes[0].data
        ymax = bndbox.getElementsByTagName('ymax')[0].childNodes[0].data
        objects_attr.append({})
        objects_attr[i]['difficult'] = difficult
        objects_attr[i]['truncated'] = truncated
        objects_attr[i]['pose'] = pose
        objects_attr[i]['xmin'] = xmin
        objects_attr[i]['xmax'] = xmax
        objects_attr[i]['ymin'] = ymin
        objects_attr[i]['ymax'] = ymax
    return object_names, objects_attr

class CaffeDetectionSegmentation:
    def __init__(self, gpu_id, bbox_margin, thresh, model_def, model_weights_detection, model_weights_segmentation,
                 image_resize, labelmap_file, test_file, image_folder, bbox_folder, seg_folder):
        caffe.set_device(gpu_id)
        caffe.set_mode_gpu()

        self.image_resize = image_resize
        # Load the net in the test phase for inference, and configure input preprocessing.
        self.net = caffe.Net(model_def,      # defines the structure of the model
                             model_weights_detection,  # contains the trained weights
                             caffe.TEST)     # use test mode (e.g., don't perform dropout)
        self.net.copy_from(model_weights_segmentation)
        # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2, 0, 1))
        self.transformer.set_mean('data', np.array([104, 117, 123])) # mean pixel
        # the reference model operates on images in [0,255] range instead of [0,1]
        self.transformer.set_raw_scale('data', 255)
        # the reference model has channels in BGR order instead of RGB
        self.transformer.set_channel_swap('data', (2, 1, 0))
        self.im_shape = []
        self.bbox_margin = bbox_margin
        self.thresh = thresh
        self.test_file = test_file
        self.image_folder = image_folder
        self.bbox_folder = bbox_folder
        self.seg_folder = seg_folder

        # load PASCAL VOC labels
        file = open(labelmap_file, 'r')
        self.labelmap = caffe_pb2.LabelMap()
        text_format.Merge(str(file.read()), self.labelmap)

    def detect(self, image_file, conf_thresh=0.5, topn=5):
        '''
        SSD detection
        '''
        # set net to batch size of 1
        # image_resize = 300
        self.net.blobs['data'].reshape(1, 3, self.image_resize, self.image_resize)
        image = caffe.io.load_image(image_file)

        #Run the net and examine the top_k results
        transformed_image = self.transformer.preprocess('data', image)
        self.net.blobs['data'].data[...] = transformed_image

        # Forward pass.
        detections = self.net.forward(start='data', end='detection_out')['detection_out']

        # Parse the outputs.
        det_label = detections[0,0,:,1]
        det_conf = detections[0,0,:,2]
        det_xmin = detections[0,0,:,3]
        det_ymin = detections[0,0,:,4]
        det_xmax = detections[0,0,:,5]
        det_ymax = detections[0,0,:,6]

        # Get detections with confidence higher than 0.6.
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= conf_thresh]

        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_labels = get_labelname(self.labelmap, top_label_indices)
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]

        result = []
        for i in xrange(min(topn, top_conf.shape[0])):
            xmin = top_xmin[i] # xmin = int(round(top_xmin[i] * image.shape[1]))
            ymin = top_ymin[i] # ymin = int(round(top_ymin[i] * image.shape[0]))
            xmax = top_xmax[i] # xmax = int(round(top_xmax[i] * image.shape[1]))
            ymax = top_ymax[i] # ymax = int(round(top_ymax[i] * image.shape[0]))
            score = top_conf[i]
            label = int(top_label_indices[i])
            label_name = top_labels[i]
            result.append([xmin, ymin, xmax, ymax, label, score, label_name])
        return result

    def detect2bbox(self, detect_results):
        # compute all different class labels
        all_diff_cls_labels = []
        for item in detect_results:
            if item[4] not in all_diff_cls_labels:
                all_diff_cls_labels.append(item[4])

        bbox = np.zeros((1, 1, len(detect_results), 8))
        for i in range(len(detect_results)):
            item = detect_results[i]
            bbox[0, 0, i, 0] = all_diff_cls_labels.index(item[4])
            bbox[0, 0, i, 1] = item[4]
            bbox[0, 0, i, 2] = i
            bbox[0, 0, i, 3] = item[0]
            bbox[0, 0, i, 4] = item[1]
            bbox[0, 0, i, 5] = item[2]
            bbox[0, 0, i, 6] = item[3]
            # print bbox[0, 0, i, 3]
            # print bbox[0, 0, i, 4]
            # print bbox[0, 0, i, 5]
            # print bbox[0, 0, i, 6]

            bbox[0, 0, i, 7] = 0
        return bbox, all_diff_cls_labels

    def seg(self, bbox, all_diff_cls_labels, image_file, key_type='name', nocls=False):

        num_diff_cls_labels = len(all_diff_cls_labels)

        self.net.blobs['data'].reshape(1, 3, self.image_resize, self.image_resize)
        image = caffe.io.load_image(image_file)

        im_shape = image.shape
        self.im_shape = im_shape

        # Run the net and examine the top_k results
        transformed_image = self.transformer.preprocess('data', image)

        self.net.blobs['data'].reshape(num_diff_cls_labels, 3, self.image_resize, self.image_resize)

        cls = np.zeros((num_diff_cls_labels, 20))
        for i in range(num_diff_cls_labels):
            self.net.blobs['data'].data[i, ...] = transformed_image

            cls[i, all_diff_cls_labels[i]-1] = 1

        if not nocls:
            self.net.blobs['cls'].reshape(num_diff_cls_labels, 20)
            self.net.blobs['cls'].data[...] = cls

        self.net.blobs['cls_specific_bbox'].reshape(1, 1, bbox.shape[2], 8)
        self.net.blobs['cls_specific_bbox'].data[...] = bbox

        # Forward pass.
        seg_prob_blob = self.net.forward()['seg_prob']
        seg_prob = {}
        for i in range(num_diff_cls_labels):
            name = get_labelname(self.labelmap, all_diff_cls_labels[i])
            if key_type == 'name':
                seg_prob[name[0]] = imresize(seg_prob_blob[i, 1], [im_shape[0], im_shape[1]])
            elif key_type == 'label':
                seg_prob[all_diff_cls_labels[i]] = imresize(seg_prob_blob[i, 1], [im_shape[0], im_shape[1]])/255.0

        return seg_prob

    def merge_seg_prob(self, seg_prob, bbox, all_diff_labels):
        mask = np.zeros((self.im_shape[0], self.im_shape[1], len(seg_prob)+1))
        bbox_margin = self.bbox_margin
        bbox = bbox[0, 0]
        num_bbox = bbox.shape[0]

        for label, seg in seg_prob.iteritems():
            for i in range(num_bbox):
                if bbox[i, 1] == label:
                    xmin = np.floor(bbox[i, 3] * self.im_shape[1])
                    ymin = np.floor(bbox[i, 4] * self.im_shape[0])
                    xmax = np.ceil(bbox[i, 5] * self.im_shape[1])
                    ymax = np.ceil(bbox[i, 6] * self.im_shape[0])
                    xmin = max(0, xmin - bbox_margin)
                    ymin = max(0, ymin - bbox_margin)
                    xmax = min(self.im_shape[1], xmax + bbox_margin)
                    ymax = min(self.im_shape[0], ymax + bbox_margin)
                    seg[np.where(seg < self.thresh)] = 0
                    mask[ymin:ymax, xmin:xmax, all_diff_labels.index(label)+1] = seg[ymin:ymax, xmin:xmax]
        if False:
            fig_id = 111
            for i in range(len(all_diff_labels)):
                plt.figure(fig_id+i)
                plt.imshow(mask[:, :, i+1])

        mask_index = np.argmax(mask, axis=2)
        mask_label = np.zeros(mask_index.shape).astype(np.uint8)
        for label in all_diff_labels:
            mask_label[np.where(mask_index==all_diff_labels.index(label)+1)] = label

        return mask_label

    def test_seg_accuracy(self, image_type='.jpg', seg_type='.png', bbox_type='.xml', use_bbox_gt=True):
        test_file = self.test_file
        image_folder = self.image_folder
        bbox_folder = self.bbox_folder
        seg_folder = self.seg_folder

        f = open(test_file, 'r')
        lines = f.readlines()

        intersection_class_compare = np.zeros(voc_classes.shape)
        union_class_compare = np.zeros(voc_classes.shape)

        for line in lines:
            name = line.split('\n')[0]
            image_file = image_folder + name + image_type
            seg_file = seg_folder + name + seg_type
            bbox_file = bbox_folder + name + bbox_type
            object_names, object_attr = parseXmlVoc(bbox_file)
            im = imread(image_file)
            im_shape = im.shape
            if use_bbox_gt:
                detect = []
                for i in range(len(object_attr)):
                    attr = object_attr[i]
                    object_name = object_names[i]
                    label = np.argwhere(voc_classes==object_name)[0][0] + 1
                    xmin = float(attr['xmin']) / im_shape[1]
                    ymin = float(attr['ymin']) / im_shape[0]
                    xmax = float(attr['xmax']) / im_shape[1]
                    ymax = float(attr['ymax']) / im_shape[0]

                    detect.append([xmin, ymin, xmax, ymax, label, 1, object_name])
            else:
                detect = self.detect(image_file)
                # print len(detect)
            num_detect = len(detect)

            if num_detect > 0:
                bbox, all_diff_cls_labels = self.detect2bbox(detect)

                seg_prob = self.seg(bbox, all_diff_cls_labels, image_file, key_type='label', nocls=True)

                mask = self.merge_seg_prob(seg_prob, bbox, all_diff_cls_labels)

                seg_label = imread(seg_file)

                # object_index = list(set(mask[np.where(mask!=0)].flatten()))
                object_index = list(set(seg_label[np.where(seg_label!=0)].flatten()))
                for idx_c in object_index:
                    if idx_c == 255:
                        continue
                    seg_anno_c = np.single(seg_label == idx_c)
                    seg_mask_c = np.single(mask == idx_c)
                    # erase the ignore label
                    # seg_mask_c[np.where(seg_label == 255)] = 0
                    intersection_class_compare[idx_c-1] += len(np.argwhere((seg_anno_c * seg_mask_c) > 0))
                    union_class_compare[idx_c-1] += len(np.argwhere((seg_anno_c + seg_mask_c) > 0))
                if False:
                    plt.figure(1)
                    plt.imshow(im)
                    ax = plt.gca()

                    for item in detect:
                        xmin = int(round(item[0] * im_shape[1]))
                        ymin = int(round(item[1] * im_shape[0]))
                        xmax = int(round(item[2] * im_shape[1]))
                        ymax = int(round(item[3] * im_shape[0]))

                        bbox_weight = xmax - xmin
                        bbox_height = ymax - ymin

                        coords = (xmin, ymin), bbox_weight, bbox_height
                        ax.add_patch(plt.Rectangle(*coords, fill=False, edgecolor='g', linewidth=3))

                        ax.text(xmin, ymin, item[-1] + str(item[-2]), bbox={'facecolor': 'b', 'alpha': 0.5})


                    color_map = np.random.randint(0, 256, [256, 3]).astype(np.uint8)
                    mask_color = color_map[mask]
                    seg_label_color = color_map[seg_label]
                    plt.figure(666)
                    plt.imshow(mask_color)
                    plt.figure(888)
                    plt.imshow(seg_label_color)

                    plt.show()
                    pass
            else:
                object_index = list(set(seg_label[np.where(seg_label != 0)].flatten()))
                for idx_c in object_index:
                    if idx_c == 255:
                        continue
                    seg_anno_c = np.single(seg_label == idx_c)
                    intersection_class_compare[idx_c - 1] += 0
                    union_class_compare[idx_c - 1] += len(np.argwhere((seg_anno_c) > 0))
        print intersection_class_compare / union_class_compare
        print (intersection_class_compare / union_class_compare).mean()



def main(args):
    '''main '''
    detection_seg = CaffeDetectionSegmentation(args.gpu_id, args.bbox_margin, args.thresh,
                               args.model_def, args.model_weights_detection,
                               args.model_weights_segmentation,
                               args.image_resize, args.labelmap_file, args.test_file,
                               args.image_folder, args.bbox_folder, args.seg_folder)

    detection_seg.test_seg_accuracy(use_bbox_gt=False)

    detect_result = detection_seg.detect(args.image_file)
    print detect_result

    bbox, all_diff_cls_labels = detection_seg.detect2bbox(detect_result)

    seg_prob = detection_seg.seg(bbox, all_diff_cls_labels, args.image_file, key_type='label')

    mask = detection_seg.merge_seg_prob(seg_prob, bbox, all_diff_cls_labels)

    color_map = np.random.randint(0, 256, [21, 3]).astype(np.uint8)
    mask_color = color_map[mask]
    plt.figure(666)
    plt.imshow(mask_color)

    img = Image.open(args.image_file)
    width, height = img.size

    plt.figure(1)
    plt.imshow(img)
    ax = plt.gca()

    for item in detect_result:
        xmin = int(round(item[0] * width))
        ymin = int(round(item[1] * height))
        xmax = int(round(item[2] * width))
        ymax = int(round(item[3] * height))

        bbox_weight = xmax - xmin
        bbox_height = ymax - ymin

        coords = (xmin, ymin), bbox_weight, bbox_height
        ax.add_patch(plt.Rectangle(*coords, fill=False, edgecolor='g', linewidth=3))

        ax.text(xmin, ymin, item[-1] + str(item[-2]), bbox={'facecolor': 'b', 'alpha': 0.5})

    # plt.figure(2)
    # seg_prob_fg = imresize(seg_prob_fg, [height, width])
    # plt.imshow(seg_prob_fg)
    fig_id = 2
    for name, seg in seg_prob.iteritems():
        plt.figure(fig_id)
        fig_id += 1
        plt.imshow(seg)
        plt.title(name)

    plt.show()
    # draw = ImageDraw.Draw(img)
    # width, height = img.size
    # print width, height
    # for item in result:
    #     xmin = int(round(item[0] * width))
    #     ymin = int(round(item[1] * height))
    #     xmax = int(round(item[2] * width))
    #     ymax = int(round(item[3] * height))
    #     draw.rectangle([xmin, ymin, xmax, ymax], outline=(255, 0, 0))
    #     draw.text([xmin, ymin], item[-1] + str(item[-2]), (0, 0, 255))
    #     print item
    #     print [xmin, ymin, xmax, ymax]
    #     print [xmin, ymin], item[-1]
    # img.save('detect_result.jpg')


def parse_args():
    '''parse args'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--bbox_margin', type=int, default=5, help='bbox margin')
    parser.add_argument('--thresh', type=float, default=0.2, help='thresh')
    parser.add_argument('--labelmap_file',
                        default='data/VOC0712/labelmap_voc.prototxt')
    parser.add_argument('--model_def',
                        default='examples/ssd/ssd_segmentation/deploy_noclass_extra3_3.prototxt')
    parser.add_argument('--image_resize', default=320, type=int)
    parser.add_argument('--model_weights_detection',
                        default='/media/amax/data2/ssd/SSD_320x320/VGG_VOC0712_SSD_320x320_iter_120000.caffemodel')
    parser.add_argument('--model_weights_segmentation',
                        default='/media/amax/data2/ssd_seg/vgg16_noclass_extra3_3_iter_30000.caffemodel')
    parser.add_argument('--image_file', default='/home/amax/NiuChuang/data/VOCdevkit/VOC2012/JPEGImages/2009_002894.jpg')
    parser.add_argument('--test_file', default='/home/amax/NiuChuang/fcn.berkeleyvision.org/data/pascal/2012/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt')
    parser.add_argument('--image_folder', default='/home/amax/NiuChuang/data/VOCdevkit/VOC2012/JPEGImages/')
    parser.add_argument('--bbox_folder', default='/home/amax/NiuChuang/fcn.berkeleyvision.org/data/pascal/2012/VOCdevkit/VOC2012/Annotations/')
    parser.add_argument('--seg_folder', default='/home/amax/NiuChuang/DecoupleNet/DecoupledNet/data/VOC2012/VOC2012_SEG_AUG/segmentations/')
    return parser.parse_args()

if __name__ == '__main__':
    main(parse_args())
