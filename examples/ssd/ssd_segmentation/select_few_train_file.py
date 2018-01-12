import os
import sys
import argparse
import numpy as np

import matplotlib.pyplot as plt
from scipy.misc import imresize, imread
import xml.dom.minidom
import random

voc_classes = np.asarray(
        ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
         'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'])

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

if __name__ == "__main__":
    train_all_file = '/home/amax/NiuChuang/data/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt'
    save_file = '/home/amax/NiuChuang/data/VOCdevkit/VOC2012/ImageSets/Segmentation/train_5_per_cls.txt'
    xml_folder = '/home/amax/NiuChuang/data/VOCdevkit/VOC2012/Annotations/'
    image_folder = '/home/amax/NiuChuang/data/VOCdevkit/VOC2012/JPEGImages/'
    num_per_class = 5
    fw = open(save_file, 'w')
    f = open(train_all_file, 'r')
    lines = f.readlines()

    cls_name_to_file_name = {}
    for i in range(len(voc_classes)):
        cls_name_to_file_name[voc_classes[i]] = []

    for line in lines:
        name = line.split('\n')[0]
        label_file = xml_folder + name + '.xml'
        object_names, objects_attr = parseXmlVoc(label_file)

        object_names = set(object_names)
        for object_name in object_names:
            cls_name_to_file_name[object_name].append(name)

    for key, value in cls_name_to_file_name.iteritems():
        print key, len(cls_name_to_file_name[key])

    select_cls_name_to_file_name = {}
    for i in range(len(voc_classes)):
        select_cls_name_to_file_name[voc_classes[i]] = []

    for i in range(len(voc_classes)):
        name = voc_classes[i]
        files = cls_name_to_file_name[name]
        random.shuffle(files)
        if i == 0:
            select_cls_name_to_file_name[name] = files[0:num_per_class]
        else:
            exist_files = []
            for _, value in select_cls_name_to_file_name.iteritems():
                exist_files = exist_files + value
            for fn in files:
                if len(select_cls_name_to_file_name[name]) == num_per_class:
                    break
                elif fn not in exist_files:
                    select_cls_name_to_file_name[name].append(fn)

    all_selected_file = []
    for key, value in select_cls_name_to_file_name.iteritems():
        print key, len(select_cls_name_to_file_name[key])
        all_selected_file = all_selected_file + value

    # for key, value in select_cls_name_to_file_name.iteritems():
    #     for i in range(len(value)):
    #         name = value[i]
    #         im = imread(image_folder + name + '.jpg')
    #         plt.figure(1)
    #         plt.imshow(im)
    #         plt.title(key + '_' + str(i+1))
    #         plt.show()
    for name in all_selected_file:
        fw.write(name+'\n')

    pass