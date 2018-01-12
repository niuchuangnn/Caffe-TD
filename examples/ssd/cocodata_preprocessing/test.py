import sys
sys.path.insert(0, '/home/amax/NiuChuang/WSSS/caffe/python')
import caffe
import numpy as np
import matplotlib.pyplot as plt
import xml.dom.minidom
import os
from scipy.misc import imread, imresize
import cv2

coco_api_path = '../cocoapi/PythonAPI'
sys.path.insert(0, coco_api_path)
from pycocotools.coco import COCO

import json

def parseXmlVoc(xmlFile):
    dom = xml.dom.minidom.parse(xmlFile)
    root = dom.documentElement
    objects = root.getElementsByTagName('object')
    # object_names = []
    objects_attr = []
    for i in range(len(objects)):
        object = objects[i]
        category_id = object.getElementsByTagName('category_id')[0].childNodes[0].data
        instance_id = object.getElementsByTagName('instance_id')[0].childNodes[0].data

        difficult = object.getElementsByTagName('difficult')[0].childNodes[0].data
        bndbox = object.getElementsByTagName('bndbox')[0]
        xmin = bndbox.getElementsByTagName('xmin')[0].childNodes[0].data
        xmax = bndbox.getElementsByTagName('xmax')[0].childNodes[0].data
        ymin = bndbox.getElementsByTagName('ymin')[0].childNodes[0].data
        ymax = bndbox.getElementsByTagName('ymax')[0].childNodes[0].data

        objects_attr.append({})
        objects_attr[i]['category_id'] = category_id
        objects_attr[i]['instance_id'] = instance_id
        objects_attr[i]['difficult'] = difficult
        objects_attr[i]['xmin'] = xmin
        objects_attr[i]['xmax'] = xmax
        objects_attr[i]['ymin'] = ymin
        objects_attr[i]['ymax'] = ymax

    return objects_attr

def showGrid(im, gridList, color='yellow'):
    fig, ax = plt.subplots(figsize=(12, 12))
    if len(im.shape) == 2:
        ax.imshow(im, aspect='equal', cmap='gray')
    else:
        ax.imshow(im, aspect='equal')
    for grid in gridList:
        ax.add_patch(
            plt.Rectangle((grid[0], grid[1]),
                          grid[2], grid[3],
                          fill=False, edgecolor=color,
                          linewidth=1)
        )
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

if __name__ == '__main__':
    dataDir = '/media/amax/data2/MSCOCO/'
    dataType = 'val2014'
    annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)

    # initialise COCO api for instance annotations
    coco = COCO(annFile)

    # display COCO categories and supercategories
    id_map_to_name = {}
    id_to_id = {}
    idid_to_name = {}
    catIds = coco.getCatIds()
    for i in range(len(catIds)):
        id_map_to_name[catIds[i]] = coco.loadCats(catIds[i])[0]['name']
        id_to_id[i+1] = catIds[i]
        idid_to_name[i+1] = coco.loadCats(catIds[i])[0]['name']

    print idid_to_name
    with open('coco_id_to_name.json', 'w') as f:
        f.write(json.dumps(idid_to_name))

    cats = coco.loadCats(catIds)
    nms = [cat['name'] for cat in cats]
    print('COCO categories: \n{}'.format(' '.join(nms)))

    nms = set([cat['supercategory'] for cat in cats])
    print('COCO supercategories: \n{}'.format(' '.join(nms)))


    coco_classes = np.asarray(
        ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
         'traffic', 'light', 'fire', 'hydrant', 'stop', 'sign', 'parking', 'meter', 'bench', 'bird',
         'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
         'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports', 'ball', 'kite', 'baseball',
         'bat', 'baseball', 'glove', 'skateboard', 'surfboard', 'tennis', 'racket', 'bottle', 'wine', 'glass', 'cup',
         'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot', 'dog',
         'pizza', 'donut', 'cake', 'chair', 'couch', 'potted', 'plant', 'bed', 'dining', 'table', 'toilet', 'tv',
         'laptop', 'mouse', 'remote', 'keyboard', 'cell', 'phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
         'book', 'clock', 'vase', 'scissors', 'teddy', 'bear', 'hair', 'drier', 'toothbrush'])

    train_list = '/media/amax/data2/MSCOCO/train2014.txt'
    data_folder = '/media/amax/data2/MSCOCO/'

    f = open(train_list, 'r')

    lines = f.readlines()

    mask_type = '.png'
    for line in lines:
        data = line.split()
        im_path = data_folder + data[0]
        mask_path = data_folder + data[1]
        xml_path = data_folder + data[2]

        im = imread(im_path)
        plt.figure(1)
        plt.imshow(im)
        ax = plt.gca()
        objects_attr = parseXmlVoc(xml_path)

        for i in range(len(objects_attr)):
            attr = objects_attr[i]
            xmin = int(float(attr['xmin']))
            ymin = int(float(attr['ymin']))
            xmax = int(float(attr['xmax']))
            ymax = int(float(attr['ymax']))
            width = xmax - xmin
            height = ymax - ymin

            coords = (xmin, ymin), width, height
            ax.add_patch(plt.Rectangle(*coords, fill=False, edgecolor='g', linewidth=3))

            category_id = int(attr['category_id'])
            # category_name = id_map_to_name[category_id]
            category_name = id_map_to_name[id_to_id[category_id]]

            ax.text(xmin, ymin, category_name, bbox={'facecolor': 'b', 'alpha': 0.5})

            instance_id = attr['instance_id']
            mask_i = imread(mask_path+'_'+str(instance_id)+mask_type)
            mask_i[np.where(mask_i==1)] = 255
            plt.figure(2+i)
            plt.imshow(mask_i)
            plt.title(category_name)

        plt.show()
        pass