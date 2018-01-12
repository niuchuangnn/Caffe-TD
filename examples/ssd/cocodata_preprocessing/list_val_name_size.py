import sys
coco_api_path = '/home/amax/NiuChuang/SSD/cocoapi/PythonAPI'
sys.path.insert(0, coco_api_path)
import os
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
pylab.rcParams['figure.figsize'] = (8.0, 10.0)
from scipy.misc import imsave

from xml.dom.minidom import Document

def CreateImgXMLFileMask(im_annos, im_name, im_size, f, xml_folder, data_type, mask_folder, id_to_id, mask_type='.png', img_type='.jpg'):
    doc = Document()
    annotation = doc.createElement('annotation')
    doc.appendChild(annotation)

    folder = doc.createElement('folder')
    folder_text = doc.createTextNode(data_type)
    folder.appendChild(folder_text)
    annotation.appendChild(folder)

    filename = doc.createElement('filename')
    filename_text = doc.createTextNode(im_name+img_type)
    filename.appendChild(filename_text)
    annotation.appendChild(filename)

    size = doc.createElement('size')
    width = doc.createElement('width')
    width_value = doc.createTextNode(str(im_size[1]))
    width.appendChild(width_value)
    size.appendChild(width)

    height = doc.createElement('height')
    height_value = doc.createTextNode(str(im_size[0]))
    height.appendChild(height_value)
    size.appendChild(height)

    depth = doc.createElement('depth')
    if len(im_size) == 3:
        depth_value = doc.createTextNode(str(im_size[2]))
    elif len(im_size) == 2:
        depth_value = doc.createTextNode(str(1))
    else:
        assert False, "invalid image size."
    depth.appendChild(depth_value)
    size.appendChild(depth)
    annotation.appendChild(size)

    for i in range(len(im_annos)):
        anno = im_annos[i]

        instance = doc.createElement('object')

        im_category_id = anno['category_id']
        im_instance_id = i + 1

        category_id = doc.createElement('category_id')
        category_id_value = doc.createTextNode(str(id_to_id[im_category_id]))
        category_id.appendChild(category_id_value)
        instance.appendChild(category_id)

        instance_id = doc.createElement('instance_id')
        instance_id_value = doc.createTextNode(str(im_instance_id))
        instance_id.appendChild(instance_id_value)
        instance.appendChild(instance_id)

        iscrowd = anno['iscrowd']
        difficult = doc.createElement('difficult')
        difficult_value = doc.createTextNode(str(iscrowd))
        difficult.appendChild(difficult_value)
        instance.appendChild(difficult)

        bndbox = doc.createElement('bndbox')

        bbox = anno['bbox']
        im_xmin = int(np.floor(bbox[0]))
        im_ymin = int(np.floor(bbox[1]))
        im_width = np.ceil(bbox[2])
        im_height = np.ceil(bbox[3])
        im_xmax = int(im_xmin + im_width)
        im_ymax = int(im_ymin + im_height)

        xmin = doc.createElement('xmin')
        xmin_value = doc.createTextNode(str(im_xmin))
        xmin.appendChild(xmin_value)
        bndbox.appendChild(xmin)

        ymin = doc.createElement('ymin')
        ymin_value = doc.createTextNode(str(im_ymin))
        ymin.appendChild(ymin_value)
        bndbox.appendChild(ymin)

        xmax = doc.createElement('xmax')
        xmax_value = doc.createTextNode(str(im_xmax))
        xmax.appendChild(xmax_value)
        bndbox.appendChild(xmax)

        ymax = doc.createElement('ymax')
        ymax_value = doc.createTextNode(str(im_ymax))
        ymax.appendChild(ymax_value)
        bndbox.appendChild(ymax)

        instance.appendChild(bndbox)

        annotation.appendChild(instance)

        binary_mask = coco.annToMask(anno).astype(np.uint8)

        imsave(mask_folder + im_name + '_' + str(im_instance_id) + mask_type, binary_mask)

    with open(xml_folder + im_name + '.xml', 'w') as f_xml:
        f_xml.write(doc.toprettyxml(indent='\t', encoding='utf-8'))

    f.write(dataType+'/'+im_name+img_type + ' ' + 'instance/'+dataType+'/'+im_name + ' ' + 'xml/'+dataType+'/'+im_name+'.xml\n')

    print im_name

if __name__ == '__main__':

    dataDir = '/media/amax/data2/MSCOCO/MS COCO 2017/'
    dataType = 'val2017'
    annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)

    save_file = '/home/amax/NiuChuang/SSD/caffe-ssd/data/coco/val2017_name_size.txt'
    f = open(save_file, 'w')
    # initialise COCO api for instance annotations
    coco = COCO(annFile)

    # map the real category id of COCO to the class id in the training.
    catIds = coco.getCatIds()

    cats = coco.loadCats(catIds)

    imgIds = coco.getImgIds()

    for i in range(len(imgIds)):
        imgs = coco.loadImgs(imgIds[i])
        im_name = imgs[0]['file_name']
        im_name = im_name[0:len(im_name)-4]

        # load and display image
        I = io.imread('{}{}/{}'.format(dataDir, dataType, imgs[0]['file_name']))
        height = I.shape[0]
        width = I.shape[1]
        f.write(im_name + ' ' + str(height) + ' ' + str(width) + '\n')
        pass