import sys
coco_api_path = '/home/amax/NiuChuang/SSD/cocoapi/PythonAPI'
sys.path.insert(0, coco_api_path)

from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
pylab.rcParams['figure.figsize'] = (8.0, 10.0)

dataDir = '/media/amax/data2/MSCOCO/MS_COCO_2017/'
dataType = 'train2017'
annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)

# initialise COCO api for instance annotations
coco = COCO(annFile)

# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
nms = [cat['name'] for cat in cats]
print('COCO categories: \n{}'.format(' '.join(nms)))

nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms)))

# get all images containing given categories, select one at random
catIds = coco.getCatIds(catNms=['person', 'dog', 'skateboard'])
imgIds = coco.getImgIds(catIds=catIds)
# imgIds = coco.getImgIds(imgIds=[452746]) #324158
img = coco.loadImgs(imgIds[np.random.randint(0, len(imgIds))])[0]

# load and display image
I = io.imread('{}{}/{}'.format(dataDir, dataType, img['file_name']))
plt.axis('off')
plt.imshow(I)
plt.show()

# load and display instance annotations
plt.figure(1)
plt.imshow(I)
plt.axis('off')
annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
anns = coco.loadAnns(annIds)
coco.showAnns(anns)
print len(anns)
for i in range(len(anns)):
    plt.figure(2+i)
    mask = coco.annToMask(anns[i])
    plt.imshow(mask)
    bbox = anns[i]['bbox']
    xmin = np.floor(bbox[0])
    ymin = np.floor(bbox[1])
    width = np.ceil(bbox[2])
    height = np.ceil(bbox[3])

    ax = plt.gca()
    coords = (xmin, ymin), width, height
    ax.add_patch(plt.Rectangle(*coords, fill=False, edgecolor='g', linewidth=3))

    ax.text(xmin, ymin, coco.loadCats(anns[i]['category_id']), bbox={'facecolor': 'b', 'alpha': 0.5})
plt.show()

# load and display keypoints annotations
annFile = '{}/annotations/person_keypoints_{}.json'.format(dataDir, dataType)
coco_kps = COCO(annFile)
plt.imshow(I)
plt.axis('off')
ax = plt.gca()
annIds = coco_kps.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
anns = coco_kps.loadAnns(annIds)
coco_kps.showAnns(anns)
plt.show()

pass