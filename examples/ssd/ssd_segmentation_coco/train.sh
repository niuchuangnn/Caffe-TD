cd /home/amax/NiuChuang/SSD/caffe-ssd/examples/ssd/ssd_segmentation_coco/
./../../../build/tools/caffe train \
--solver="solver.prototxt" \
--weights="/media/amax/data2/ssd_coco/models/VGGNet/coco/SSD_300x300/VGG_coco_SSD_300x300_iter_400000.caffemodel" \
--gpu 2,3 2>&1 | tee ./vgg16.log
