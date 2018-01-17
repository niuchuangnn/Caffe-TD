cd /home/amax/NiuChuang/SSD/caffe-ssd/examples/ssd/ssd_segmentation/
./../../../build/tools/caffe train \
--solver="solver_noclass.prototxt" \
--weights="/media/amax/data2/ssd/SSD_320x320/VGG_VOC0712_SSD_320x320_iter_120000.caffemodel" \
--gpu 0 2>&1 | tee ./vgg16.log
