cd /home/amax/NiuChuang/SSD/caffe-ssd/examples/ssd/refineNet/
./../../../build/tools/caffe train \
--solver="solver.prototxt" \
--weights="/media/amax/data2/RefineDet/models/VGGNet/VOC0712/refinedet_vgg16_320x320/VOC0712_refinedet_vgg16_320x320_final.caffemodel" \
--gpu 1 2>&1 | tee ./vgg16.log
