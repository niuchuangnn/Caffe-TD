cd /home/amax/NiuChuang/SSD/caffe-ssd/examples/ssd/ssd_segmentation_coco/
./../../../build/tools/caffe train \
--solver="solver.prototxt" \
--snapshot="/media/amax/data2/ssd_coco/vgg16_seg_iter_28000.solverstate" \
--gpu 2,3 2>&1 | tee ./vgg16.log
