cd /home/amax/NiuChuang/SSD/caffe-ssd/examples/ssd/ssd_segmentation/
./../../../build/tools/caffe train \
--solver="solver_noclass.prototxt" \
--snapshot="/media/amax/data2/ssd_seg/vgg16_noclass_crop4_3_iter_28000.solverstate" \
--gpu 0 2>&1 | tee ./vgg16.log
