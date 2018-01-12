cd /home/amax/NiuChuang/SSD/caffe-ssd/examples/ssd/ssd_segmentation/
./../../../build/tools/caffe train \
--solver="solver_finetune.prototxt" \
--weights="/media/amax/data2/ssd_seg/vgg16_detect_seg_full_voc_pretrain.caffemodel" \
--gpu 0,1 2>&1 | tee ./vgg16.log
