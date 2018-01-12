//
// Created by Niu Chuang on 17-11-27.
//

#include <algorithm>
#include <functional>
#include <map>
#include <set>
#include <vector>


#include "caffe/layer.hpp"
#include "caffe/layers/crop_bbox_layer.hpp"
#include "caffe/net.hpp"


namespace caffe {

    template <typename Dtype>
    void CropBBoxLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                      const vector<Blob<Dtype>*>& top) {
        CropBBoxParameter crop_bbox_param = this->layer_param_.crop_bbox_param();
        background_label_id_ = crop_bbox_param.background_label_id();
        use_difficult_gt_ = crop_bbox_param.use_difficult_gt();
    }

    template <typename Dtype>
    void CropBBoxLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                   const vector<Blob<Dtype>*>& top) {
        vector<int> bottom_shape = bottom[0]->shape();
        top[0]->Reshape(bottom_shape);
        mask_crop_.Reshape(bottom_shape);
    }

    template <typename Dtype>
    void CropBBoxLayer<Dtype>::copy_bbox(const Dtype* bottom_data, const int channels, const int width, const int height,
                   const int xmin, const int ymin, const int xmax, const int ymax, Dtype* top_data){
        for (int c = 0; c < channels; c++){
            for (int h = 0; h < height; h++){
                for (int w = 0; w < width; w++){
                    if (InBBox(w, h, xmin, ymin, xmax, ymax)){
                        top_data[c*height*width + h*width + w] = bottom_data[c*height*width + h*width + w];
                    }
                }
            }
        }
    }


    template <typename Dtype>
    void CropBBoxLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top) {
        vector<int> top_shape = bottom[0]->shape();
        const Dtype* gt_data = bottom[1]->cpu_data();
        num_gt_ = bottom[1]->height();

        // Retrieve all ground truth.
        GetGroundTruth(gt_data, num_gt_, background_label_id_, use_difficult_gt_,
                       &all_gt_boxes_);

        num_img_ = bottom[0]->num();
        CHECK_LE(all_gt_boxes_.size(), num_img_) << "Number of image with bbox must be less or equal to the number of image.";

        num_class_ = 0;
        for (int i = 0; i < num_img_; ++i) {
            num_class_ += all_gt_boxes_[i].size();
        }

//        CHECK_GT(num_output_, 0) << "num_output_ should be larger than 0.";
        CHECK_LE(num_class_, num_img_) << "Current version only support one class at most selected for each image.";

//        top_shape[0] = num_output_;
//        top[0]->Reshape(top_shape);

        const Dtype* bottom_data = bottom[0]->cpu_data();
        Dtype* top_data = top[0]->mutable_cpu_data();
        Dtype* mask_data = mask_crop_.mutable_cpu_data();

        float xmin_norm, ymin_norm, xmax_norm, ymax_norm;
        int label, channels, width, height, xmin, ymin, xmax, ymax, img_id;
        vector<int> label_indices;
        int inner_dim = bottom[0]->offset(1);
        channels = bottom[0]->channels();
        width = bottom[0]->width();
        height = bottom[0]->height();

//        for (int i = 0; i < num_img_; ++i){
//            LabelBBox::iterator it;
//            label_indices.clear();
//            for (it = all_gt_boxes_[i].begin(); it != all_gt_boxes_[i].end(); it++){
//                label_indices.push_back(it->first);
//            }
//            int num_class_i = label_indices.size();
//            for (int l = 0; l < num_class_i; ++l){
//                label = label_indices[l];
//                vector<NormalizedBBox> bboxes = all_gt_boxes_[i][label];
//                for (int b = 0; b < bboxes.size(); ++b){
//                    xmin_norm = bboxes[b].xmin();
//                    ymin_norm = bboxes[b].ymin();
//                    xmax_norm = bboxes[b].xmax();
//                    ymax_norm = bboxes[b].ymax();
//
//                    xmin = static_cast<int>(floor(xmin_norm * static_cast<Dtype>(width)));
//                    ymin = static_cast<int>(floor(ymin_norm * static_cast<Dtype>(height)));
//                    xmax = static_cast<int>(ceil(xmax_norm * static_cast<Dtype>(width)));
//                    ymax = static_cast<int>(ceil(ymax_norm * static_cast<Dtype>(height)));
//
//                    copy_bbox(bottom_data + i*inner_dim, channels, width, height,
//                              xmin, ymin, xmax, ymax, top_data + (i*num_class_i+l)*inner_dim);
//                }
//            }
//        }

        // compute corp mask
        caffe_set(mask_crop_.count(), Dtype(0), mask_crop_.mutable_cpu_data());
        map<int, LabelBBox>::const_iterator iter_im;
        for (iter_im = all_gt_boxes_.begin(); iter_im != all_gt_boxes_.end(); ++iter_im){
            img_id = iter_im->first;
            CHECK(img_id >=0 && img_id < num_img_) << "img_id must be less than the number of images.";
            LabelBBox::iterator it;
            label_indices.clear();
            for (it = all_gt_boxes_[img_id].begin(); it != all_gt_boxes_[img_id].end(); it++){
                label_indices.push_back(it->first);
            }
            int num_class_i = label_indices.size();
            for (int l = 0; l < num_class_i; ++l){
                label = label_indices[l];
                vector<NormalizedBBox> bboxes = all_gt_boxes_[img_id][label];
                for (int b = 0; b < bboxes.size(); ++b){
                    xmin_norm = bboxes[b].xmin();
                    ymin_norm = bboxes[b].ymin();
                    xmax_norm = bboxes[b].xmax();
                    ymax_norm = bboxes[b].ymax();

                    xmin = static_cast<int>(floor(xmin_norm * static_cast<Dtype>(width)));
                    ymin = static_cast<int>(floor(ymin_norm * static_cast<Dtype>(height)));
                    xmax = static_cast<int>(ceil(xmax_norm * static_cast<Dtype>(width)));
                    ymax = static_cast<int>(ceil(ymax_norm * static_cast<Dtype>(height)));

                    xmin = std::max(0, xmin);
                    ymin = std::max(0, ymin);
                    xmax = std::min(xmax, width);
                    ymax = std::min(ymax, height);

                    for (int c = 0; c < channels; ++c) {
                        for (int h = ymin; h < ymax; ++h) {
                            for (int w = xmin; w < xmax; ++w) {

                                mask_data[img_id*inner_dim + c*height*width + h*width + w] = 1;
                            }
                        }
                    }

                }
            }
        }

        // crop the bottom data to top blob with crop mask.
        caffe_mul(top[0]->count(), bottom_data, mask_crop_.cpu_data(), top_data);

    }

    template <typename Dtype>
    void CropBBoxLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
        if (propagate_down[0]) {
            const Dtype* top_diff = top[0]->cpu_diff();
            const Dtype* mask_crop = mask_crop_.cpu_data();
            Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

            caffe_mul<Dtype>(top[0]->count(), top_diff, mask_crop, bottom_diff);
        }
    }

#ifdef CPU_ONLY
    STUB_GPU(CropBBoxLayer);
#endif

    INSTANTIATE_CLASS(CropBBoxLayer);
    REGISTER_LAYER_CLASS(CropBBox);

}  // namespace caffe
