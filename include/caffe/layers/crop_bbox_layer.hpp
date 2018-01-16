//
// Created by Niu Chuang on 17-11-27.
//

#ifndef CAFFE_CROP_BBOX_LAYER_HPP
#define CAFFE_CROP_BBOX_LAYER_HPP

#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/util/bbox_util.hpp"

namespace caffe {

/**
 * @brief Keeps the blob values within a given set of bounding boxes and set all other values zero.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */

    template <typename Dtype>
    class CropBBoxLayer : public Layer<Dtype> {
    public:
        explicit CropBBoxLayer(const LayerParameter& param)
                : Layer<Dtype>(param) {}
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                const vector<Blob<Dtype>*>& top);
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);

        virtual inline const char* type() const { return "CropBBox"; }
        virtual inline int MinBottomBlobs() const { return 2; }
        virtual inline int ExactNumTopBlobs() const { return 1; }

    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                 const vector<Blob<Dtype>*>& top);
        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                                  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
        virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                 const vector<Blob<Dtype>*>& top);
        virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                                  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

        void copy_bbox(const Dtype* bottom_data, const int channels, const int width, const int height,
               const int xmin, const int ymin, const int xmax, const int ymax, Dtype* top_data);

        map<int, LabelBBox> all_gt_boxes_;
        int num_gt_;
        int background_label_id_;
        bool use_difficult_gt_;
        int num_class_;
        int num_img_;
        Blob<Dtype> mask_crop_;
    };
}  // namespace caffe


#endif //CAFFE_CROP_BBOX_LAYER_HPP
