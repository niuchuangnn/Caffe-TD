//
// Created by Niu Chuang on 17-11-25.
//

#ifndef CAFFE_SELECT_BINARY_LAYER_HPP
#define CAFFE_SELECT_BINARY_LAYER_HPP

#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_reader.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/layers/bbox_seg_prefetching.hpp"
#include "caffe/util/bbox_util.hpp"

namespace caffe {

    template <typename Dtype>
    class SelectBinaryLayer : public Layer<Dtype> {
    public:
        explicit SelectBinaryLayer(const LayerParameter& param)
                : Layer<Dtype>(param) {}
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                const vector<Blob<Dtype>*>& top);
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);

        virtual inline const char* type() const { return "SelectBinary"; }
        virtual inline int MinBottomBlobs() const { return 2; }
        virtual inline int MinTopBlobs() const { return 2; }

    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                 const vector<Blob<Dtype>*>& top);

        // Backward not implemented
        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                                  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
            return;
        }

        bool random_select_;
        int num_class_;
        int background_label_id_;
        int ignore_label_;
        bool use_difficult_gt_;
        bool random_instance_;
        map<int, vector<NormalizedBBox> > all_gt_bboxes_;
        map<int, vector<int>> cls_labels_;
        map<int, int> selected_labels_;

    };

}  // namespace caffe



#endif //CAFFE_SELECT_BINARY_LAYER_HPP
