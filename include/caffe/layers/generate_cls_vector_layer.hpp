//
// Created by NiuChuang on 17-12-14.
//

#ifndef CAFFE_GENERATE_CLS_VECTOR_LAYER_HPP
#define CAFFE_GENERATE_CLS_VECTOR_LAYER_HPP

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
    class GenerateClsVectorLayer : public Layer<Dtype> {
    public:
        explicit GenerateClsVectorLayer(const LayerParameter& param)
                : Layer<Dtype>(param) {}
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                const vector<Blob<Dtype>*>& top);
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);

        virtual inline const char* type() const { return "GenerateClsVector"; }
        virtual inline int MinBottomBlobs() const { return 1; }
        virtual inline int MinTopBlobs() const { return 1; }

    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                 const vector<Blob<Dtype>*>& top);

        // Backward not implemented
        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                                  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
            return;
        }

        int num_class_;
        int batch_size_;
        int background_label_id_;

    };

}  // namespace caffe



#endif //CAFFE_GENERATE_CLS_VECTOR_LAYER_HPP
