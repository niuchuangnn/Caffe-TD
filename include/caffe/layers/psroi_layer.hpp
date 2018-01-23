//
// Created by Niu Chuang on 18-1-18.
//

#ifndef CAFFE_PSROI_LAYER_HPP
#define CAFFE_PSROI_LAYER_HPP

#include <vector>
#include <utility>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {

    /**
 * @brief psroi assembling a set of scrore maps into the instance mask.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
    template <typename Dtype>
    class PSROILayer : public Layer<Dtype> {
    public:
        explicit PSROILayer(const LayerParameter& param)
                : Layer<Dtype>(param) {}
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                const vector<Blob<Dtype>*>& top);
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);

        virtual inline const char* type() const { return "PSROI"; }
        virtual inline int ExactNumBottomBlobs() const { return 2; }
        virtual inline int ExactNumTopBlobs() const { return 1; }

    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                 const vector<Blob<Dtype>*>& top);
        virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                 const vector<Blob<Dtype>*>& top);
        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                                  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
        virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                                  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

        int num_class_;
        int roi_size_;
    };

}


#endif //CAFFE_PSROI_LAYER_HPP
