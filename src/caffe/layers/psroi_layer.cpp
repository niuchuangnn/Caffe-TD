//
// Created by Niu Chuang on 18-1-18.
//

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/psroi_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

    using std::min;
    using std::max;

    template <typename Dtype>
    void PSROILayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                         const vector<Blob<Dtype>*>& top) {
        PSROIParameter psroi_param = this->layer_param_.psroi_param();
        roi_size_ = psroi_param.roi_size();
        num_class_ = psroi_param.num_class();

    }

    template <typename Dtype>
    void PSROILayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                      const vector<Blob<Dtype>*>& top) {
        vector<int> data_shape = bottom[0]->shape();
        int channels = bottom[0]->channels();
        int num_bbox = bottom[1]->height();
        CHECK_EQ(channels, 2*num_class_*roi_size_*roi_size_) << "Data channels mismatch roi size.";
        data_shape[0] = num_bbox;
        data_shape[1] = 2;

//        std::cout << "data_shape: " << data_shape[0] << " " << data_shape[1] << " " << data_shape[2] << " " << data_shape[3] << std::endl;

        top[0]->Reshape(data_shape);

    }

// TODO(Yangqing): Is there a faster way to do pooling in the channel-first
// case?
    template <typename Dtype>
    void PSROILayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top) {
        NOT_IMPLEMENTED; // Only implemented in gpu version.

    }

    template <typename Dtype>
    void PSROILayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                           const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
        NOT_IMPLEMENTED; // Only implemented in gpu version.
    }


#ifdef CPU_ONLY
    STUB_GPU(PSROILayer);
#endif

    INSTANTIATE_CLASS(PSROILayer);
    REGISTER_LAYER_CLASS(PSROI);

}  // namespace caffe
