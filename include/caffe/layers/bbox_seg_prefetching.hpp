//
// Created by Niu Chuang on 17-11-20.
//

#ifndef CAFFE_BBOX_SEG_PREFETCHING_HPP
#define CAFFE_BBOX_SEG_PREFETCHING_HPP


#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/blocking_queue.hpp"
#include "caffe/layers/base_data_layer.hpp"

namespace caffe {

    template <typename Dtype>
    class BBoxSegBatch {
    public:
        Blob<Dtype> data_, bbox_, seg_;
    };

    template <typename Dtype>
    class BBoxSegPrefetchingLayer :
            public BaseDataLayer<Dtype>, public InternalThread {
    public:
        explicit BBoxSegPrefetchingLayer(const LayerParameter& param);
        // LayerSetUp: implements common data layer setup functionality, and calls
        // DataLayerSetUp to do special data layer setup for individual layer types.
        // This method may not be overridden.
        void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                        const vector<Blob<Dtype>*>& top);

        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                 const vector<Blob<Dtype>*>& top);
        virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                 const vector<Blob<Dtype>*>& top);

        // Prefetches batches (asynchronously if to GPU memory)
        static const int PREFETCH_COUNT = 3;

    protected:
        virtual void InternalThreadEntry();
        virtual void load_batch(BBoxSegBatch<Dtype>* batch) = 0;

        BBoxSegBatch<Dtype> prefetch_[PREFETCH_COUNT];
        BlockingQueue<BBoxSegBatch<Dtype>*> prefetch_free_;
        BlockingQueue<BBoxSegBatch<Dtype>*> prefetch_full_;

        Blob<Dtype> transformed_data_;
        Blob<Dtype> transformed_seg_;
    };

}  // namespace caffe

#endif //CAFFE_BBOX_SEG_PREFETCHING_HPP
