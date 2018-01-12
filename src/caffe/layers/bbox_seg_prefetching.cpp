//
// Created by Niu Chuang on 17-11-20.
//

#include <boost/thread.hpp>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/bbox_seg_prefetching.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/blocking_queue.hpp"

namespace caffe {

    template <typename Dtype>
    BBoxSegPrefetchingLayer<Dtype>::BBoxSegPrefetchingLayer(
            const LayerParameter& param)
            : BaseDataLayer<Dtype>(param),
              prefetch_free_(), prefetch_full_() {
        for (int i = 0; i < PREFETCH_COUNT; ++i) {
            prefetch_free_.push(&prefetch_[i]);
        }
    }

    template <typename Dtype>
    void BBoxSegPrefetchingLayer<Dtype>::LayerSetUp(
            const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
        BaseDataLayer<Dtype>::LayerSetUp(bottom, top);
        // Before starting the prefetch thread, we make cpu_data and gpu_data
        // calls so that the prefetch thread does not accidentally make simultaneous
        // cudaMalloc calls when the main thread is running. In some GPUs this
        // seems to cause failures if we do not so.
        for (int i = 0; i < PREFETCH_COUNT; ++i) {
            prefetch_[i].data_.mutable_cpu_data();

            prefetch_[i].bbox_.mutable_cpu_data();
            prefetch_[i].seg_.mutable_cpu_data();
        }
#ifndef CPU_ONLY
        if (Caffe::mode() == Caffe::GPU) {
            for (int i = 0; i < PREFETCH_COUNT; ++i) {
                prefetch_[i].data_.mutable_gpu_data();

                prefetch_[i].bbox_.mutable_gpu_data();
                prefetch_[i].seg_.mutable_cpu_data();

            }
        }
#endif
        DLOG(INFO) << "Initializing prefetch";
        this->data_transformer_->InitRand();
        StartInternalThread();
        DLOG(INFO) << "Prefetch initialized.";
    }

    template <typename Dtype>
    void BBoxSegPrefetchingLayer<Dtype>::InternalThreadEntry() {
#ifndef CPU_ONLY
        cudaStream_t stream;
        if (Caffe::mode() == Caffe::GPU) {
            CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        }
#endif

        try {
            while (!must_stop()) {
                BBoxSegBatch<Dtype>* batch = prefetch_free_.pop();
                load_batch(batch);
#ifndef CPU_ONLY
                if (Caffe::mode() == Caffe::GPU) {
                    batch->data_.data().get()->async_gpu_push(stream);
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                }
#endif
                prefetch_full_.push(batch);
            }
        } catch (boost::thread_interrupted&) {
            // Interrupted exception is expected on shutdown
        }
#ifndef CPU_ONLY
        if (Caffe::mode() == Caffe::GPU) {
            CUDA_CHECK(cudaStreamDestroy(stream));
        }
#endif
    }

    template <typename Dtype>
    void BBoxSegPrefetchingLayer<Dtype>::Forward_cpu(
            const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
        BBoxSegBatch<Dtype>* batch = prefetch_full_.pop("Data layer prefetch queue empty");
        // Reshape to loaded data.
        top[0]->ReshapeLike(batch->data_);
        // Copy the data
        caffe_copy(batch->data_.count(), batch->data_.cpu_data(),
                   top[0]->mutable_cpu_data());
        DLOG(INFO) << "Prefetch bbox copied";

        // Reshape to loaded bbox.
        top[1]->ReshapeLike(batch->bbox_);
        // Copy the bbox.
        caffe_copy(batch->bbox_.count(), batch->bbox_.cpu_data(),
                   top[1]->mutable_cpu_data());

        // Reshape to loaded seg.
        top[2]->ReshapeLike(batch->seg_);
        // Copy the seg.
        caffe_copy(batch->seg_.count(), batch->seg_.cpu_data(),
                   top[2]->mutable_cpu_data());
        DLOG(INFO) << "Prefetch seg copied";

        prefetch_free_.push(batch);
    }

#ifdef CPU_ONLY
    STUB_GPU_FORWARD(BasePrefetchingDataLayer, Forward);
#endif

    INSTANTIATE_CLASS(BBoxSegPrefetchingLayer);

}  // namespace caffe
