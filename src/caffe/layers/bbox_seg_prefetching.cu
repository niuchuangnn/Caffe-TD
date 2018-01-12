//
// Created by Niu Chuang on 17-11-22.
//

#include <vector>

#include "caffe/layers/bbox_seg_prefetching.hpp"

namespace caffe {

template <typename Dtype>
void BBoxSegPrefetchingLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  BBoxSegBatch<Dtype>* batch = prefetch_full_.pop("Data layer prefetch queue empty");
  // Reshape to loaded data.
  top[0]->ReshapeLike(batch->data_);
  // Copy the data
  caffe_copy(batch->data_.count(), batch->data_.gpu_data(),
      top[0]->mutable_gpu_data());

  // Reshape to loaded bbox.
  top[1]->ReshapeLike(batch->bbox_);
  // Copy the bbox.
  caffe_copy(batch->bbox_.count(), batch->bbox_.gpu_data(),
      top[1]->mutable_gpu_data());

  // Reshape to loaded seg.
  top[2]->ReshapeLike(batch->seg_);
  // Copy the seg.
  caffe_copy(batch->seg_.count(), batch->seg_.gpu_data(),
             top[2]->mutable_gpu_data());

  // Ensure the copy is synchronous wrt the host, so that the next batch isn't
  // copied in meanwhile.
  CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
  prefetch_free_.push(batch);
}

INSTANTIATE_LAYER_GPU_FORWARD(BBoxSegPrefetchingLayer);

}  // namespace caffe
