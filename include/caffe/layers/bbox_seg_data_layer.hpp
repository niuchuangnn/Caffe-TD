//
// Created by NiuChuang on 17-11-20.
//

#ifndef CAFFE_BBOX_SEG_LAYER_HPP
#define CAFFE_BBOX_SEG_LAYER_HPP

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
class BBoxSegDataLayer : public BBoxSegPrefetchingLayer<Dtype> {
 public:
  explicit BBoxSegDataLayer(const LayerParameter& param);
  virtual ~BBoxSegDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // AnnotatedDataLayer uses DataReader instead for sharing for parallelism
  virtual inline bool ShareInParallel() const { return false; }
  virtual inline const char* type() const { return "BBoxSegData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 3; }

 protected:
  virtual void load_batch(BBoxSegBatch<Dtype>* bbox_seg_batch);

  DataReader<BBoxSegDatum> reader_;
  vector<BatchSampler> batch_samplers_;
  string label_map_file_;
};

}  // namespace caffe

#endif //CAFFE_BBOX_SEG_LAYER_HPP
