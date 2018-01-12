//
// Created by Niu Chuang on 17-12-14.
//

#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/generate_cls_vector_layer.hpp"

namespace caffe {

    template <typename Dtype>
    void GenerateClsVectorLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                              const vector<Blob<Dtype>*>& top) {
        GenerateClsVectorParameter generate_cls_vector_param = this->layer_param_.generate_cls_vector_param();
        num_class_ = generate_cls_vector_param.num_class();
        batch_size_ = generate_cls_vector_param.batch_size();
        background_label_id_ = generate_cls_vector_param.background_label_id();
    }

    template <typename Dtype>
    void GenerateClsVectorLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                           const vector<Blob<Dtype>*>& top) {
        vector<int> top_shape;
        top_shape.push_back(batch_size_);
        top_shape.push_back(num_class_);
        top[0]->Reshape(top_shape);
    }

    template <typename Dtype>
    void GenerateClsVectorLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                               const vector<Blob<Dtype>*>& top) {
        // Reshape bbox blob.
        vector<int> bbox_gt_shape = bottom[0]->shape();
        int num_bbox = bbox_gt_shape[2];
        const Dtype* gt_bbox_data = bottom[0]->cpu_data();

        // Retrieve all ground truth.
        map<int, vector<NormalizedBBox> > all_gt_bboxes;
        GetGroundTruth(gt_bbox_data, num_bbox, background_label_id_, true,
                       &all_gt_bboxes);

        // Get all class labels presented in each image.
        map<int, vector<int> > cls_labels;
        GetAllClassLabels(all_gt_bboxes, &cls_labels);

        // Check if each image contain a single class
        map<int, vector<int> >::const_iterator it;
        for (it = cls_labels.begin(); it != cls_labels.end(); ++it){
            CHECK_EQ(it->second.size(), 1) << "Current version only support generate single class vector.";
        }

        // Randomly select class label.
        map<int, int> selected_labels;
        RandomSelectLabel(cls_labels, &selected_labels);

        // class label.
        Dtype* cls_data = top[0]->mutable_cpu_data();
        caffe_set(top[0]->count(), Dtype(0), cls_data);
        GenerateClass(selected_labels, num_class_, cls_data);
    }

    INSTANTIATE_CLASS(GenerateClsVectorLayer);
    REGISTER_LAYER_CLASS(GenerateClsVector);

}  // namespace caffe
