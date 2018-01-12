//
// Created by Niu Chuang on 17-11-25.
//

#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/select_binary_layer.hpp"

namespace caffe {

    template <typename Dtype>
    void SelectBinaryLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top) {
        SelectBinaryParameter select_binary_param = this->layer_param_.select_binary_param();
        random_select_ = select_binary_param.random_select();
        random_instance_ = select_binary_param.random_instance();

        if (random_select_){
            CHECK(select_binary_param.has_num_class())
            << "If set randomly select, the number of class must be provided";
            num_class_ = select_binary_param.num_class();
            CHECK_GT(num_class_, 1) << "The number of class must be greater than 1";
            CHECK_EQ(top.size(), 3)
                << "If set randomly select, the number of output must be 3: a binary bboxes, a binary mask, and a single class vector";
        }

        use_difficult_gt_ = select_binary_param.use_difficult_gt();
        background_label_id_ = select_binary_param.background_label_id();
        ignore_label_ = select_binary_param.ignore_label();
    }

    template <typename Dtype>
    void SelectBinaryLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top) {
        // Reshape binary mask blob.
        vector<int> mask_shape = bottom[1]->shape();
        top[1]->Reshape(mask_shape);

        // Reshape class blob.
        vector<int> class_shape;
        class_shape.push_back(mask_shape[0]);
        class_shape.push_back(num_class_);
        top[2]->Reshape(class_shape);

        // Fake reshape of bbox blob.
        top[0]->Reshape(1, 1, 1, 8);

    }

    template <typename Dtype>
    void SelectBinaryLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                           const vector<Blob<Dtype>*>& top) {
//        std::cout << "bottom[0] shape: " << bottom[0]->shape()[0] << bottom[0]->shape()[1] << bottom[0]->shape()[2] << bottom[0]->shape()[3] << std::endl;
//        std::cout << "bottom[0] shape: " << bottom[1]->shape()[0] << bottom[1]->shape()[1] << bottom[1]->shape()[2] << bottom[1]->shape()[3] << std::endl;

        // Reshape bbox blob.
        vector<int> bbox_gt_shape = bottom[0]->shape();
        int num_bbox = bbox_gt_shape[2];
        const Dtype* gt_bbox_data = bottom[0]->cpu_data();

        // Retrieve all ground truth.
        GetGroundTruth(gt_bbox_data, num_bbox, background_label_id_, use_difficult_gt_,
                       &all_gt_bboxes_);

//        std::cout << "img_num: " << all_gt_bboxes_.size() << std::endl;

        // Get all class labels presented in each image.
        GetAllClassLabels(all_gt_bboxes_, &cls_labels_);

//        std::cout << "img_num: " << cls_labels_.size() << std::endl;

        // Randomly select class label.
        RandomSelectLabel(cls_labels_, &selected_labels_);

        // Generate class specific bboxes and binary mask according to the selected class label.
        if (random_instance_) {
            // Randomly select some of the class specific bboxes.
            map<int, LabelBBox> selected_cls_bboxes;

            int num_all_selected_cls_bboxes = SelectClsBBoxes(all_gt_bboxes_, selected_labels_,
                                                              &selected_cls_bboxes);

            // Pass the selected class specific bboxes to the top blob.
            // Reshape the bbox blob.
            vector<int> bbox_data_shape;
            bbox_data_shape.push_back(1);
            bbox_data_shape.push_back(1);
            bbox_data_shape.push_back(num_all_selected_cls_bboxes);
            bbox_data_shape.push_back(8);
            top[0]->Reshape(bbox_data_shape);

            // Selected class specific bboxes.
            GenerateClassSpeicficBBox(selected_cls_bboxes, top[0]);

            // Selected binary instance mask.
            const Dtype *seg_data = bottom[1]->cpu_data();
            Dtype *binary_seg_data = top[1]->mutable_cpu_data();
            const int num = top[1]->num();
            const int channels = top[1]->channels();
            CHECK_EQ(channels, 1) << "Mask channel number must be 1.";
            CHECK_EQ(num, selected_cls_bboxes.size()) << "Image number must be the size of the selected_cls_bboxes.";
            const int height = top[1]->height();
            const int width = top[1]->width();
            GenerateBinaryMask(seg_data, selected_cls_bboxes, binary_seg_data, ignore_label_, num, height, width);

        } else {
            // Compute the number of the class specific bboxes.
            int num_cls_specific_box = NumClsBBox(all_gt_bboxes_, selected_labels_);
            vector<int> bbox_data_shape;
            bbox_data_shape.push_back(1);
            bbox_data_shape.push_back(1);
            bbox_data_shape.push_back(num_cls_specific_box);
            bbox_data_shape.push_back(8);
            top[0]->Reshape(bbox_data_shape);

            // class specific bboxes.
            GenerateClassSpeicficBBox(all_gt_bboxes_, selected_labels_, top[0]);

            // binary mask.
            const Dtype *seg_data = bottom[1]->cpu_data();
//            std::cout << bottom[1]->shape()[0] << " " << bottom[1]->shape()[1] << " " << bottom[1]->shape()[2] << " " << bottom[1]->shape()[3] << std::endl;
            Dtype *binary_seg_data = top[1]->mutable_cpu_data();
            caffe_set(top[1]->count(), Dtype(255), binary_seg_data);
            const int num = top[1]->num();
            const int channels = top[1]->channels();
            CHECK_EQ(channels, 1) << "Mask channel number must be 1.";
//            CHECK_EQ(num, selected_labels_.size()) << "Image number must be the size of the selected_cls_bboxes.";
            const int height = top[1]->height();
            const int width = top[1]->width();
            GenerateBinaryMask(seg_data, selected_labels_, binary_seg_data, ignore_label_, num, height, width);
        }

        // class label.
        Dtype* cls_data = top[2]->mutable_cpu_data();
        caffe_set(top[2]->count(), Dtype(0), cls_data);
        GenerateClass(selected_labels_, num_class_, cls_data);
    }

    INSTANTIATE_CLASS(SelectBinaryLayer);
    REGISTER_LAYER_CLASS(SelectBinary);

}  // namespace caffe
