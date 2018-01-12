//
// Created by Niu Chuang on 17-11-20.
//

#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <algorithm>
#include <map>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/bbox_seg_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/sampler.hpp"

namespace caffe {

    template <typename Dtype>
    BBoxSegDataLayer<Dtype>::BBoxSegDataLayer(const LayerParameter& param)
            : BBoxSegPrefetchingLayer<Dtype>(param),
              reader_(param) {
    }

    template <typename Dtype>
    BBoxSegDataLayer<Dtype>::~BBoxSegDataLayer() {
        this->StopInternalThread();
    }

    template <typename Dtype>
    void BBoxSegDataLayer<Dtype>::DataLayerSetUp(
            const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

        const int batch_size = this->layer_param_.data_param().batch_size();
        const BBoxSegDataParameter& bbox_seg_data_param =
                this->layer_param_.bbox_seg_data_param();
        for (int i = 0; i < bbox_seg_data_param.batch_sampler_size(); ++i) {
            batch_samplers_.push_back(bbox_seg_data_param.batch_sampler(i));
        }
        label_map_file_ = bbox_seg_data_param.label_map_file();
        // Make sure dimension is consistent within batch.
        const TransformationParameter& transform_param =
                this->layer_param_.transform_param();
        if (transform_param.has_resize_param()) {
            if (transform_param.resize_param().resize_mode() ==
                ResizeParameter_Resize_mode_FIT_SMALL_SIZE) {
                CHECK_EQ(batch_size, 1)
                    << "Only support batch size of 1 for FIT_SMALL_SIZE.";
            }
        }

        // Read a data point, and use it to initialize the top blob.
        BBoxSegDatum& bbox_seg_datum = *(reader_.full().peek());

        // Use data_transformer to infer the expected blob shape from bbox_seg_datum.
        vector<int> top_shape =
                this->data_transformer_->InferBlobShape(bbox_seg_datum.seg_datum());
        this->transformed_data_.Reshape(top_shape);
        // Reshape top[0] and prefetch_data according to the batch_size.
        top_shape[0] = batch_size;
        top[0]->Reshape(top_shape);
        for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
            this->prefetch_[i].data_.Reshape(top_shape);
        }
        LOG(INFO) << "output data size: " << top[0]->num() << ","
                  << top[0]->channels() << "," << top[0]->height() << ","
                  << top[0]->width();
        // bbox
        vector<int> bbox_shape(4, 1);

        // Infer the label shape from anno_datum.AnnotationGroup().
        int num_bboxes = 0;
        // Since the number of bboxes can be different for each image,
        // we store the bbox information in a specific format. In specific:
        // All bboxes are stored in one spatial plane (num and channels are 1)
        // And each row contains one and only one box in the following format:
        // [item_id, group_label, instance_id, xmin, ymin, xmax, ymax, diff]
        // Note: Refer to caffe.proto for details about group_label and
        // instance_id.
        for (int g = 0; g < bbox_seg_datum.annotation_group_size(); ++g) {
            num_bboxes += bbox_seg_datum.annotation_group(g).annotation_size();
        }
        bbox_shape[0] = 1;
        bbox_shape[1] = 1;
        // BasePrefetchingDataLayer<Dtype>::LayerSetUp() requires to call
        // cpu_data and gpu_data for consistent prefetch thread. Thus we make
        // sure there is at least one bbox.
        bbox_shape[2] = std::max(num_bboxes, 1);
        bbox_shape[3] = 8;
        top[1]->Reshape(bbox_shape);
        for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
            this->prefetch_[i].bbox_.Reshape(bbox_shape);
        }

        // seg
        vector<int> seg_shape(top_shape);
        seg_shape[1] = 1;
        top[2]->Reshape(seg_shape);
        for (int i =0; i < this->PREFETCH_COUNT; ++i) {
            this->prefetch_[i].seg_.Reshape(seg_shape);
        }

        seg_shape[0] = 1;
        this->transformed_seg_.Reshape(seg_shape);

    }

// This function is called on prefetch thread
    template<typename Dtype>
    void BBoxSegDataLayer<Dtype>::load_batch(BBoxSegBatch<Dtype>* bbox_seg_batch) {
        CPUTimer batch_timer;
        batch_timer.Start();
        double read_time = 0;
        double trans_time = 0;
        CPUTimer timer;
        CHECK(bbox_seg_batch->data_.count());
        CHECK(this->transformed_data_.count());

        // Reshape according to the first anno_datum of each batch
        // on single input batches allows for inputs of varying dimension.
        const int batch_size = this->layer_param_.data_param().batch_size();
        const BBoxSegDataParameter& bbox_seg_data_param =
                this->layer_param_.bbox_seg_data_param();
        const TransformationParameter& transform_param =
                this->layer_param_.transform_param();
        bool is_object_mask = bbox_seg_data_param.is_object_mask();

        BBoxSegDatum& bbox_seg_datum = *(reader_.full().peek());

        // Use data_transformer to infer the expected blob shape from anno_datum.
        vector<int> top_shape =
                this->data_transformer_->InferBlobShape(bbox_seg_datum.seg_datum());
        this->transformed_data_.Reshape(top_shape);
        // Reshape batch according to the batch_size.
        top_shape[0] = batch_size;
        bbox_seg_batch->data_.Reshape(top_shape);

//        std::cout << "channels: " << top_shape[1] << std::endl;

        Dtype* top_data = bbox_seg_batch->data_.mutable_cpu_data();
        Dtype* top_bbox = bbox_seg_batch->bbox_.mutable_cpu_data();
        Dtype* top_seg = bbox_seg_batch->seg_.mutable_cpu_data();

        caffe_set(bbox_seg_batch->seg_.count(), Dtype(255), top_seg);

        // Store transformed annotation.
        map<int, vector<AnnotationGroup> > all_anno;
        int num_bboxes = 0;

        for (int item_id = 0; item_id < batch_size; ++item_id) {
            timer.Start();
            // get a anno_datum
            BBoxSegDatum& bbox_seg_datum_ori = *(reader_.full().pop("Waiting for data"));

            // If the data is formated as object mask, then randomly select a class, and randomly select
            // subset of instances (one instance at least), and thus the segmentation data is a binary mask.
            BBoxSegDatum& bbox_seg_datum = *(new BBoxSegDatum);

            if (is_object_mask && (bbox_seg_datum_ori.annotation_group_size()>0)){
                SelectClsBBoxes(bbox_seg_datum_ori, &bbox_seg_datum);
            } else {
                bbox_seg_datum.CopyFrom(bbox_seg_datum_ori);
            }

            read_time += timer.MicroSeconds();
            timer.Start();
            BBoxSegDatum distort_datum;
            BBoxSegDatum* expand_datum = NULL;
            if (transform_param.has_distort_param()) {
                distort_datum.CopyFrom(bbox_seg_datum);
                this->data_transformer_->DistortImage(bbox_seg_datum.seg_datum(),
                                                      distort_datum.mutable_seg_datum());
                if (transform_param.has_expand_param()) {
                    expand_datum = new BBoxSegDatum();
                    this->data_transformer_->ExpandImage(distort_datum, expand_datum);
                } else {
                    expand_datum = &distort_datum;
                }
            } else {
                if (transform_param.has_expand_param()) {
                    expand_datum = new BBoxSegDatum();
                    this->data_transformer_->ExpandImage(bbox_seg_datum, expand_datum);
                } else {
                    expand_datum = &bbox_seg_datum;
                }
            }
            BBoxSegDatum* sampled_datum = NULL;
            bool has_sampled = false;
            if (batch_samplers_.size() > 0) {
                // Generate sampled bboxes from expand_datum.
                vector<NormalizedBBox> sampled_bboxes;
                GenerateBatchSamples(*expand_datum, batch_samplers_, &sampled_bboxes);
                if (sampled_bboxes.size() > 0) {
                    // Randomly pick a sampled bbox and crop the expand_datum.
                    int rand_idx = caffe_rng_rand() % sampled_bboxes.size();
                    sampled_datum = new BBoxSegDatum();

//                    std::cout<<sampled_bboxes[rand_idx].xmin()<<std::endl;
//                    std::cout<<sampled_bboxes[rand_idx].ymin()<<std::endl;
//                    std::cout<<sampled_bboxes[rand_idx].xmax()<<std::endl;
//                    std::cout<<sampled_bboxes[rand_idx].ymax()<<std::endl;

//                    std::cout<<sampled_bboxes[rand_idx].label()<<std::endl;

                    this->data_transformer_->CropImageSeg(*expand_datum,
                                                       sampled_bboxes[rand_idx],
                                                       sampled_datum);

                    if (is_object_mask && sampled_datum->annotation_group().size()>0){
                        this->data_transformer_->UpdateBinaryMask(sampled_datum->annotation_group().Get(0), sampled_datum->mutable_seg_datum());
                    }
                    else if (is_object_mask && sampled_datum->annotation_group().size()<=0){
                        // set all seg data to ignore label (e.g. 255).
//                        cv::Mat cv_mat;
//                        if (is_object_mask) {
//                            cv_mat = DecodeDatumToCVMatNative(sampled_datum->seg_datum());

//                        } else {
//                            cv_mat = DecodeDatumToCVMatSegNative(sampled_datum->seg_datum());
//                            cv::Mat cv_seg = DecodeDatumToCVMatSegNative(sampled_datum->seg_datum());
//                            cv::Mat cv_seg_update(cv_seg.rows, cv_seg.cols, CV_8UC1, cv::Scalar(255));
//                        }

                        cv::Mat cv_mat = DecodeDatumToCVMatNative(sampled_datum->seg_datum());
                        cv::Mat cv_seg_update(cv_mat.rows, cv_mat.cols, CV_8UC1, cv::Scalar(255));

                        // Encode the cv_seg_update into the seg data.
                        std::vector<uchar> buf;
                        cv::imencode(".png", cv_seg_update, buf);
                        sampled_datum->mutable_seg_datum()->set_seg(std::string(reinterpret_cast<char*>(&buf[0]),
                                                       buf.size()));
                    }

//                    std::cout << expand_datum->annotation_group().size() << std::endl;
//                    std::cout << sampled_datum->annotation_group().size() << std::endl;
                    has_sampled = true;
//                    sampled_datum = expand_datum;
                } else {
                    sampled_datum = expand_datum;
                }
            } else {
                sampled_datum = expand_datum;
            }
            CHECK(sampled_datum != NULL);
            timer.Start();
            vector<int> shape =
                    this->data_transformer_->InferBlobShape(sampled_datum->seg_datum());
            if (transform_param.has_resize_param()) {
                if (transform_param.resize_param().resize_mode() ==
                    ResizeParameter_Resize_mode_FIT_SMALL_SIZE) {
                    this->transformed_data_.Reshape(shape);
                    bbox_seg_batch->data_.Reshape(shape);
                    top_data = bbox_seg_batch->data_.mutable_cpu_data();
                } else {
                    CHECK(std::equal(top_shape.begin() + 1, top_shape.begin() + 4,
                                     shape.begin() + 1));
                }
            } else {
                CHECK(std::equal(top_shape.begin() + 1, top_shape.begin() + 4,
                                 shape.begin() + 1));
            }
            // Apply data transformations (mirror, scale, crop...)
            int offset_img = bbox_seg_batch->data_.offset(item_id);
            this->transformed_data_.set_cpu_data(top_data + offset_img);

            int offset_seg = bbox_seg_batch->seg_.offset(item_id);
            this->transformed_seg_.set_cpu_data(top_seg + offset_seg);

            vector<AnnotationGroup> transformed_anno_vec;

            // Transform datum and annotation_group at the same time
            transformed_anno_vec.clear();
            this->data_transformer_->Transform(*sampled_datum,
                                               &(this->transformed_data_), &(this->transformed_seg_),
                                               &transformed_anno_vec);

            // Count the number of bboxes.
            for (int g = 0; g < transformed_anno_vec.size(); ++g) {
                num_bboxes += transformed_anno_vec[g].annotation_size();
            }

            all_anno[item_id] = transformed_anno_vec;


            // clear memory
            if (has_sampled) {
                delete sampled_datum;
            }
            if (transform_param.has_expand_param()) {
                delete expand_datum;
            }
            trans_time += timer.MicroSeconds();

            reader_.free().push(const_cast<BBoxSegDatum*>(&bbox_seg_datum));
        }

        // Store bbox annotation.
        vector<int> label_shape(4);

        label_shape[0] = 1;
        label_shape[1] = 1;
        label_shape[3] = 8;
        if (num_bboxes == 0) {
            // Store all -1 in the label.
            label_shape[2] = 1;
            bbox_seg_batch->bbox_.Reshape(label_shape);
            caffe_set<Dtype>(8, -1, bbox_seg_batch->bbox_.mutable_cpu_data());
        } else {
            // Reshape the label and store the annotation.
            label_shape[2] = num_bboxes;
            bbox_seg_batch->bbox_.Reshape(label_shape);
            top_bbox = bbox_seg_batch->bbox_.mutable_cpu_data();
            int idx = 0;
            for (int item_id = 0; item_id < batch_size; ++item_id) {
                const vector<AnnotationGroup>& anno_vec = all_anno[item_id];
                for (int g = 0; g < anno_vec.size(); ++g) {
                    const AnnotationGroup& anno_group = anno_vec[g];
                    for (int a = 0; a < anno_group.annotation_size(); ++a) {
                        const Annotation& anno = anno_group.annotation(a);
                        const NormalizedBBox& bbox = anno.bbox();
                        top_bbox[idx++] = item_id;
                        top_bbox[idx++] = anno_group.group_label();
                        top_bbox[idx++] = anno.instance_id();
                        top_bbox[idx++] = bbox.xmin();
                        top_bbox[idx++] = bbox.ymin();
                        top_bbox[idx++] = bbox.xmax();
                        top_bbox[idx++] = bbox.ymax();
                        top_bbox[idx++] = bbox.difficult();
                    }
                }
            }
        }


        timer.Stop();
        batch_timer.Stop();
        DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
        DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
        DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
    }

    INSTANTIATE_CLASS(BBoxSegDataLayer);
    REGISTER_LAYER_CLASS(BBoxSegData);

}  // namespace caffe
