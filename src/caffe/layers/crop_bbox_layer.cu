//
// Created by Niu Chuang on 18-1-15.
//

#include <vector>

#include "caffe/layers/crop_bbox_layer.hpp"

namespace caffe {

// Train stage.
template <typename Dtype>
__global__ void set_mask_kernel(const int n, const int height, const int width,
    const int ymin, const int ymax,
    const int xmin, const int xmax,
    Dtype* mask_data) {
  CUDA_KERNEL_LOOP(index, n) {
    const int w = index % width;
    const int h = (index / width) % height;
    if (w >= xmin && w < xmax && h >= ymin && h < ymax){
    	mask_data[index] = Dtype(1);
    }
  }
}

// Test stage.
template <typename Dtype>
__global__ void copy_bbox_data_kernel(const int N, const int channels, const int height, const int width,
    const Dtype* bbox_data, const Dtype* bottom_data, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, N) {
    const int w = index % width;
    const int h = (index / width) % height;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;

    const Dtype xmin_norm = bbox_data[n*8 + 3];
	const Dtype ymin_norm = bbox_data[n*8 + 4];
	const Dtype xmax_norm = bbox_data[n*8 + 5];
	const Dtype ymax_norm = bbox_data[n*8 + 6];

	int xmin = int(floor(xmin_norm * Dtype(width)));
	int ymin = int(floor(ymin_norm * Dtype(height)));
	int xmax = int(ceil(xmax_norm * Dtype(width)));
	int ymax = int(ceil(ymax_norm * Dtype(height)));

	xmin = max(0, xmin);
	ymin = max(0, ymin);
	xmax = min(xmax, width);
	ymax = min(ymax, height);

    if (w >= xmin && w < xmax && h >= ymin && h < ymax){
    	top_data[index] = bottom_data[c*height*width + h*width + w];
    }
  }
}


template <typename Dtype>
void CropBBoxLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	vector<int> top_shape = top[0]->shape();
	const Dtype* gt_data = bottom[1]->gpu_data();
	num_gt_ = bottom[1]->height();
	const Dtype* bottom_data = bottom[0]->gpu_data();
	Dtype* top_data = top[0]->mutable_gpu_data();
	float xmin_norm, ymin_norm, xmax_norm, ymax_norm;
	int label, width, height, xmin, ymin, xmax, ymax, img_id, channels, N;
	vector<int> label_indices;
	int inner_dim = bottom[0]->offset(1);
	width = bottom[0]->width();
	height = bottom[0]->height();
	channels = bottom[0]->channels();
	N = top[0]->count();

	if (this->phase_ == TRAIN) {
//  	        	std::cout << "gpu" << std::endl;
		// Retrieve all ground truth.
		GetGroundTruth(bottom[1]->cpu_data(), num_gt_, background_label_id_,
				use_difficult_gt_, &all_gt_boxes_);

//  	        	std::cout << "get ground truth" << std::endl;

		num_img_ = bottom[0]->num();
		CHECK_LE(all_gt_boxes_.size(), num_img_)
				<< "Number of image with bbox must be less or equal to the number of image.";

		num_class_ = 0;
		for (int i = 0; i < num_img_; ++i) {
			num_class_ += all_gt_boxes_[i].size();
		}

		if (this->phase_ == TRAIN) {
			CHECK_LE(num_class_, num_img_)
					<< "Current version only support one class at most selected for each image at train stage.";
		}

		Dtype* mask_data = NULL;

//  	        	std::cout << width << " " << height << " " << channels << std::endl;
		// compute corp mask
		caffe_gpu_set(mask_crop_.count(), Dtype(0),
				mask_crop_.mutable_gpu_data());
		map<int, LabelBBox>::const_iterator iter_im;
		for (iter_im = all_gt_boxes_.begin(); iter_im != all_gt_boxes_.end();
				++iter_im) {
//  	            	std::cout << "loop" << std::endl;
			img_id = iter_im->first;
			CHECK(img_id >= 0 && img_id < num_img_)
					<< "img_id must be less than the number of images.";
			LabelBBox::iterator it;
			label_indices.clear();
			for (it = all_gt_boxes_[img_id].begin();
					it != all_gt_boxes_[img_id].end(); it++) {
				label_indices.push_back(it->first);
			}
			int num_class_i = label_indices.size();
			mask_data = mask_crop_.mutable_gpu_data() + img_id * inner_dim;
			for (int l = 0; l < num_class_i; ++l) {
				label = label_indices[l];
				vector<NormalizedBBox> bboxes = all_gt_boxes_[img_id][label];
				for (int b = 0; b < bboxes.size(); ++b) {
					xmin_norm = bboxes[b].xmin();
					ymin_norm = bboxes[b].ymin();
					xmax_norm = bboxes[b].xmax();
					ymax_norm = bboxes[b].ymax();

					xmin = static_cast<int>(floor(
							xmin_norm * static_cast<Dtype>(width)));
					ymin = static_cast<int>(floor(
							ymin_norm * static_cast<Dtype>(height)));
					xmax = static_cast<int>(ceil(
							xmax_norm * static_cast<Dtype>(width)));
					ymax = static_cast<int>(ceil(
							ymax_norm * static_cast<Dtype>(height)));

					xmin = std::max(0, xmin);
					ymin = std::max(0, ymin);
					xmax = std::min(xmax, width);
					ymax = std::min(ymax, height);

//  	                        std::cout << xmin << " " << ymin << " " << xmax << " " << ymax << std::endl;

					set_mask_kernel<<<CAFFE_GET_BLOCKS(inner_dim),
							CAFFE_CUDA_NUM_THREADS>>>(inner_dim, height, width,
							ymin, ymax, xmin, xmax, mask_data);
				}
			}
		}

		// crop the bottom data to top blob with crop mask.
		caffe_gpu_mul(top[0]->count(), bottom_data, mask_crop_.gpu_data(),
				top_data);
	} else {
//		std::cout << "gpu" << std::endl;
		// compute output at test stage.
		copy_bbox_data_kernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
				N, channels, height, width, gt_data, bottom_data, top_data);

	}
}

template <typename Dtype>
void CropBBoxLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	if (propagate_down[0]) {
//		std::cout << "gpu" << std::endl;
	  const Dtype* top_diff = top[0]->gpu_diff();
	  const Dtype* mask_crop = mask_crop_.gpu_data();
	  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

	  caffe_gpu_mul<Dtype>(top[0]->count(), top_diff, mask_crop, bottom_diff);
	}
}

INSTANTIATE_LAYER_GPU_FUNCS(CropBBoxLayer);


}  // namespace caffe

