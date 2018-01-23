//
// Created by Niu Chuang on 18-1-18.
//

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/psroi_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void PSROI_forward_kernel(const int nthreads, const int num_class,
    const int channels, const int height, const int width, const int roi_size,
    const Dtype* const bottom_data, const Dtype* const bbox_data, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
		const int w = index % width;
		const int h = (index / width) % height;
		const int c = (index / width / height) % channels;
		const int n = index / width / height / channels;

		const Dtype xmin_norm = bbox_data[n * 8 + 3];
		const Dtype ymin_norm = bbox_data[n * 8 + 4];
		const Dtype xmax_norm = bbox_data[n * 8 + 5];
		const Dtype ymax_norm = bbox_data[n * 8 + 6];

		int xmin = int(floor(xmin_norm * Dtype(width)));
		int ymin = int(floor(ymin_norm * Dtype(height)));
		int xmax = int(ceil(xmax_norm * Dtype(width)));
		int ymax = int(ceil(ymax_norm * Dtype(height)));

		xmin = max(0, xmin);
		ymin = max(0, ymin);
		xmax = min(xmax, width);
		ymax = min(ymax, height);

		const int bbox_width = xmax - xmin;
		const int bbox_height = ymax - ymin;
		const int bin_size_width = int(round(Dtype(bbox_width) / Dtype(roi_size)));
		const int bin_size_height = int(round(Dtype(bbox_height) / Dtype(roi_size)));

		const int channels_per_group = num_class*roi_size*roi_size;
		const int area = width*height;
		const int image_id = bbox_data[n*8];

		if (w >= xmin && w < xmax && h >= ymin && h < ymax){
			const int bin_id_x = min(roi_size-1, (w-xmin) / bin_size_width);
			const int bin_id_y = min(roi_size-1, (h-ymin) / bin_size_height);
			const int position_index = bin_id_y*roi_size + bin_id_x;
			const int data_index = image_id*2*channels_per_group*area + c*channels_per_group*area
					+ position_index*area + h*width + w;
			top_data[index] = bottom_data[data_index];
		}

  }
}

template <typename Dtype>
void PSROILayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* bbox_data = bottom[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int N = top[0]->count();
  const int channels = top[0]->channels();
  const int height = top[0]->height();
  const int width = top[0]->width();
  caffe_gpu_set(N, Dtype(0), top_data);

  PSROI_forward_kernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, num_class_,
    channels, height, width, roi_size_, bottom_data, bbox_data, top_data);

  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void PSROI_backward_kernel(const int nthreads,
    const int channels, const int height, const int width, const int roi_size,
    const int xmin, const int xmax, const int ymin, const int ymax,
    const Dtype* const top_diff, Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
	  const int w = index % width;
	  const int h = (index / width) % height;
	  const int c = index / width / height;

	  const int bbox_width = xmax - xmin;
	  const int bbox_height = ymax - ymin;
	  const int bin_size_width = int(round(Dtype(bbox_width) / Dtype(roi_size)));
	  const int bin_size_height = int(round(Dtype(bbox_height) / Dtype(roi_size)));

	  const int group_id = c > (channels/2 -1) ? 1:0;

	  if (w >= xmin && w < xmax && h >= ymin && h < ymax){
		  const int bin_id_x = c % roi_size;
		  const int bin_id_y = (c / roi_size) % roi_size;

		  const int w_start = xmin + bin_id_x * bin_size_width;

		  int w_end;
		  if (bin_id_x == (roi_size-1)){
			  w_end = xmax;
		  } else {
		      w_end = w_start + bin_size_width;
		  }
		  const int h_start = ymin + bin_id_y * bin_size_height;

		  int h_end;
		  if (bin_id_y == (roi_size-1)){
			  h_end = ymax;
		  } else {
		     h_end = h_start + bin_size_height;
		  }

		  if (w >= w_start && w < w_end && h >= h_start && h < h_end){
			  bottom_diff[index] += top_diff[group_id*height*width + h*width + w];
		  }
	  }
  }
}

template <typename Dtype>
void PSROILayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->gpu_diff();

  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int N = bottom[0]->count();
  const int channels = bottom[0]->channels();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();
  const int dim = channels * height * width;

  const int num_bbox = bottom[1]->height();
  const Dtype* bbox_data = bottom[1]->cpu_data();
  caffe_gpu_set(N, Dtype(0), bottom_diff);
	for (int b = 0; b < num_bbox; ++b) {
		const int image_id = bbox_data[b * 8];
		const Dtype xmin_norm = bbox_data[b * 8 + 3];
		const Dtype ymin_norm = bbox_data[b * 8 + 4];
		const Dtype xmax_norm = bbox_data[b * 8 + 5];
		const Dtype ymax_norm = bbox_data[b * 8 + 6];

		int xmin = int(floor(xmin_norm * Dtype(width)));
		int ymin = int(floor(ymin_norm * Dtype(height)));
		int xmax = int(ceil(xmax_norm * Dtype(width)));
		int ymax = int(ceil(ymax_norm * Dtype(height)));

		xmin = max(0, xmin);
		ymin = max(0, ymin);
		xmax = min(xmax, width);
		ymax = min(ymax, height);

		bottom_diff = bottom[0]->mutable_gpu_diff() + image_id*channels*height*width;
		top_diff = top[0]->gpu_diff() + b*top[0]->offset(1);

		PSROI_backward_kernel<<<CAFFE_GET_BLOCKS(dim), CAFFE_CUDA_NUM_THREADS>>>(dim, channels,
				height, width, roi_size_, xmin, xmax, ymin, ymax, top_diff, bottom_diff);

	}

  CUDA_POST_KERNEL_CHECK;
}


INSTANTIATE_LAYER_GPU_FUNCS(PSROILayer);


}  // namespace caffe
