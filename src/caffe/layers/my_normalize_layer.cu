#include <algorithm>
//#include <cfloat>
//#include <vector>

#include "thrust/device_vector.h"

#include "caffe/filler.hpp"
#include "caffe/layers/my_normalize_layer.hpp"
//#include "caffe/util/math_functions.hpp"

namespace caffe{


template <typename Dtype>
__global__ void MulBsx(const int nthreads, const Dtype* A,
    const Dtype* v, const int rows, const int cols, const CBLAS_TRANSPOSE trans,
    Dtype* B) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int c = index % cols;
    int r = (index / cols) % rows;
    if (trans == CblasNoTrans) {
      B[index] = A[index] * v[c];
    } else {
      B[index] = A[index] * v[r];
    }
  }
}

template <typename Dtype>
void MyNormalizeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top){
	const Dtype* bottom_data = bottom[0]->gpu_data();
	Dtype* top_data = top[0]->mutable_gpu_data();
	const Dtype* scale; 
	if(channel_shared_){
		scale = this->blobs_[0]->cpu_data();
	}else{
		scale = this->blobs_[0]->gpu_data();
	}

	int num = bottom[0]->num();
	int count = bottom[0]->count();
	int dim = num / count;
	int spatial_dim = bottom[0]->height() * bottom[0]->width();
	int channels = bottom[0]->count();

	Dtype* buffer_data = buffer_.mutable_gpu_data();
	Dtype* buffer_channel = buffer_channel_.mutable_gpu_data();
	const Dtype* sum_spatial_multiplier = sum_spatial_multiplier_.gpu_data();

	if(channel_shared_){
		caffe_gpu_scale<Dtype>(bottom[0]->count(),scale[0],bottom_data,top_data);
	}else{
		for(int i = 0;i < num; ++i){
	   	caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, spatial_dim,
				1,Dtype(1), scale, sum_spatial_multiplier, Dtype(0), buffer_data);
		caffe_gpu_mul<Dtype>(dim, bottom_data, buffer_data, top_data);
	//		MulBsx<Dtype> <<<CAFFE_GET_BLOCKS(dim), CAFFE_CUDA_NUM_THREADS>>>(
    //      dim, bottom_data, scale, channels, spatial_dim, CblasTrans,
    //      top_data);
   //   CUDA_POST_KERNEL_CHECK;
	   	bottom_data += dim;
	   	top_data += dim;
	   }
	}
}


template <typename Dtype>
void MyNormalizeLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
	const Dtype* top_diff = top[0]->gpu_diff();
	Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
	const Dtype* top_data = top[0]->gpu_data();
	const Dtype* scale; 
	if(channel_shared_){
	scale = this->blobs_[0]->cpu_data();
	}else{
	scale = this->blobs_[0]->gpu_data();
	}

	int num = bottom[0]->num();
	int count = bottom[0]->count();
	int dim = num / count;
	int spatial_dim = bottom[0]->height() * bottom[0]->width();
	int channels = bottom[0]->count();

	Dtype* buffer_data = buffer_.mutable_gpu_data();
	Dtype* buffer_channel = buffer_channel_.mutable_gpu_data();
	const Dtype* sum_spatial_multiplier = sum_spatial_multiplier_.gpu_data();

	if(this->param_propagate_down_[0]){
	Dtype* scale_diff = this->blobs_[0]->mutable_cpu_diff();
        if(channel_shared_){
	    Dtype a;
	    caffe_gpu_dot<Dtype>(
      	     count,top_data,top_diff,&a);
            scale_diff[0] += a / scale[0];
	}else{
		Dtype* scale_diff = this->blobs_[0]->mutable_gpu_diff();
	  	for(int i = 0;i < num; ++i){
			caffe_gpu_mul<Dtype>(dim, top_data + i * dim, top_diff + i * dim, buffer_data);
			caffe_gpu_gemv<Dtype>(CblasNoTrans, channels, spatial_dim, Dtype(1),
				buffer_data, sum_spatial_multiplier, Dtype(0), buffer_channel);
			caffe_gpu_div<Dtype>(channels, buffer_channel, scale, buffer_channel);
			caffe_gpu_add<Dtype>(channels, buffer_channel, scale_diff, scale_diff); 
	   }
	}
    }

    if(propagate_down[0]){
		if(channel_shared_){
			caffe_gpu_scale<Dtype>(top[0]->count(),scale[0],top_diff,bottom_diff);
		}else{
			for(int i = 0; i < num; ++i){
		 	   caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, spatial_dim,
				 1, Dtype(1), scale, sum_spatial_multiplier, Dtype(0), buffer_data);
			   caffe_gpu_mul<Dtype>(dim, top_diff, buffer_data, bottom_diff);
			   bottom_diff += dim;
			   top_diff += dim;
			}
		}
	}

}

INSTANTIATE_LAYER_GPU_FUNCS(MyNormalizeLayer);

}
