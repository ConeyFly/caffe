#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/my_normalize_layer.hpp"

namespace caffe{

template <typename Dtype>
void MyNormalizeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){

	MyNormalizeParameter norm_param = this->layer_param().my_norm_param();
	channel_shared_ = norm_param.channel_shared();
	LOG(INFO)<<"channel_shared_ = "<<channel_shared_;
	int channels = bottom[0]->channels();
	int spatial_dim = bottom[0]->height() * bottom[0]->width();
	sum_spatial_multiplier_.Reshape(1, 1, bottom[0]->height(), 
			bottom[0]->width());
	caffe_set(spatial_dim,Dtype(1),sum_spatial_multiplier_.mutable_cpu_data());
	buffer_.Reshape(1, bottom[0]->channels(), bottom[0]->height(),bottom[0]->width());
	buffer_channel_.Reshape(1, bottom[0]->channels(), 1, 1);
	if(this->blobs_.size()>0){
		LOG(INFO) << "Skipping parameter initialization";
	}else{
		this->blobs_.resize(1);
		if(channel_shared_){
			this->blobs_[0].reset(new Blob<Dtype>(vector<int>(0)));
		}else{
			this->blobs_[0].reset(new Blob<Dtype>(vector<int>(1,channels)));
		}
		shared_ptr<Filler<Dtype> >scale_filler;
		if(norm_param.has_scale_filler()){
			scale_filler.reset(GetFiller<Dtype>(norm_param.scale_filler()));
		}else{
			FillerParameter filler_param;
			filler_param.set_type("constant");
			filler_param.set_value(1);
			scale_filler.reset(GetFiller<Dtype>(filler_param));
		}
		scale_filler->Fill(this->blobs_[0].get());
	}
	norm_=0.3;
}

template <typename Dtype>
void MyNormalizeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){

	CHECK_GE(bottom[0]->num_axes(),2)
		<<"Number of axes of bottom blob must be >=2.";
	top[0]->ReshapeLike(*bottom[0]);
	int spatial_dim = bottom[0]->height() * bottom[0]->width();
	if(spatial_dim != sum_spatial_multiplier_.count()){
		sum_spatial_multiplier_.Reshape(1, 1, bottom[0]->height(), 
			bottom[0]->width());
		caffe_set(spatial_dim,Dtype(1),sum_spatial_multiplier_.mutable_cpu_data());
	}
	buffer_.Reshape(1, bottom[0]->num(), bottom[0]->height(),bottom[0]->width());
	norm_ = 0.3;
}

template <typename Dtype>
void MyNormalizeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* top_data = top[0]->mutable_cpu_data();
	const Dtype* scale = this->blobs_[0]->cpu_data();
	int num = bottom[0]->num();
	int count = bottom[0]->count();
	int dim = count / num;
	int spatial_dim = bottom[0]->height() * bottom[0]->width();
	int channels = bottom[0]->channels();
	const Dtype* sum_spatial_multiplier = sum_spatial_multiplier_.cpu_data();
	Dtype* buffer_data = buffer_.mutable_cpu_data();
	if(channel_shared_){
		caffe_cpu_scale<Dtype>(count, scale[0], bottom_data, top_data);
	}else{
		for(int i = 0; i < num; ++i){
			caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, spatial_dim, 1,
				Dtype(1), scale, sum_spatial_multiplier, Dtype(0), buffer_data);
			caffe_mul<Dtype>(dim, bottom_data, buffer_data, top_data);
			bottom_data += dim;
			top_data += dim;
		}
	}
}

template <typename Dtype>
void MyNormalizeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down,const vector<Blob<Dtype>*>& bottom){
	const Dtype* top_diff = top[0]->cpu_diff();
	Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	const Dtype* scale = this->blobs_[0]->cpu_data();
	const Dtype* top_data = top[0]->cpu_data();
	int num = top[0]->num();
	int count = top[0]->count();
	int dim = count / num;
	int channels = top[0]->channels();
	int spatial_dim = top[0]->height() * top[0]->width();
	const Dtype* sum_spatial_multiplier = sum_spatial_multiplier_.cpu_data();
	Dtype* buffer_data = buffer_.mutable_cpu_data();
	Dtype* buffer_channel = buffer_channel_.mutable_cpu_data();
	
	if(this->param_propagate_down_[0]){
		Dtype* scale_diff = this->blobs_[0]->mutable_cpu_diff();
		if(channel_shared_){
			scale_diff[0] += caffe_cpu_dot<Dtype>(
				count, top_data, top_diff)/scale[0];
		}else{
			for(int i = 0; i < num; ++i){
				caffe_mul<Dtype>(dim, top_data + i * dim, top_diff + i*dim, buffer_data);
				caffe_cpu_gemv<Dtype>(CblasNoTrans, channels, spatial_dim, Dtype(1),
					buffer_data, sum_spatial_multiplier, Dtype(0),
					buffer_channel);
				caffe_div<Dtype>(channels, buffer_channel, scale, buffer_channel);
				caffe_add<Dtype>(channels, buffer_channel, scale_diff,scale_diff);
			}
		}
	}
	
	if(propagate_down[0]){	
		if(channel_shared_){
			caffe_cpu_scale<Dtype>(count,scale[0],top_diff,bottom_diff);
			LOG(INFO)<<"top_data = "<<top[0]->cpu_diff()[0];
			LOG(INFO)<<"bottom_data = "<<bottom[0]->cpu_diff()[0];
		}else{
			for(int i = 0; i < num; ++i){
				caffe_cpu_gemm<Dtype>(CblasNoTrans,CblasNoTrans, channels, spatial_dim, 1,
					Dtype(1), scale, sum_spatial_multiplier, Dtype(0),buffer_data);
				caffe_mul<Dtype>(dim , top_diff, buffer_data, bottom_diff);
				bottom_diff += dim;
				top_diff += dim;
			}
		}
	}
}

#ifdef CPU_ONLF
STUB_GPU(MyNormalizeLayer);
#endif

INSTANTIATE_CLASS(MyNormalizeLayer);
REGISTER_LAYER_CLASS(MyNormalize);

}
