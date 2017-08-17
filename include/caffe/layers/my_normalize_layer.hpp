#ifndef CAFFE_MY_NORMALIZE_LAYER_HPP_
#define CAFFE_MY_NORMALIZE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe{


template <typename Dtype>
class MyNormalizeLayer : public Layer<Dtype>{
public:
	explicit MyNormalizeLayer(const LayerParameter& param)
			:Layer<Dtype>(param){}

	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const {return "MyNormalize";}
	virtual inline int ExactNumBottomBlobs() const {return 1;}
	virtual inline int ExactNumTopBlobs() const {return 1;}

protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	 virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
	 	const vector<Blob<Dtype>*>& top);
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,const vector<Blob<Dtype>*>& bottom);
	 virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
	 	const vector<bool>& propagate_down,const vector<Blob<Dtype>*>& bottom);

	Blob<Dtype> sum_spatial_multiplier_;
	Blob<Dtype> buffer_;
	Blob<Dtype> buffer_channel_;
	bool channel_shared_;
	Dtype eps_;
	Dtype norm_ ;
};

}
#endif