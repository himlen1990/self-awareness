#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <boost/algorithm/string.hpp>

using namespace std;
using namespace caffe;  // NOLINT(build/namespaces)                
using namespace cv;
using namespace boost;
using std::string;

typedef std::pair<string, float> Prediction;

class Classifier {
public:
  Classifier(const string& model_file,
             const string& trained_file);                                 
  //const string& label_file);                                            
  std::vector<Prediction> Classify(const vector<cv::Mat>& imgs, const vector<float>& joint, int N = 5);
  std::vector<float> Predict(const vector<cv::Mat>& imgs, const vector<float>& joint, int clip_length_, bool preprocessed = false);
  cv::Size get_input_geometry()  {return input_geometry_;}
  cv::Mat get_mean_img() {return mean_;}
  int get_num_channels() {return num_channels_;}

private:
  void SetMean(float channel1_mean, float channel2_mean, float channel3_mean);



  void WrapInputLayer(std::vector<cv::Mat>& input_channels, std::vector<cv::Mat>& clip_input, std::vector<cv::Mat>& joint_input);


  void Preprocess(const vector<cv::Mat>& imgs,
		  std::vector<cv::Mat>& input_channels,
		  const vector<float>& joint,
		  vector<cv::Mat>& joint_input,
		  vector<cv::Mat>& clip_input,
		  bool preprocessed = false);   //raw_images_preprocess
  


private:
  shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  int clip_length_;
  int num_joint_pos_;
  cv::Mat mean_;
  std::vector<string> labels_;
};

Classifier::Classifier(const string& model_file,
                       const string& trained_file):clip_length_(12)
{                                                                          
  //const string& label_file) {                                           
  /*#ifdef CPU_ONLY
  Caffe::set_mode(Caffe::CPU);
#else
     Caffe::set_mode(Caffe::GPU);
     #endif*/
  /* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(trained_file);
  
  // image input
  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
  // clip input
  Blob<float>* clip_input_layer = net_->input_blobs()[1];

  //joint input
  Blob<float>* joint_input_layer = net_->input_blobs()[2];
  num_joint_pos_ = joint_input_layer->channels();



  /* Load the binaryproto mean file. */
  SetMean(103.939, 116.779 , 128.68);                                      
  Blob<float>* output_layer = net_->output_blobs()[0];


}

static bool PairCompare(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs) {
  return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
static std::vector<int> Argmax(const std::vector<float>& v, int N) {
  std::vector<std::pair<float, int> > pairs;
  for (size_t i = 0; i < v.size(); ++i)
    pairs.push_back(std::make_pair(v[i], i));
  std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

  std::vector<int> result;
  for (int i = 0; i < N; ++i)
    result.push_back(pairs[i].second);
  return result;
}

std::vector<Prediction> Classifier::Classify(const vector<cv::Mat>& imgs, const vector<float>& joint, int N) {
  std::vector<float> output = Predict(imgs, joint, clip_length_);

  N = std::min<int>(labels_.size(), N);
  std::vector<int> maxN = Argmax(output, N);
  std::vector<Prediction> predictions;
  for (int i = 0; i < N; ++i) {
    int idx = maxN[i];
    predictions.push_back(std::make_pair(labels_[idx], output[idx]));
  }

  return predictions;
}


void Classifier::SetMean(float channel1_mean, float channel2_mean, float channel3_mean) {

  std::vector<cv::Mat> channels;


    cv::Mat channel1(227, 227, CV_32FC1, channel1_mean);
    channels.push_back(channel1);
    cv::Mat channel2(227, 227, CV_32FC1, channel2_mean);
    channels.push_back(channel2);
    cv::Mat channel3(227, 227, CV_32FC1, channel3_mean);
    channels.push_back(channel3);


  /* Merge the separate channels into a single image. */
  cv::Mat mean;
  cv::merge(channels, mean);

  /* Compute the global mean pixel value and create a mean image          
   * filled with this value. */
  cv::Scalar channel_mean = cv::mean(mean);
  mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);

}

std::vector<float> Classifier::Predict(const vector<cv::Mat>& imgs, const vector<float>& joint, int clip_length_, bool preprocessed) {
  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(clip_length_, num_channels_,
                       input_geometry_.height, input_geometry_.width);

  Blob<float>* clip_input_layer = net_->input_blobs()[1];
  clip_input_layer->Reshape(clip_length_, 1, 1, 1);

  Blob<float>* joint_input_layer = net_->input_blobs()[2];
  joint_input_layer->Reshape(clip_length_, num_joint_pos_, 1, 1);
  
  /* Forward dimension change to all layers. */
  net_->Reshape();

  std::vector<cv::Mat> input_channels;
  std::vector<cv::Mat> clip_input;
  std::vector<cv::Mat> joint_input;

  WrapInputLayer(input_channels, clip_input, joint_input);

  Preprocess(imgs, input_channels, joint, joint_input, clip_input, preprocessed);
  cout<<"after preprocess"<<endl;

  
  net_->ForwardPrefilled();

  cout<<"after forward"<<endl;
  // Copy the output layer to a std::vector 
  Blob<float>* output_layer = net_->output_blobs()[0];
  
  const float* begin = output_layer->cpu_data();
  const float* end = begin + output_layer->height();
  return std::vector<float>(begin, end);
}


void Classifier::WrapInputLayer(std::vector<cv::Mat>& input_channels, std::vector<cv::Mat>& clip_input, std::vector<cv::Mat>& joint_input) {
  // wrap image input
  Blob<float>* input_layer = net_->input_blobs()[0];
  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels()*clip_length_; ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data); //here, it sends data to input layer via a pointer input_data
    input_channels.push_back(channel);
    input_data += width * height;
  }

  // wrap and preprocess clip input
  Blob<float>* clip_input_layer = net_->input_blobs()[1];
  float* clip_input_data = clip_input_layer->mutable_cpu_data();
  for (int i = 0; i < clip_input_layer->channels()*clip_length_; ++i) {
    cv::Mat clip_channel(1, 1, CV_32FC1, clip_input_data);
    clip_input.push_back(clip_channel);
    clip_input_data +=1;
    }

  // wrap joint input
  Blob<float>* joint_input_layer = net_->input_blobs()[2];
  float* joint_input_data = joint_input_layer->mutable_cpu_data();
  for (int i = 0; i < joint_input_layer->channels()*clip_length_; ++i) {
    cv::Mat joint_channel(1, 1, CV_32FC1, joint_input_data);
    joint_input.push_back(joint_channel);
    joint_input_data += 1;
  }

}


void Classifier::Preprocess(const vector<cv::Mat>& imgs,
                            std::vector<cv::Mat>& input_channels,
			    const vector<float>& joint,
			    vector<cv::Mat>& joint_input,
			    vector<cv::Mat>& clip_input,
			    bool preprocessed ) {
  /* Convert the input image to the input image format of the network. */
  if(preprocessed)
    {
      if(imgs.size() != clip_length_ * num_channels_)
	cout<<"processed imgs have the wrong size, please check before call predict"<<endl;
      else
	{
	  for(int i=0; i<clip_length_ * num_channels_; i++)
	    {
	      imgs[i].copyTo(input_channels[i]);
	    }
	}
	
    }
  else
    {
      for(int i=0; i<clip_length_; i++)
	{
	  cv::Mat sample_resized;
	  if (imgs[i].size() != input_geometry_)
	    cv::resize(imgs[i], sample_resized, input_geometry_);
	  else
	    sample_resized = imgs[i];
      
	  cv::Mat sample_float;
	  if (num_channels_ == 3)
	    sample_resized.convertTo(sample_float, CV_32FC3);
	  else
	    sample_resized.convertTo(sample_float, CV_32FC1);
      
	  cv::Mat sample_normalized;
	  cv::subtract(sample_float, mean_, sample_normalized);

	  /* This operation will write the separate BGR planes directly to the    
	   * input layer of the network because it is wrapped by the cv::Mat      
	   * objects in input_channels. */
	  vector<Mat> splited_channels;
	  cv::split(sample_normalized, splited_channels);
     
	  for (int j=0; j<num_channels_; j++)
	    {
	  
	      splited_channels[j].copyTo(input_channels[i*num_channels_+j]);
	    }
	}
    }
  CHECK(reinterpret_cast<float*>(input_channels.at(0).data)
	      == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
  
  //process clip input
  for(int i=1; i<clip_length_; i++)
    clip_input[i].at<float>(0,0) = 1;

  //process joint input
  for(int i=0; i<joint.size(); i++)
    {
    joint_input[i].at<float>(0,0) = joint[i]; 
    }
}

