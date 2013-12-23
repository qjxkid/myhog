#ifndef HOGFEATURE_H
#define HOGFEATURE_H
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/gpu/gpu.hpp>

using namespace std;

class HogFeature
{
	//��Ϊ
public:
	HogFeature();
	~HogFeature();

	void CreateHogDescriptor(int image_width,int image_height);
	void CreateHogDescriptor_OnePass(cv::Size win_size=cv::Size(64,128),cv::Size block_size=cv::Size(16,16),cv::Size block_stride=cv::Size(8,8),cv::Size cell_size=cv::Size(8,8),int nbins=9);

	void ExtractHogFeatures(const cv::Mat &image);
	void ExtractHogFeatures(const cv::Mat &image,vector<float> &features);
	void ExtractHogFeatures_OnePass(const cv::Mat &image,cv::Size winStride);
	void ExtractHogFeatures_OnePass(const cv::Mat &image);
	void ExtractHogFeatures_OnePass(const cv::Mat &image,cv::Size winStride,vector<float> &features);
	void ExtractHogFeatures_OnePass(const cv::Mat &image, vector<float> &features);
	unsigned int GetFeaturesDim();

	//��Ϊ
private:
	void Init();
	void Release();
	
	//����
public:
	vector<float> m_features;
	unsigned int m_featuresdim;

	//����
private:
	cv::HOGDescriptor *m_cpuhog;
	
	vector<cv::HOGDescriptor *> m_vec_cpuhog;

	cv::Size m_block_stride;
};

#endif