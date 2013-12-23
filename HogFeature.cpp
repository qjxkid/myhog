#include "HogFeature.h"

HogFeature::HogFeature()
{
	//���캯��
	Init();
}

HogFeature::~HogFeature()
{
	//��������
	if(m_cpuhog)
		delete m_cpuhog;
}

void HogFeature::Init()
{
	m_cpuhog=NULL;
	m_featuresdim=36;
}

void HogFeature::Release()
{
	if(m_vec_cpuhog.size()>0)
	{
		for(int i=0;i<m_vec_cpuhog.size();i++)
		{
			delete m_vec_cpuhog[i];
		}
		vector<cv::HOGDescriptor *>().swap(m_vec_cpuhog);
	}
}

void HogFeature::CreateHogDescriptor(int width,int height)
{
	Release();

	cv::HOGDescriptor *tmp_hog;
	for(int w=8;w<(int)(width*2/3);w+=4)
		for(int h=8;h<(int)(height*2/3);h+=4)
		{
			cv::HOGDescriptor *tmphog=new cv::HOGDescriptor(cv::Size(w,h),cv::Size(w,h),cv::Size(w/2,h/2),cv::Size(w/2,h/2),9);
			m_vec_cpuhog.push_back(tmphog);
		}
}

void HogFeature::CreateHogDescriptor_OnePass(cv::Size win_size,cv::Size block_size,cv::Size block_stride,cv::Size cell_size,int nbins)
{
	if(m_cpuhog)
	{
		delete m_cpuhog;
		m_cpuhog=NULL;
	}
	m_block_stride = block_stride;
	m_cpuhog=new cv::HOGDescriptor(win_size,block_size,block_stride,cell_size,nbins);
}

void HogFeature::ExtractHogFeatures(const cv::Mat &image)
{
	m_features.clear();

	CreateHogDescriptor(image.rows,image.cols);
	for(int i=0;i<m_vec_cpuhog.size();i++)
	{
		vector<float> tmpfeatures;
		m_vec_cpuhog[i]->compute(image,tmpfeatures,m_vec_cpuhog[i]->blockStride,cv::Size(0,0));
		m_features.insert(m_features.end(),tmpfeatures.begin(),tmpfeatures.end());
	}
}

void HogFeature::ExtractHogFeatures(const cv::Mat &image,vector<float> &features)
{
	features.clear();

	ExtractHogFeatures(image);
	features=m_features;
}

void HogFeature::ExtractHogFeatures_OnePass(const cv::Mat &image,cv::Size winStride)
{
	m_features.clear();

	if(m_cpuhog)
	{
		vector<float> tmpfeatures;
		m_cpuhog->compute(image,tmpfeatures,winStride,cv::Size(0,0));
		m_featuresdim = tmpfeatures.size();
		m_features.assign(tmpfeatures.begin(),tmpfeatures.end());
	}
}

void HogFeature::ExtractHogFeatures_OnePass(const cv::Mat &image,cv::Size winStride,vector<float> &features)
{
	features.clear();

	ExtractHogFeatures_OnePass(image,winStride);

	if(m_cpuhog)
	{
		features=m_features;

	}
}

void HogFeature::ExtractHogFeatures_OnePass( const cv::Mat &image )
{
	m_features.clear();
	if(m_cpuhog)
	{
		vector<float> tmpfeatures;
		m_cpuhog->compute(image,tmpfeatures,m_block_stride,cv::Size(0,0));
		m_featuresdim = tmpfeatures.size();
		m_features.assign(tmpfeatures.begin(),tmpfeatures.end());
	}
}

void HogFeature::ExtractHogFeatures_OnePass(const cv::Mat &image, vector<float> &features)
{
	features.clear();

	ExtractHogFeatures_OnePass(image);

	if(m_cpuhog)
	{
		features=m_features;
	}
}

unsigned int HogFeature::GetFeaturesDim()
{
	return m_featuresdim;
}