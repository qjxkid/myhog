#include <iostream>
#include <fstream>
#include "HogFeature.h"
#include <opencv2/ml/ml.hpp>
using namespace std;
using namespace cv;
int main(int argc, char **argv){
	HogFeature hogfeature;
	hogfeature.CreateHogDescriptor_OnePass(cv::Size(200,50),cv::Size(20,20),cv::Size(10,10),cv::Size(10,10));

// 	cv::Mat image = cv::imread("f:/jz/logo/3.png");
// 	hogfeature.ExtractHogFeatures_OnePass(image);
//	((200-25)/5+1)*((50-25)/5+1)*25*9 = 48600
//	((200-10)/5+1)*((50-10)/5+1)*4*9 = 12636
//	((200-20)/10+1)*((50-20)/10+1)*4*9 = 2736
// 	cout<<hogfeature.m_features.size()<<endl;
// 	cv::Mat_<float> atraindata(hogfeature.m_features,true);
// 	cout<<atraindata[0][0]<<endl<<atraindata[1][0]<<endl<<atraindata[2][0]<<endl<<atraindata[3][0];

	string filepath;
	cv::Mat image;
	vector<float> tmp_feature;
	vector< vector<float> > feature_vec;
	ifstream flogoin("F:/jz/logo/logo.txt",ios_base::in);
	
	if(!flogoin.is_open())
	{
#ifdef DEBUG_OUTPUT
		cout<<"Open File Error"<<endl;
#endif
		exit(1);
	}
//	int counter=0;
	while(getline(flogoin,filepath))
	{
		image.release();
		image=cv::imread(filepath,0);
		hogfeature.ExtractHogFeatures_OnePass(image);
//		cout<<++counter<<endl;
		feature_vec.push_back(hogfeature.m_features);
	}
	flogoin.close();

	int logonum = feature_vec.size();

	cout<<logonum<<endl;

	ifstream fbkgin("F:/jz/tmp1/tmp1.txt",ios_base::in);

	if(!fbkgin.is_open())
	{
#ifdef DEBUG_OUTPUT
		cout<<"Open File Error"<<endl;
#endif
		exit(1);
	}
//	int counter=0;
	while(getline(fbkgin,filepath))
	{
		image.release();
		image=cv::imread(filepath,0);
		hogfeature.ExtractHogFeatures_OnePass(image);
//		cout<<++counter<<endl;
		feature_vec.push_back(hogfeature.m_features);
	}
	fbkgin.close();

	int bkgnum = feature_vec.size() - logonum;
	cout<<bkgnum<<endl;

	int dim = hogfeature.m_featuresdim;
	int num = feature_vec.size();

 	float **trainingData;
	trainingData = new float *[num];
	int i,j;
	for (i=0; i<num; ++i)
	{
		trainingData[i] = new float[dim];
	}
	
	for (i=0;i<num;++i)
	{
		for (j=0;j<dim;++j)
		{
			trainingData[i][j] = feature_vec[i][j];
		}
	}
	cv::Mat trainingDataMat(num, dim, CV_32FC1, trainingData);

	float *labels = new float[num];
	cv::Mat labelsMat(num, 1, CV_32FC1, labels);
	for (i=0;i<logonum;++i)
	{
		labels[i] = 1.0;
	}
	for (i=logonum; i<num; ++i)
	{
		labels[i] = 2.0;
	}


	CvSVMParams params;
	params.svm_type    = SVM::C_SVC;
	params.kernel_type = SVM::RBF;
	params.term_crit   = TermCriteria(CV_TERMCRIT_ITER, (int)1e11, 1e-10);	cout << "Starting training process" << endl;
	CvSVM svm;
	svm.train_auto(trainingDataMat,labelsMat,cv::Mat(),cv::Mat(),params,10);
	for (i=0; i<num; ++i)
	{
		delete trainingData[i];
	}
	delete trainingData;

	delete labels;

	system("pause");
	return 0;
}