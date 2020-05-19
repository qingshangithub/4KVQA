// 4kScoreTest.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include "brisque.h"
#include "svm.h"
#include <windows.h>
#include <direct.h>
#include <io.h>
#include<fstream>
#include<vector>
#include<string>
//#include "ffmpeghead.h"

bool is4k_flag;

//int read_range_file_4k_fast_5();
float computescore_4kQuality_fast(char *imgPath);
void getFiles(string path, vector<string>& files);
cv::Mat DisplayYUV(unsigned char* pYuvBuf);
extern "C" __declspec(dllexport) void free_range(float **rescale_vector_4k);
extern "C" __declspec(dllexport) void free_dct_data(int **dct_data);
extern "C" __declspec(dllexport) int** loadMatlab(const char* paraPath);
extern "C" __declspec(dllexport) VqaResult dllvqa(int** data,float**rescale_vector_4k_1, float**rescale_vector_4k_2, svm_model* model1, svm_model* model2, unsigned char* pYuvBuf);
extern "C" __declspec(dllexport) float** read_range_file_4k_fast(const char* range_fname);

int main(int argc, _TCHAR* argv[])
{
	int w = 3840;
	int h = 2160;

	const char* pathTmp = "./data/pic001.yuv";

	FILE* pFileIn = fopen(pathTmp, "rb+");
	int bufLen = w * h * 3 / 2;

	unsigned char* pYuvBuf = new unsigned char[bufLen];
	fread(pYuvBuf, bufLen  * sizeof(unsigned char), 1, pFileIn);
	
	//---------------------------------------------------------------------------------------------------------------------------------------
	//start 4kScore program

	//load model only once
	//load polar parameters
	int** dct_data; //Polar conversion parameters
	const char* paraPath = "./model/parametersData.txt";
	dct_data = loadMatlab(paraPath);
	int a = 0;
	//load model1
	const char* modelfile_1 = "./model/svm_1.model";
	struct svm_model* model_1;
	if ((model_1 = svm_load_model(modelfile_1)) == 0) {
		fprintf(stderr, "can't open model file allmodel\n");
		printf("can't open model file allmodel\n");
		exit(1);
	}
	//load model2
	const char* modelfile_2 = "./model/svm_2.model";
	struct svm_model* model_2;
	if ((model_2 = svm_load_model(modelfile_2)) == 0) {
		fprintf(stderr, "can't open model file allmodel\n");
		printf("can't open model file allmodel\n");
		exit(1);
	}
	//load range file1
	float** rescale_vector_4k_1;
	const char* range_fname_1 = "./model/svm_1.range";
	rescale_vector_4k_1 = read_range_file_4k_fast(range_fname_1);

	//load range file2
	float** rescale_vector_4k_2;
	const char* range_fname_2 = "./model/svm_2.range";
	rescale_vector_4k_2 = read_range_file_4k_fast(range_fname_2);

	//--------------------------------------------------------------------------------------------------------
	//calculate 4k scores for each yuv file
	VqaResult result;
	result = dllvqa(dct_data, rescale_vector_4k_1, rescale_vector_4k_2, model_1, model_2, pYuvBuf);

	//--------------------------------------------------------------------------------------------------------
	//destroy model only once
	svm_free_and_destroy_model(&model_1);
	svm_free_and_destroy_model(&model_2);
	free_range(rescale_vector_4k_1);
	free_range(rescale_vector_4k_2);
	free_dct_data(dct_data);

	cout << "qualityscore_1:"<<result.qualityscore_1 << endl;
	cout << "qualityscore_2:" << result.qualityscore_2 << endl;
	cout << "qualityscore:" << result.qualityscore << endl;
	cout << "is4k_flag:" << result.is4k_flag << endl;


	return 0;
}


void getFiles(string path, vector<string>& files)
{
	//文件句柄  
	intptr_t   hFile = 0;
	//文件信息  
	struct _finddata_t fileinfo;
	string p;
	intptr_t HANDLE = _findfirst(p.assign(path).append("\\*.*").c_str(), &fileinfo);
	if ((hFile = _findfirst(p.assign(path).append("\\*.*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			//如果是目录,迭代之  
			//如果不是,加入列表  
			if ((fileinfo.attrib &  _A_SUBDIR))
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
					getFiles(p.assign(path).append("\\").append(fileinfo.name), files);
			}
			else
			{
				files.push_back(p.assign(path).append("\\").append(fileinfo.name));
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}

float** read_range_file_4k_fast(const char* range_fname) {
	float **rescale_vector_4k;
	rescale_vector_4k = (float **)malloc(25 * sizeof(float *));
	for (int i = 0; i < 25; i++)
		rescale_vector_4k[i] = (float *)malloc(2 * sizeof(float));

	//check if file exists
	char buff[100];
	int i;
	//string range_fname = "./model/svm_1.range";
	
	FILE* range_file = fopen(range_fname, "r");

	if (range_file == NULL)
	{
		printf("Load svm.range failed\n");
	}
	//assume standard file format for this program
	fgets(buff, 100, range_file);
	fgets(buff, 100, range_file);
	//now we can fill the array
	//for (i = 0; i < 11; ++i) {
	for (i = 0; i < 25; ++i) {
		float a, b, c;
		fscanf(range_file, "%f %f %f", &a, &b, &c);
		rescale_vector_4k[i][0] = b;
		rescale_vector_4k[i][1] = c;
	}
	fclose(range_file);
	return rescale_vector_4k;
}

void free_range(float **rescale_vector_4k) {
	for (int i = 0; i < 25; i++)
	{
		free(rescale_vector_4k[i]);
	}
	free(rescale_vector_4k);
}

void free_dct_data(int **dct_data) {
	for (int i = 0; i < 241; i++)
	{
		free(dct_data[i]);
	}
	free(dct_data);
}

//
//int read_range_file_4k_fast_5() {
//	//check if file exists
//	char buff[100];
//	int i;
//	string range_fname = "./model/svm_5.range";
//
//	FILE* range_file = fopen(range_fname.c_str(), "r");
//
//	if (range_file == NULL)
//	{
//		printf("Load svm.range failed\n");
//		return 1;
//	}
//	//assume standard file format for this program
//	fgets(buff, 100, range_file);
//	fgets(buff, 100, range_file);
//	//now we can fill the array
//	//for (i = 0; i < 11; ++i) {
//	for (i = 0; i < 25; ++i) {
//		float a, b, c;
//		fscanf(range_file, "%f %f %f", &a, &b, &c);
//		rescale_vector_4k_5[i][0] = b;
//		rescale_vector_4k_5[i][1] = c;
//	}
//	fclose(range_file);
//	return 0;
//}

//float computescore_4kQuality_fast(char *imgPath)
//{
//	/************load SVM5************/
//	//read_range_file_4k_fast_1();
//	//read_range_file_4k_fast_5();
//	struct svm_model* model_1 = NULL;
//	struct svm_model* model_5 = NULL;
//	double qualityscore_1;
//	double qualityscore_5;
//	int i;
//	/***1. load model***/
//	string modelfile_1 = "./model/svm_1.model"; // 评价分数
//	string modelfile_5 = "./model/svm_5.model"; // 真假参考分数
//	printf("model at %lx\n computescore been called\n", &model_1);
//	if ((model_1 = svm_load_model(modelfile_1.c_str())) == 0) {
//		fprintf(stderr, "can't open model file allmodel\n");
//		printf("can't open model file allmodel\n");
//		exit(1);
//	}
//	printf("model at %lx\n computescore been called\n", &model_5);
//	if ((model_5 = svm_load_model(modelfile_5.c_str())) == 0) {
//		fprintf(stderr, "can't open model file allmodel\n");
//		printf("can't open model file allmodel\n");
//		exit(1);
//	}
//	/***2. read image***/
//	//for test imgpath yfw
//	char *imgPathTest = "./data/imgSlice/1.bmp";
//	cv::Mat src_Img = cv::imread(imgPathTest, 0);
//	vector<double> features4K;
//	if (src_Img.empty())
//	{
//		return -1;
//	}
//	src_Img.convertTo(src_Img, CV_64F, 1.0);
//	/***3. compute feature***/
//	is4k_flag = Compute4KFeature_fast(src_Img, features4K);
//	// clock_t time2 = clock() - time1 - time_begin;
//	//rescale the brisqueFeatures vector from -1 to 1
//	// 评价分数计算
//	struct svm_node x_1[26];
//	for (i = 0; i < 25; ++i) {
//		float min = rescale_vector_4k_1[i][0];
//		float max = rescale_vector_4k_1[i][1];
//		x_1[i].value = -1 + (2.0 / (max - min) * (features4K[i] - min));
//		x_1[i].index = i + 1;
//	}
//	x_1[25].index = -1;
//	int nr_class_1 = svm_get_nr_class(model_1);
//	double *prob_estimates_1 = (double *)malloc(nr_class_1 * sizeof(double));
//	qualityscore_1 = svm_predict_probability(model_1, x_1, prob_estimates_1);
//	if (qualityscore_1 > 100)
//		qualityscore_1 = 100.0f / (1 + exp(-qualityscore_1 / 10.0f));
//	// clock_t time3 = clock() - time1 - time_begin - time2;
//	free(prob_estimates_1);
//	svm_free_and_destroy_model(&model_1);
//
//	// 真假参考分数
//	struct svm_node x_5[26];
//	for (i = 0; i < 25; ++i) {
//		float min = rescale_vector_4k_5[i][0];
//		float max = rescale_vector_4k_5[i][1];
//		x_5[i].value = -1 + (2.0 / (max - min) * (features4K[i] - min));
//		x_5[i].index = i + 1;
//	}
//	x_5[25].index = -1;
//	int nr_class_5 = svm_get_nr_class(model_5);
//	double *prob_estimates_5 = (double *)malloc(nr_class_5 * sizeof(double));
//	qualityscore_5 = svm_predict_probability(model_5, x_5, prob_estimates_5);
//	cout << "score5" << qualityscore_5 << endl;
//	if (qualityscore_5 > 100)
//		qualityscore_5 = 100.0f / (1 + exp(-qualityscore_5 / 10.0f));
//	// clock_t time3 = clock() - time1 - time_begin - time2;
//	free(prob_estimates_5);
//	svm_free_and_destroy_model(&model_5);
//
//	double qualityscore = (qualityscore_1 + qualityscore_5) / 2;
//	is4k_flag = true;
//
//	if (qualityscore_5<88)
//	{
//		is4k_flag = false;
//	}
//	else if (qualityscore_1<70)
//	{
//		is4k_flag = false;
//	}
//
//	//测试用
//	//ofstream outFile;//创建了一个ofstream 对象
//	//outFile.open("information.txt", ios::app);//outFile 与一个文本文件关联
//	//cout << "分数" <<qualityscore << endl;
//	//outFile << qualityscore_1 << endl;
//	//outFile << qualityscore_5 << endl;
//	//outFile << qualityscore << endl;
//	//outFile.close();
//
//	return qualityscore;
//}

int** loadMatlab(const char* paraPath) {
	//Used to load polar conversion parameters

	int **dct_data;
	dct_data = (int **)malloc(241 * sizeof(int *));
	for (int i = 0; i < 241; i++)
		dct_data[i] = (int *)malloc(383 * sizeof(int));
	ifstream fin(paraPath);
	if (!fin.is_open())
	{
		cout << "Error opening file"; exit(1);
	}
	for (int i = 0; i < 241; i++)
	{
		for (int j = 0; j < 382; j++) {
			dct_data[i][j] = 0;
		}
	}
	int tmpi = 0, tmpj = 1, flag = 0;
	while (fin)
	{
		char tmp;
		fin >> noskipws;
		fin >> tmp;

		if (tmp == '\"'&&flag == 0) {
			flag = 1;
			continue;
		}
		if (tmp == ' ' || tmp == '\n')continue;
		if (tmp >= '0'&&tmp <= '9')
		{
			int eachdata = tmp - '0';
			char tmp2;
			do {
				fin >> noskipws;
				fin >> tmp2;
				if (tmp2 == '\n')break;
				if (tmp2 == '\"')
				{
					flag = 0;
					break;
				}
				int addData = tmp2 - '0';
				eachdata = eachdata * 10 + addData;
			} while (1);
			if (flag != 0) {
				dct_data[tmpi][tmpj++] = eachdata;
			}
			else {
				dct_data[tmpi][tmpj++] = eachdata;
				dct_data[tmpi][0] = tmpj-1;//Record the index of the last parameter in the array
				tmpj = 1;
				tmpi++;
			}
		}

	}
	fin.close();
	return dct_data;
}

cv::Mat DisplayYUV(unsigned char* pYuvBuf)
{
	int w = 3840;
	int h = 2160;

	//FILE* pFileIn = fopen(yuvname, "rb+");
	int bufLen = w * h * 3 / 2;
	//unsigned char* pYuvBuf = new unsigned char[bufLen];
	//int iCount = 0;
	//fread(pYuvBuf, bufLen * sizeof(unsigned char), 1, pFileIn);
	cv::Mat yuvImg;
	yuvImg.create(h * 3 / 2, w, CV_8UC1);
	memcpy(yuvImg.data, pYuvBuf, bufLen * sizeof(unsigned char));
	cv::Mat rgbImg;
	cv::Mat rgbImgGray;
	cv::cvtColor(yuvImg, rgbImg, CV_YUV2BGR_I420);
	//cv::imwrite("pic001.bmp", rgbImg);
	cv::cvtColor(rgbImg, rgbImgGray, CV_RGB2GRAY);
	//cv::imwrite("source.bmp",rgbImgGray);
	//delete[] pYuvBuf;

	//fclose(pFileIn);
	return rgbImgGray;
}

VqaResult dllvqa(int** data, float**rescale_vector_4k_1, float**rescale_vector_4k_2, svm_model* model_1, svm_model* model_2, unsigned char* pYuvBuf) {

	int w = 3840;
	int h = 2160;

	//FILE* pFileIn = fopen(yuvname, "rb+");
	int bufLen = w * h * 3 / 2;
	double qualityscore_1;
	double qualityscore_2;
	int i;

	vector<double> features4K;

	cv::Mat src_Img = DisplayYUV(pYuvBuf);
	//cv::imwrite("img.bmp", src_Img);
	//cv::Mat src_Img = cv::imread("test/10.bmp",0);
	src_Img.convertTo(src_Img, CV_64F, 1.0);
	/***3. compute feature***/
	is4k_flag = Compute4KFeature_fast(src_Img, features4K, data);
	// clock_t time2 = clock() - time1 - time_begin;
	//rescale the brisqueFeatures vector from -1 to 1
	// 评价分数计算
	struct svm_node x_1[26];
	for (i = 0; i < 25; ++i) {
		float min = rescale_vector_4k_1[i][0];
		float max = rescale_vector_4k_1[i][1];
		x_1[i].value = -1 + (2.0 / (max - min) * (features4K[i] - min));
		x_1[i].index = i + 1;
	}
	x_1[25].index = -1;
	int nr_class_1 = svm_get_nr_class(model_1);
	double *prob_estimates_1 = (double *)malloc(nr_class_1 * sizeof(double));
	qualityscore_1 = svm_predict_probability(model_1, x_1, prob_estimates_1);
	if (qualityscore_1 > 100)
		qualityscore_1 = 100.0f / (1 + exp(-qualityscore_1 / 10.0f));
	// clock_t time3 = clock() - time1 - time_begin - time2;
	free(prob_estimates_1);

	// 真假参考分数
	struct svm_node x_2[26];
	for (i = 0; i < 25; ++i) {
		float min = rescale_vector_4k_2[i][0];
		float max = rescale_vector_4k_2[i][1];
		x_2[i].value = -1 + (2.0 / (max - min) * (features4K[i] - min));
		x_2[i].index = i + 1;
	}
	x_2[25].index = -1;
	int nr_class_2 = svm_get_nr_class(model_2);
	double *prob_estimates_2 = (double *)malloc(nr_class_2 * sizeof(double));
	qualityscore_2 = svm_predict_probability(model_2, x_2, prob_estimates_2);
	/*cout << "score5" << qualityscore_2 << endl;*/
	if (qualityscore_2 > 100)
		qualityscore_2 = 100.0f / (1 + exp(-qualityscore_2 / 10.0f));
	// clock_t time3 = clock() - time1 - time_begin - time2;
	free(prob_estimates_2);

	double qualityscore = (qualityscore_1 + qualityscore_2) / 2;
	VqaResult finalResult;
	finalResult.qualityscore_1 = qualityscore_1;
	finalResult.qualityscore_2 = qualityscore_2;
	finalResult.qualityscore = qualityscore;
	//cout << "qulityscore_1 " << qualityscore_1 << endl;
	//cout << "qulityscore_2 " << qualityscore_2 << endl;
	//cout << "qualityscore " << qualityscore << endl;

	is4k_flag = true;

	if (qualityscore_2 < 88)
	{
		is4k_flag = false;
	}
	else if (qualityscore_1 < 70)
	{
		is4k_flag = false;
	}
	finalResult.is4k_flag = is4k_flag;
	//cout << "is4k_flag " << is4k_flag << endl;

	if (qualityscore>=0&&qualityscore<=100)
	{
		return finalResult;
	}
	else
	{
		finalResult.qualityscore_1 = -1.0;
		finalResult.qualityscore_2 = -1.0;
		finalResult.qualityscore = -1.0;
		return finalResult;
	}
	
}
