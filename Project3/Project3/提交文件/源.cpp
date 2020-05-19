#define _CRT_SECURE_NO_WARNINGS
#include<iostream>
#include "4k.h"
#include <time.h>

using namespace std;

int main() {
	int w = 3840;
	int h = 2160;

	const char* pathTmp = "./data/pic001.yuv";

	FILE* pFileIn = fopen(pathTmp, "rb+");
	int bufLen = w * h * 3 / 2;

	unsigned char* pYuvBuf = new unsigned char[bufLen];
	fread(pYuvBuf, bufLen * sizeof(unsigned char), 1, pFileIn);

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

	cout << "qualityscore_1:" << result.qualityscore_1 << endl;
	cout << "qualityscore_2:" << result.qualityscore_2 << endl;
	cout << "qualityscore:" << result.qualityscore << endl;
	cout << "is4k_flag:" << result.is4k_flag << endl;


	return 0;
}

