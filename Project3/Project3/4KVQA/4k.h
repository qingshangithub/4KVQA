#pragma once

#include <string>
#include "svm.h"
using namespace std;
extern "C" __declspec(dllimport) double dllvqa(int** data, float**rescale_vector_4k_1, float**rescale_vector_4k_2, svm_model* model1, svm_model* model2, unsigned char* pYuvBuf);
extern "C" __declspec(dllimport) int** loadMatlab(string paraPath);
extern "C" __declspec(dllimport) float** read_range_file_4k_fast(string range_fname);
extern "C" __declspec(dllimport) struct svm_model *svm_load_model(const char *model_file_name);
extern "C" __declspec(dllimport) void svm_free_and_destroy_model(struct svm_model **model_ptr_ptr);