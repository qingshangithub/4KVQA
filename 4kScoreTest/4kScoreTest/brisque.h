#ifndef JD_BRISQUE
#define JD_BRISQUE

#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <math.h>
#include "opencv/cv.h"
#include "opencv/cxcore.h"
#include "opencv/highgui.h"
#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>  
#include <opencv2/core/core.hpp>
#include <vector>
#include <string.h>
//using namespace cv;
using namespace std;
#define M_PI 3.14159265358979323846
#define kernelSize  11
#define kernelSigma 1.5

typedef struct MyStruct
{
	double final_distance = 0;
	int H_dct[10];
	int X_dct[4];
};

int find_option( int argc, char **argv, const char *option );
int read_int( int argc, char **argv, const char *option, int default_value );
char *read_string( int argc, char **argv, const char *option, char *default_value);
extern float rescale_vector[36][2];


template<class T> class Image
{
  private:

  IplImage* imgp;


  public:

  Image(IplImage* img=0)
  {
   imgp=img;
  }
  ~Image()
  {
   imgp=0;
  }
  void operator=(IplImage* img) 
  {
    imgp=img;
  }
  inline T* operator[](const int rowIndx)
  {
     return ((T *)(imgp->imageData + rowIndx*imgp->widthStep));
  }
};

typedef Image<double> BwImage;

//function declarations
void AGGDfit(IplImage* structdis, double& lsigma_best, double& rsigma_best, double& gamma_best);
void ComputeBrisqueFeature(IplImage *orig, vector<double>& featurevector);
MyStruct compute_FeatureSD(cv::Mat &src_Img,int** data_dct);
void compute_LvValue_FeatureVar(cv::Mat &src_Img, double * lvValue, double *featureVar, int gaussian_kernelSize, float gaussian_kernelSigma);
void compute_LvValue_FeatureVar_fast(cv::Mat &src_Img, double * lvValue, double *featureVar, cv::Mat *subImg, cv::Mat *subImg_1, cv::Mat *subImg_2, cv::Mat *subImg_3, int gaussian_kernelSize, float gaussian_kernelSigma);
double compute_FeatureVar(cv::Mat &sigmaMat, cv::Range *row_Range, cv::Range *col_Range);
bool Compute4KFeature(cv::Mat &src_Img, vector<double>& featurevector);
bool Compute4KFeature_fast(cv::Mat &src_Img, vector<double>& featurevector,int** data_dct);
void trainModel();
float computescore(char* imname);
float computescore_4kQuality(char *imgPath);
float computescore_4kQuality_fast(char *imgPath);
float computescore_tbm_test(char* imname, struct svm_model* model, IplImage* orig);
float computescore_SSIR(char* ipath1, char* ipath2);

template <typename Type>
void  printVector(vector<Type> vec)
{
    for(int i=0; i<vec.size(); i++)
    {
        cout<<i+1<<":"<<vec[i]<<endl;
    }
}

template <typename Type>
void printVectortoFile(char*filename , vector<Type> vec,float score)
{
  FILE* fid = fopen(filename,"a");
  //cout<<"file opened"<<endl;
  fprintf(fid,"%f ",score);
  for(int itr_param = 0; itr_param < vec.size();itr_param++)
    fprintf(fid,"%d:%f ",itr_param+1,vec[itr_param]);
  fprintf(fid,"\n");
  fclose(fid);
}


#endif
