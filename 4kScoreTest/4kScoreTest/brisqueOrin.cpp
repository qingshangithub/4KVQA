#include "stdafx.h"
#include "brisque.h"
#include<time.h>
/**********************************
时间：2019.7.31
GPU 并行加速
作用  包含CUDA相关的头文件
4k 图像:3840*2160
****************************************/
//////////////////////陈康的gpu优化 头文件///////////////////////////////
//#include <cuda_runtime.h>
//////////////////////ck的gpu优化 外部声明函数///////////////////////////////
extern "C" void cuda_test(cv::Mat &src_Img);
////////////////////////////////////////////////////////////////////////////////
//function definitions
void ComputeBrisqueFeature(IplImage *orig, vector<double>& featurevector)
{
    IplImage *orig_bw_int = cvCreateImage(cvGetSize(orig), orig->depth, 1); 
    cvCvtColor(orig, orig_bw_int, CV_RGB2GRAY);
    IplImage *orig_bw = cvCreateImage(cvGetSize(orig_bw_int), IPL_DEPTH_64F, 1);
    cvConvertScale(orig_bw_int, orig_bw, 1.0/255);
    cvReleaseImage(&orig_bw_int);
    
    //orig_bw now contains the grayscale image normalized to the range 0,1
    
    int scalenum = 2;
    for (int itr_scale = 1; itr_scale<=scalenum; itr_scale++)
	{
		IplImage *imdist_scaled = cvCreateImage(cvSize(orig_bw->width/pow((double)2,itr_scale-1), orig_bw->height/pow((double)2,itr_scale-1)), IPL_DEPTH_64F, 1);
		cvResize(orig_bw, imdist_scaled,CV_INTER_CUBIC); 
		
		//compute mu and mu squared
		IplImage* mu = cvCreateImage(cvGetSize(imdist_scaled), IPL_DEPTH_64F, 1);
		cvSmooth( imdist_scaled, mu, CV_GAUSSIAN, 7, 7, 1.16666 );
		IplImage* mu_sq = cvCreateImage(cvGetSize(imdist_scaled), IPL_DEPTH_64F, 1);
		cvMul(mu, mu, mu_sq);

		//compute sigma
		IplImage* sigma = cvCreateImage(cvGetSize(imdist_scaled), IPL_DEPTH_64F, 1);
		cvMul(imdist_scaled, imdist_scaled, sigma);
		cvSmooth(sigma, sigma, CV_GAUSSIAN, 7, 7, 1.16666 );
		cvSub(sigma, mu_sq, sigma);
		cvPow(sigma, sigma, 0.5);

		//compute structdis = (x-mu)/sigma
		cvAddS(sigma, cvScalar(1.0/255), sigma);
		IplImage* structdis = cvCreateImage(cvGetSize(imdist_scaled), IPL_DEPTH_64F, 1);
		cvSub(imdist_scaled, mu, structdis);
		cvDiv(structdis, sigma, structdis);

		//Compute AGGD fit
        double lsigma_best, rsigma_best, gamma_best;
        AGGDfit(structdis, lsigma_best, rsigma_best, gamma_best);
		featurevector.push_back(gamma_best);
		featurevector.push_back((lsigma_best*lsigma_best + rsigma_best*rsigma_best)/2);
		
		//Compute paired product images
		int shifts[4][2]={{0,1},{1,0},{1,1},{-1,1}};
		for(int itr_shift=1; itr_shift<=4; itr_shift++)
		{
			int* reqshift = shifts[itr_shift-1];

                        IplImage* shifted_structdis = cvCreateImage(cvGetSize(imdist_scaled), IPL_DEPTH_64F, 1);
			BwImage OrigArr(structdis);
			BwImage ShiftArr(shifted_structdis);
			for(int i=0; i<structdis->height; i++)
			{
				for(int j=0; j<structdis->width; j++)
				{
					if(i+reqshift[0]>=0 && i+reqshift[0]<structdis->height && j+reqshift[1]>=0 && j+reqshift[1]<structdis->width)
					{
						ShiftArr[i][j]=OrigArr[i+reqshift[0]][j+reqshift[1]];
					}
					else
					{
						ShiftArr[i][j]=0;
					}
				}
			}
		
			//computing correlation
			cvMul(structdis, shifted_structdis, shifted_structdis);
			AGGDfit(shifted_structdis, lsigma_best, rsigma_best, gamma_best);
		
			double constant = sqrt(tgamma(1/gamma_best))/sqrt(tgamma(3/gamma_best));
			double meanparam = (rsigma_best-lsigma_best)*(tgamma(2/gamma_best)/tgamma(1/gamma_best))*constant;
			
			featurevector.push_back(gamma_best);
			featurevector.push_back(meanparam);
			featurevector.push_back(pow(lsigma_best,2));
			featurevector.push_back(pow(rsigma_best,2));

            cvReleaseImage(&shifted_structdis);
		}

        cvReleaseImage(&mu);
		cvReleaseImage(&mu_sq);
		cvReleaseImage(&sigma);
		cvReleaseImage(&structdis);
		cvReleaseImage(&imdist_scaled);
	}
	
}


bool Compute4KFeature(cv::Mat &src_Img, vector<double>& featureVector4K)
{
	double feature_SD = 0.0;
	double lvValue = 0.0;
	double feature_Var = 0.0;
	bool is4K = false;
	clock_t start, finish;
	double totaltime;
	start = clock();
	feature_SD = compute_FeatureSD(src_Img);//待优化1    1.878s
	finish = clock();
	totaltime = (double)(finish - start) / CLOCKS_PER_SEC;
	//printf("compute_FeatureSD:%f\r\n", totaltime);
	/////////////////////////////////
	start = clock();
	compute_LvValue_FeatureVar(src_Img, &lvValue, &feature_Var, kernelSize, kernelSigma);//待优化2  2.378s
	finish = clock();
	totaltime = (double)(finish - start) / CLOCKS_PER_SEC;
	//printf("compute_LvValue_FeatureVar:%f\r\n", totaltime);
	//////////////////////////////////////////////////////////
	if (lvValue <= 40)
		feature_SD = feature_SD - (0.02 + (50 - lvValue)*0.002);
	if (feature_SD < 0.00001)
		feature_SD = 0.12;

	if (feature_SD < 0.115)
		is4K = true;

	IplImage* orig = cvLoadImage("./temp/subImg.bmp");
	vector<double> brisqueFeatures;
	ComputeBrisqueFeature(orig, brisqueFeatures);
	int featureIndex[9] = { 1, 4, 5, 8, 9, 12, 13, 16, 17 };
	for (int i = 0; i < 9; i++)
		featureVector4K.push_back(brisqueFeatures[featureIndex[i]]);
	featureVector4K.push_back(feature_SD);
	featureVector4K.push_back(feature_Var);
	return is4K;
}



//function definitions
void AGGDfit(IplImage* structdis, double& lsigma_best, double& rsigma_best, double& gamma_best)
{
	BwImage ImArr(structdis);
	
	//int width = structdis->width;
	//int height = structdis->height;
	long int poscount=0, negcount=0;
	double possqsum=0, negsqsum=0, abssum=0;
	for(int i=0;i<structdis->height;i++)
	{
		for (int j =0; j<structdis->width; j++)
		{
			double pt = ImArr[i][j];
			if(pt>0)
			{
				poscount++;
				possqsum += pt*pt; 
				abssum += pt;
			}
			else if(pt<0)
			{
				negcount++;
				negsqsum += pt*pt;
				abssum -= pt;
			}
		}
	}
	lsigma_best = pow(negsqsum/negcount, 0.5); //1st output parameter set
	rsigma_best = pow(possqsum/poscount, 0.5); //2nd output parameter set
	 
	double gammahat = lsigma_best/rsigma_best;
	long int totalcount = structdis->width*structdis->height;
	double rhat = pow(abssum/totalcount, static_cast<double>(2))/((negsqsum + possqsum)/totalcount);
	double rhatnorm = rhat*(pow(gammahat,3) +1)*(gammahat+1)/pow(pow(gammahat,2)+1,2);
	
	double prevgamma = 0;
	double prevdiff = 1e10;	
        float sampling = 0.001;
	for (float gam=0.2; gam<10; gam+=sampling) //possible to coarsen sampling to quicken the code, with some loss of accuracy
	{
		double r_gam = tgamma(2/gam)*tgamma(2/gam)/(tgamma(1/gam)*tgamma(3/gam));
		double diff = abs(r_gam-rhatnorm);
		if(diff> prevdiff) break;
		prevdiff = diff;
		prevgamma = gam;
	}
	gamma_best = prevgamma;
}

/**********************************
时间：2019.7.31
对compute_FeatureSD 函数进行 GPU 并行加速
作用 加速测试
注意：后面有保存备份的版本
****************************************/
/**
double compute_FeatureSD(cv::Mat &src_Img)//1.892
{
	clock_t start, finish;
	double totaltime;

	///////////////////////0.909s//////////////////////////
	//start = clock();
	cv::Mat dct_Img;

	double *power_x = new double[src_Img.cols];
	double *power_y = new double[src_Img.rows];
	memset(power_x, 0, src_Img.cols * sizeof(double));
	memset(power_y, 0, src_Img.rows * sizeof(double));

	dct(src_Img, dct_Img);
	double *dct_ImgData = (double *)dct_Img.data;
	//finish = clock();
	///////////////////////0.936s//////////////////////////
	for (int i = 0; i < src_Img.cols; i++)
	{
		for (int j = 0; j < src_Img.rows; j++)
		{
			double *rowPtr = (double *)(dct_Img.ptr(j));
			//double data = (double)dct_ImgData[j * dct_Img.cols + i];
			double data = rowPtr[i];
			power_x[i] += abs(pow(data, 2)) / src_Img.rows;
			power_y[j] += abs(pow(data, 2)) / src_Img.cols;
		}
	}
	//////////////////////GPU 测试////////////////////////
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	if (deviceCount == 0) {
		fprintf(stderr, "error: no devices supporting CUDA.\n");
		exit(EXIT_FAILURE);
	}
	int dev = 0;
	cudaSetDevice(dev);

	cudaDeviceProp devProps;
	if (cudaGetDeviceProperties(&devProps, dev) == 0)
	{
		printf("Using device %d:\n", dev);
		printf("%s; global mem: %dB; compute v%d.%d; clock: %d kHz\n",
			devProps.name, (int)devProps.totalGlobalMem,
			(int)devProps.major, (int)devProps.minor,
			(int)devProps.clockRate);
	}
	///////////这个是CPU RGB to Gray///////////
	start = clock();
    cv::Mat dstimg;
	printf("src_Img.type():%d\n\r", src_Img.type());
	//CV_8UC1
//	cvtColor(src_Img, dstimg, cv::COLOR_BGR2GRAY);
	finish = clock();
	totaltime = (double)(finish - start) / CLOCKS_PER_SEC;
	printf("RGB to Gray on CPU:%f\r\n", totaltime);
	start = clock();
	cuda_test(src_Img);//on GPU
	finish = clock();
	totaltime = (double)(finish - start) / CLOCKS_PER_SEC;
	printf("RGB to Gray on GPU:%f\r\n", totaltime);
	//////////////////////////////////////////////
	/////////////////////////////////0.07s///////////////////////////////////////////////////////////
	//start = clock();
	double max_x = log(1 + power_x[0]), min_x = log(1 + power_x[0]);
	for (int i = 0; i < src_Img.cols; i++)
	{
		power_x[i] = log(1 + power_x[i]);
		if (power_x[i] > max_x)
			max_x = power_x[i];
		if (power_x[i] < min_x)
			min_x = power_x[i];
	}
	//finish = clock();
	double max_y = log(1 + power_y[0]), min_y = log(1 + power_y[0]);
	for (int i = 0; i < src_Img.rows; i++)
	{
		power_y[i] = log(1 + power_y[i]);
		if (power_y[i] > max_y)
			max_y = power_y[i];
		if (power_y[i] < min_y)
			min_y = power_y[i];
	}
	double Sx = 0.0;
	for (int i = 0; i < src_Img.cols; i++)
	{
		power_x[i] /= max_x;
		Sx += power_x[i];
	}
	double Sy = 0.0;
	for (int i = 0; i < src_Img.rows; i++)
	{
		power_y[i] /= max_y;
		Sy += power_y[i];
	}

	double *S_x = new double[src_Img.cols];
	double *S_y = new double[src_Img.rows];
	memset(S_x, 0, src_Img.cols * sizeof(double));
	memset(S_y, 0, src_Img.rows * sizeof(double));
	for (int i = 0; i < src_Img.cols; i++)
	{
		double sum = 0.0;
		for (int j = 0; j <= i; j++)
		{
			sum += power_x[j];
		}
		S_x[i] = sum / Sx;
	}
	for (int i = 0; i < src_Img.rows; i++)
	{
		double sum = 0.0;
		for (int j = 0; j <= i; j++)
		{
			sum += power_y[j];
		}
		S_y[i] = sum / Sy;
	}

	double *gradient_x = new double[src_Img.cols - 1];
	double *gradient_y = new double[src_Img.rows - 1];
	memset(gradient_x, 0, (src_Img.cols - 1) * sizeof(double));
	memset(gradient_y, 0, (src_Img.rows - 1) * sizeof(double));

	for (int i = 0; i < src_Img.cols - 1; i++)
	{
		gradient_x[i] = S_x[i + 1] - S_x[i];
	}
	for (int i = 0; i < src_Img.rows - 1; i++)
	{
		gradient_y[i] = S_y[i + 1] - S_y[i];
	}

	double minGradData_x = abs(gradient_x[0] - 1.0f / src_Img.cols);
	int minPos_x = 0;
	for (int i = 0; i < src_Img.cols - 1; i += 1)
	{
		double data = abs(gradient_x[i] - 1.0f / src_Img.cols);
		if (data < minGradData_x)
		{
			minGradData_x = data;
			minPos_x = i;
		}
	}
	double minGradData_y = abs(gradient_y[0] - 1.0f / src_Img.rows);
	int minPos_y = 0;
	for (int i = 0; i < src_Img.rows - 1; i += 1)
	{
		double data = abs(gradient_y[i] - 1.0f / src_Img.rows);
		if (data < minGradData_y)
		{
			minGradData_y = data;
			minPos_y = i;
		}
	}

	double P_x[2] = { minPos_x + 1, S_x[minPos_x] };
	double P_y[2] = { minPos_y + 1, S_y[minPos_y] };
	double Q1[2] = { 0, 0 };
	double Q2[2] = { src_Img.cols, 1 };
	double Q3[2] = { src_Img.rows, 1 };

	double distance_x = abs((Q2[0] - Q1[0])*(P_x[1] - Q1[1]) - (Q2[1] - Q1[1])*(P_x[0] - Q1[0])) / sqrt(pow(Q2[0] - Q1[0], 2) + pow(Q2[1] - Q1[1], 2));
	double distance_y = abs((Q3[0] - Q1[0])*(P_y[1] - Q1[1]) - (Q3[1] - Q1[1])*(P_y[0] - Q1[0])) / sqrt(pow(Q3[0] - Q1[0], 2) + pow(Q3[1] - Q1[1], 2));
	double final_distance = sqrt(pow(distance_x, 2) * pow(distance_y, 2));

	//totaltime = (double)(finish - start) / CLOCKS_PER_SEC;
	//printf("find 1 max :%f\r\n", totaltime);
	return final_distance;
}*/
/**********************************
时间：2019.7.31
对compute_FeatureSD 进行备份
作用： 备份compute_FeatureSD文件
****************************************/

double compute_FeatureSD(cv::Mat &src_Img)//1.892
{
	clock_t start, finish;
	double totaltime;

	///////////////////////0.909s//////////////////////////
	//start = clock();
	cv::Mat dct_Img;

	double *power_x = new double[src_Img.cols];
	double *power_y = new double[src_Img.rows];
	memset(power_x, 0, src_Img.cols * sizeof(double));
	memset(power_y, 0, src_Img.rows * sizeof(double));

	dct(src_Img, dct_Img);
	double *dct_ImgData = (double *)dct_Img.data;
	//finish = clock();
	///////////////////////0.936s//////////////////////////
	for (int i = 0; i < src_Img.cols; i++)
	{
		for (int j = 0; j < src_Img.rows; j++)
		{
			double *rowPtr = (double *)(dct_Img.ptr(j));
			//double data = (double)dct_ImgData[j * dct_Img.cols + i];
			double data = rowPtr[i];
			power_x[i] += abs(pow(data, 2)) / src_Img.rows;
			power_y[j] += abs(pow(data, 2)) / src_Img.cols;
		}
	}
/////////////////////////////////0.07s///////////////////////////////////////////////////////////
	//start = clock();
	double max_x = log(1 + power_x[0]), min_x = log(1 + power_x[0]);
	for (int i = 0; i < src_Img.cols; i++)
	{
		power_x[i] = log(1 + power_x[i]);
		if (power_x[i] > max_x)
			max_x = power_x[i];
		if (power_x[i] < min_x)
			min_x = power_x[i];
	}
	//finish = clock();
	double max_y = log(1 + power_y[0]), min_y = log(1 + power_y[0]);
	for (int i = 0; i < src_Img.rows; i++)
	{
		power_y[i] = log(1 + power_y[i]);
		if (power_y[i] > max_y)
			max_y = power_y[i];
		if (power_y[i] < min_y)
			min_y = power_y[i];
	}
	double Sx = 0.0;
	for (int i = 0; i < src_Img.cols; i++)
	{
		power_x[i] /= max_x;
		Sx += power_x[i];
	}
	double Sy = 0.0;
	for (int i = 0; i < src_Img.rows; i++)
	{
		power_y[i] /= max_y;
		Sy += power_y[i];
	}

	double *S_x = new double[src_Img.cols];
	double *S_y = new double[src_Img.rows];
	memset(S_x, 0, src_Img.cols * sizeof(double));
	memset(S_y, 0, src_Img.rows * sizeof(double));
	for (int i = 0; i < src_Img.cols; i++)
	{
		double sum = 0.0;
		for (int j = 0; j <= i; j++)
		{
			sum += power_x[j];
		}
		S_x[i] = sum / Sx;
	}
	for (int i = 0; i < src_Img.rows; i++)
	{
		double sum = 0.0;
		for (int j = 0; j <= i; j++)
		{
			sum += power_y[j];
		}
		S_y[i] = sum / Sy;
	}

	double *gradient_x = new double[src_Img.cols - 1];
	double *gradient_y = new double[src_Img.rows - 1];
	memset(gradient_x, 0, (src_Img.cols - 1) * sizeof(double));
	memset(gradient_y, 0, (src_Img.rows - 1) * sizeof(double));

	for (int i = 0; i < src_Img.cols - 1; i++)
	{
		gradient_x[i] = S_x[i + 1] - S_x[i];
	}
	for (int i = 0; i < src_Img.rows - 1; i++)
	{
		gradient_y[i] = S_y[i + 1] - S_y[i];
	}

	double minGradData_x = abs(gradient_x[0] - 1.0f / src_Img.cols);
	int minPos_x = 0;
	for (int i = 0; i < src_Img.cols - 1; i += 1)
	{
		double data = abs(gradient_x[i] - 1.0f / src_Img.cols);
		if (data < minGradData_x)
		{
			minGradData_x = data;
			minPos_x = i;
		}
	}
	double minGradData_y = abs(gradient_y[0] - 1.0f / src_Img.rows);
	int minPos_y = 0;
	for (int i = 0; i < src_Img.rows - 1; i += 1)
	{
		double data = abs(gradient_y[i] - 1.0f / src_Img.rows);
		if (data < minGradData_y)
		{
			minGradData_y = data;
			minPos_y = i;
		}
	}

	double P_x[2] = { minPos_x + 1, S_x[minPos_x] };
	double P_y[2] = { minPos_y + 1, S_y[minPos_y] };
	double Q1[2] = { 0, 0 };
	double Q2[2] = { src_Img.cols, 1 };
	double Q3[2] = { src_Img.rows, 1 };

	double distance_x = abs((Q2[0] - Q1[0])*(P_x[1] - Q1[1]) - (Q2[1] - Q1[1])*(P_x[0] - Q1[0])) / sqrt(pow(Q2[0] - Q1[0], 2) + pow(Q2[1] - Q1[1], 2));
	double distance_y = abs((Q3[0] - Q1[0])*(P_y[1] - Q1[1]) - (Q3[1] - Q1[1])*(P_y[0] - Q1[0])) / sqrt(pow(Q3[0] - Q1[0], 2) + pow(Q3[1] - Q1[1], 2));
	double final_distance = sqrt(pow(distance_x, 2) * pow(distance_y, 2));

	//totaltime = (double)(finish - start) / CLOCKS_PER_SEC;
	//printf("find 1 max :%f\r\n", totaltime);
	return final_distance;
}

void compute_LvValue_FeatureVar(cv::Mat &src_Img, double * lvValue, double *featureVar, int gaussian_kernelSize, float gaussian_kernelSigma)
{
	clock_t start, finish;
	double totaltime;
	cv::Mat gaussianFilter_ImgTemp;
	///////////////////////////0.55s/////////////////////////////////////
	GaussianBlur(src_Img, gaussianFilter_ImgTemp, cv::Size(gaussian_kernelSize, gaussian_kernelSize), gaussian_kernelSigma, gaussian_kernelSigma, cv::BORDER_CONSTANT);//0.55ms
	//cv::Mat gaussianFilter_Img = gaussianFilter_ImgTemp(Range(gaussian_kernelSize / 2, gaussianFilter_ImgTemp.rows - gaussian_kernelSize / 2), Range(gaussian_kernelSize / 2, gaussianFilter_ImgTemp.cols - gaussian_kernelSize / 2));
	/////////////////////////////0.193s///////////////////////////////////////
	cv::Mat gaussianFilter_ImgSqAll = gaussianFilter_ImgTemp.mul(gaussianFilter_ImgTemp);
	cv::Mat gaussianFilter_ImgSq = gaussianFilter_ImgSqAll(cv::Range(gaussian_kernelSize / 2, gaussianFilter_ImgTemp.rows - gaussian_kernelSize / 2), cv::Range(gaussian_kernelSize / 2, gaussianFilter_ImgTemp.cols - gaussian_kernelSize / 2));//加上 上一句 0.193s
	
	//start = clock();
	///////////////////////////0.94s/////////////////////////////////////
	cv::Mat gaussianFilter_ImgSigmaTemp;
	GaussianBlur(src_Img.mul(src_Img), gaussianFilter_ImgSigmaTemp, cv::Size(gaussian_kernelSize, gaussian_kernelSize), gaussian_kernelSigma, gaussian_kernelSigma, cv::BORDER_CONSTANT);
	cv::Mat gaussianFilter_ImgSigma = gaussianFilter_ImgSigmaTemp(cv::Range(gaussian_kernelSize / 2, gaussianFilter_ImgTemp.rows - gaussian_kernelSize / 2), cv::Range(gaussian_kernelSize / 2, gaussianFilter_ImgTemp.cols - gaussian_kernelSize / 2)) - gaussianFilter_ImgSq;
	//////////////////////////////////////////////////////////////////////
	//finish = clock();
	*lvValue = cv::mean(gaussianFilter_ImgSigma)[0];
	

	cv::Mat sigma_Img = gaussianFilter_ImgSigmaTemp - gaussianFilter_ImgSqAll;

	cv::Range rowRange;
	cv::Range colRange;

	*featureVar = compute_FeatureVar(sigma_Img, &rowRange, &colRange); // 0.432s
	cv::Mat subImg = src_Img(rowRange, colRange);
	imwrite("./temp/subImg.bmp", subImg);
	//totaltime = (double)(finish - start) / CLOCKS_PER_SEC;
	//printf("find time:%f\r\n", totaltime);
}

double compute_FeatureVar(cv::Mat &sigmaMat, cv::Range *row_Range, cv::Range *col_Range) // 0.432s
{
	clock_t start, finish;
	double totaltime;
	start = clock();
	////////////////////////////////////////////////////////////////////
	cv::Mat sigma_ImgTemp, sigma_Img;
	int mm = 20, nn = 20;
	sigma_ImgTemp = cv::abs(sigmaMat);
	cv::sqrt(sigma_ImgTemp, sigma_Img);
	cv::Mat sigmaTest = cv::Mat::zeros(cv::Size(mm, nn), CV_64F);
	double maxValue = 0.0;
	int rowIndex = 0, colIndex = 0;
	finish = clock();
	/////////////0.019s////////////////////////////////
	for (int row = 0; row < mm; row++)
	{
		for (int col = 0; col < nn; col++)
		{
			double *rowPtr = (double *)sigmaTest.ptr(row);
			cv::Mat subMat = sigma_Img(cv::Range(row * sigma_Img.rows / mm, (row + 1) * sigma_Img.rows / mm), cv::Range(col * sigma_Img.cols / nn, (col + 1) * sigma_Img.cols / nn));
			rowPtr[col] = cv::mean(subMat)[0];
			if (rowPtr[col] > maxValue)
			{
				maxValue = rowPtr[col];
				rowIndex = row;
				colIndex = col;
			}
		}
	}
	/////////////////////////////////////////////////////
	//finish = clock();
	row_Range->start = rowIndex * sigma_Img.rows / mm;
	row_Range->end = (rowIndex + 1) * sigma_Img.rows / mm;
	col_Range->start = colIndex * sigma_Img.cols / nn;
	col_Range->end = (colIndex + 1) * sigma_Img.cols / nn;
	totaltime = (double)(finish - start) / CLOCKS_PER_SEC;
	//printf("find time:%f\r\n", totaltime);
	return maxValue;
}

