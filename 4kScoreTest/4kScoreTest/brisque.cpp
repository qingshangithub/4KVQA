#include "stdafx.h"
#include "brisque.h"//ComputeBrisqueFeature
#include <time.h>
#include <vector>
#include <cmath>
#include <numeric>



//function definitions
void ComputeBrisqueFeature(IplImage *orig, vector<double>& featurevector)
{
    IplImage *orig_bw_int = cvCreateImage(cvGetSize(orig), orig->depth, 1); 
	if (orig->nChannels != 1) 
	{
		cvCvtColor(orig, orig_bw_int, CV_RGB2GRAY);
	}
	else
	{
		orig_bw_int->imageData = orig->imageData;
	}
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
	cvReleaseImage(&orig_bw);
}

//bool Compute4KFeature(cv::Mat &src_Img, vector<double>& featureVector4K)
//{
//	double feature_SD = 0.0;
//	double lvValue = 0.0;
//	double feature_Var = 0.0;
//	bool is4K = false;
//	clock_t time_s = clock();
//	MyStruct resultFeature;
//	resultFeature = compute_FeatureSD(src_Img);
//	feature_SD = resultFeature.final_distance;
//	//feature_SD = compute_FeatureSD(src_Img);
//	clock_t time1 = clock() - time_s;
//	compute_LvValue_FeatureVar(src_Img, &lvValue, &feature_Var, kernelSize, kernelSigma);
//	if (lvValue <= 40)
//		feature_SD = feature_SD - (0.02 + (50 - lvValue)*0.002);
//	if (feature_SD < 0.00001)
//		feature_SD = 0.12;
//
//	if (feature_SD < 0.115)
//		is4K = true;
//
//	IplImage* orig = cvLoadImage("./temp/subImg.bmp");
//	vector<double> brisqueFeatures;
//	clock_t time_s2 = clock();
//	ComputeBrisqueFeature(orig, brisqueFeatures);
//	clock_t time2 = clock() - time_s2;
//	int featureIndex[9] = { 1, 4, 5, 8, 9, 12, 13, 16, 17 };
//	for (int i = 0; i < 9; i++)
//		featureVector4K.push_back(brisqueFeatures[featureIndex[i]]);
//	featureVector4K.push_back(feature_SD);
//	featureVector4K.push_back(feature_Var);
//	return is4K;
//}

bool Compute4KFeature_fast(cv::Mat &src_Img, vector<double>& featureVector4K, int** data_dct)
{
	double feature_SD = 0.0;
	double lvValue = 0.0;
	double feature_Var = 0.0;
	bool is4K = false;
	clock_t time_s = clock();
	cv::Mat subImg, subImg_1, subImg_2, subImg_3;
	// 计算局部复杂度
	compute_LvValue_FeatureVar_fast(src_Img, &lvValue, &feature_Var, &subImg, &subImg_1, &subImg_2, &subImg_3, kernelSize, kernelSigma);
	// 计算能量
	// cv::Mat subImg = cv::imread("./temp/subImg_1.bmp", 0);
	subImg_1.convertTo(subImg_1, CV_64F, 1.0);
	subImg_2.convertTo(subImg_2, CV_64F, 1.0);
	subImg_3.convertTo(subImg_3, CV_64F, 1.0);
	// src_Img.convertTo(src_Img, CV_64F, 1.0);
	MyStruct resultFeature_1;
	MyStruct resultFeature_2;
	MyStruct resultFeature_3;
	cv::Mat sub_tmp = subImg_2;
	sub_tmp.convertTo(sub_tmp, CV_8U, 1.0);
	resultFeature_1 = compute_FeatureSD(subImg_1, data_dct);
	resultFeature_2 = compute_FeatureSD(subImg_2, data_dct);
	resultFeature_3 = compute_FeatureSD(subImg_3, data_dct);
	feature_SD = (resultFeature_1.final_distance + resultFeature_2.final_distance + resultFeature_3.final_distance)/3.0;
	double h_feature[10];
	double x_feature[4];
	for (int j=0; j<10; j++)
	{
		h_feature[j] = (resultFeature_1.H_dct[j] + resultFeature_2.H_dct[j] + resultFeature_3.H_dct[j]) / 3.0;
	}
	for (int k=0; k<4; k++)
	{
		x_feature[k] = (resultFeature_1.X_dct[k] + resultFeature_2.X_dct[k] + resultFeature_3.X_dct[k]) / 3.0;
	}
	clock_t time1 = clock() - time_s;
	
	//if (lvValue <= 16)
	//	feature_SD = feature_SD - (0.01 + (16 - lvValue)*0.002);
	if (feature_SD < 0.00001)
		feature_SD = 0.05;

	//if (feature_SD < 0.06)
	//	is4K = true;

	// IplImage* orig = cvLoadImage("./temp/subImg.bmp");
	IplImage* orig = &IplImage(subImg);
	IplImage *ori32 = cvCreateImage(cvSize(orig->width, orig->height), IPL_DEPTH_32F, 1);
	cvConvertScale(orig, ori32);
	vector<double> brisqueFeatures;
	clock_t time_s2 = clock();
	ComputeBrisqueFeature(ori32, brisqueFeatures);
	clock_t time2 = clock() - time_s2;
	int featureIndex[9] = { 1, 4, 5, 8, 9, 12, 13, 16, 17 };
	for (int i = 0; i < 9; i++)
		featureVector4K.push_back(brisqueFeatures[featureIndex[i]]);
	featureVector4K.push_back(feature_SD);
	featureVector4K.push_back(feature_Var);
	for (int m=0; m<10; m++)
	{
		featureVector4K.push_back(h_feature[m]);
	}
	for (int n=0; n<4; n++)
	{
		featureVector4K.push_back(x_feature[n]);
	}
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

MyStruct compute_FeatureSD(cv::Mat &src_Img, int** dct_data)
{
	cv::Mat dct_Img;

	double *power_x = new double[src_Img.cols];
	double *power_y = new double[src_Img.rows];
	memset(power_x, 0, src_Img.cols * sizeof(double));
	memset(power_y, 0, src_Img.rows * sizeof(double));

	dct(src_Img, dct_Img);
	double *dct_ImgData = (double *)dct_Img.data;
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

	double max_x = log(1 + power_x[0]), min_x = log(1 + power_x[0]);
	for (int i = 0; i < src_Img.cols; i++)
	{
		power_x[i] = log(1 + power_x[i]);
		if (power_x[i] > max_x)
			max_x = power_x[i];
		if (power_x[i] < min_x)
			min_x = power_x[i];
	}
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

	MyStruct resultFeature;
	resultFeature.final_distance = final_distance;

	//new feature hist yfw
	int N = src_Img.cols;
	cv::Mat impf = src_Img.clone();
	for (int i = 0; i < src_Img.cols; i++)
	{
		for (int j = 0; j < src_Img.rows; j++)
		{
			double *rowPtrHist = (double *)(dct_Img.ptr(j));
			double dataHist = rowPtrHist[i];

			double *rowPtrImpf = (double *)(impf.ptr(j));
			rowPtrImpf[i] = abs(pow(dataHist, 2)) ;
		}
	}

	vector<double> fHist;//P_dct in matlab code
	for (int fi=0;fi<N+1;fi++)
	{
		//这里初始值设为1而不是0是因为0求log会得到负无穷
		double meanSum = 1.0;
		for (int fj=1;fj<= dct_data[fi][0];fj++)
		{
			int indexImpf = dct_data[fi][fj];
			int index1 = indexImpf / N;
			int index2 = indexImpf % N;
			if (index2 == 0)index2 = N;
			double *rowImpf = (double *)(impf.ptr(index2-1));
			int tmpppp = rowImpf[index1];
			meanSum += rowImpf[index1];
		}
		fHist.push_back(log10(meanSum / dct_data[fi][0]));
	}
	fHist.pop_back();
	double totalN=accumulate(fHist.begin(), fHist.end(), 0.0);

	double F1 = round(totalN*0.2);
	double F2 = round(totalN*0.4);
	double F3 = round(totalN*0.6);
	double F4 = round(totalN*0.8);

	for (int w=0;w<10;w++)
	{
		resultFeature.H_dct[w] = 0;
	}
	double tmp = *max_element(fHist.begin(), fHist.end());
	double tmp2 = *min_element(fHist.begin(), fHist.end());
	double maxValue = *max_element(fHist.begin(), fHist.end());
	double minValue = *min_element(fHist.begin(), fHist.end());

	double span = maxValue - minValue;

	for (auto it:fHist)
	{
		double dif = (it - minValue)*10.0;
		int indexHdct = floor( dif / span);
		if (it==maxValue)
		{
			indexHdct=9;
		}
		// int indexHdct = ((it - minValue) * 10) / (maxValue - minValue);
		resultFeature.H_dct[indexHdct]++;
	}

	for (int w = 0; w < 4; w++)
	{
		resultFeature.X_dct[w] = 0;
	}

	double temp = fHist[0];
	for (int i=0;i<N;i++)
	{
		if (temp>F1)
		{
			resultFeature.X_dct[0] = i+1;
			break;
		}
		temp = temp + fHist[i + 1];
	}

	temp = fHist[0];
	for (int i = 0; i < N; i++)
	{
		if (temp > F2)
		{
			resultFeature.X_dct[1] = i+1;
			break;
		}
		temp = temp + fHist[i + 1];
	}

	temp = fHist[0];
	for (int i = 0; i < N; i++)
	{
		if (temp > F3)
		{
			resultFeature.X_dct[2] = i+1;
			break;
		}
		temp = temp + fHist[i + 1];
	}

	temp = fHist[0];
	for (int i = 0; i < N; i++)
	{
		if (temp > F4)
		{
			resultFeature.X_dct[3] = i+1;
			break;
		}
		temp = temp + fHist[i + 1];
	}

	return resultFeature;
}

void compute_LvValue_FeatureVar(cv::Mat &src_Img, double * lvValue, double *featureVar, int gaussian_kernelSize, float gaussian_kernelSigma)
{
	cv::Mat gaussianFilter_ImgTemp;
	GaussianBlur(src_Img, gaussianFilter_ImgTemp, cv::Size(gaussian_kernelSize, gaussian_kernelSize), gaussian_kernelSigma, gaussian_kernelSigma, cv::BORDER_CONSTANT);
	//cv::Mat gaussianFilter_Img = gaussianFilter_ImgTemp(Range(gaussian_kernelSize / 2, gaussianFilter_ImgTemp.rows - gaussian_kernelSize / 2), Range(gaussian_kernelSize / 2, gaussianFilter_ImgTemp.cols - gaussian_kernelSize / 2));
	cv::Mat gaussianFilter_ImgSqAll = gaussianFilter_ImgTemp.mul(gaussianFilter_ImgTemp);
	cv::Mat gaussianFilter_ImgSq = gaussianFilter_ImgSqAll(cv::Range(gaussian_kernelSize / 2, gaussianFilter_ImgTemp.rows - gaussian_kernelSize / 2), cv::Range(gaussian_kernelSize / 2, gaussianFilter_ImgTemp.cols - gaussian_kernelSize / 2));


	cv::Mat gaussianFilter_ImgSigmaTemp;
	GaussianBlur(src_Img.mul(src_Img), gaussianFilter_ImgSigmaTemp, cv::Size(gaussian_kernelSize, gaussian_kernelSize), gaussian_kernelSigma, gaussian_kernelSigma, cv::BORDER_CONSTANT);
	cv::Mat gaussianFilter_ImgSigma = gaussianFilter_ImgSigmaTemp(cv::Range(gaussian_kernelSize / 2, gaussianFilter_ImgTemp.rows - gaussian_kernelSize / 2), cv::Range(gaussian_kernelSize / 2, gaussianFilter_ImgTemp.cols - gaussian_kernelSize / 2)) - gaussianFilter_ImgSq;
	*lvValue = cv::mean(gaussianFilter_ImgSigma)[0];

	cv::Mat sigma_Img = gaussianFilter_ImgSigmaTemp - gaussianFilter_ImgSqAll;

	cv::Range rowRange;
	cv::Range colRange;

	*featureVar = compute_FeatureVar(sigma_Img, &rowRange, &colRange);
	cv::Mat subImg = src_Img(rowRange, colRange);
	imwrite("./temp/subImg.bmp", subImg);

}

double compute_FeatureVar(cv::Mat &sigmaMat, cv::Range *row_Range, cv::Range *col_Range)
{
	cv::Mat sigma_ImgTemp, sigma_Img;
	int mm = 20, nn = 20;
	sigma_ImgTemp = cv::abs(sigmaMat);
	cv::Mat sigma_ImgTemp2;
	sigma_ImgTemp.convertTo(sigma_ImgTemp2,CV_32F);
	cv::sqrt(sigma_ImgTemp2, sigma_Img);
	sigma_Img.convertTo(sigma_Img, CV_8U);

	cv::Mat sigmaTest = cv::Mat::zeros(cv::Size(mm, nn), CV_64F);
	double maxValue = 0.0;
	int rowIndex = 0, colIndex = 0;
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
	row_Range->start = rowIndex * sigma_Img.rows / mm;
	row_Range->end = (rowIndex + 1) * sigma_Img.rows / mm;
	col_Range->start = colIndex * sigma_Img.cols / nn;
	col_Range->end = (colIndex + 1) * sigma_Img.cols / nn;

	return maxValue;
}

void compute_LvValue_FeatureVar_fast(cv::Mat &src_Img, double * lvValue, double *featureVar, cv::Mat *subImg, cv::Mat *subImg_1, cv::Mat *subImg_2, cv::Mat *subImg_3, int gaussian_kernelSize, float gaussian_kernelSigma)
{
	// 高斯滤波
	cv::Mat gaussianFilter_ImgTemp;
	GaussianBlur(src_Img, gaussianFilter_ImgTemp, cv::Size(gaussian_kernelSize, gaussian_kernelSize), gaussian_kernelSigma, gaussian_kernelSigma, cv::BORDER_CONSTANT);
	cv::Mat gaussianFilter_ImgSqAll = gaussianFilter_ImgTemp.mul(gaussianFilter_ImgTemp);
	cv::Mat gaussianFilter_ImgSigmaTemp;
	GaussianBlur(src_Img.mul(src_Img), gaussianFilter_ImgSigmaTemp, cv::Size(gaussian_kernelSize, gaussian_kernelSize), gaussian_kernelSigma, gaussian_kernelSigma, cv::BORDER_CONSTANT);
	cv::Mat sigma_Img = gaussianFilter_ImgSigmaTemp - gaussianFilter_ImgSqAll;
	cv::Mat sigma_absImg = cv::abs(sigma_Img);
	cv::Mat sigma_sqrtImg;
	cout << endl;
	//cout << int(sigma_absImg.at<uchar>(0,0)) << endl;
	cv::Mat sigma_absImg2;
	sigma_absImg.convertTo(sigma_absImg2, CV_32F);
	cv::sqrt(sigma_absImg2, sigma_sqrtImg);
	sigma_sqrtImg.convertTo(sigma_sqrtImg, CV_8U);

	// 计算feature var
	cv::Range row_tmp;
	cv::Range col_tmp;
	// *featureVar = compute_FeatureVar(sigma_Img, &row_tmp, &col_tmp);
	// 计算子图局部复杂性
	int mm = 9, nn = 16;
	int row,col;
	double score[9][16];
	for (row = 0; row < mm; row++)
	{
		for (col = 0; col < nn; col++)
		{
			score[row][col] = 0.0;
		}
	}
	double maxValue = 0.0;
	int rowIndex = 0, colIndex = 0;
	for (row = 0; row < mm; row++)
	{
		for (col = 0; col < nn; col++)
		{
			cv::Mat subMat = sigma_sqrtImg(cv::Range(row * sigma_sqrtImg.rows / mm, (row + 1) * sigma_sqrtImg.rows / mm), cv::Range(col * sigma_sqrtImg.cols / nn, (col + 1) * sigma_sqrtImg.cols / nn));
			score[row][col] = cv::mean(subMat)[0];
			if (score[row][col]>maxValue)
			{
				maxValue = score[row][col];
				rowIndex = row;
				colIndex = col;
			}
		}
	}
	*featureVar = maxValue;
	cv::Range row_Range;
	cv::Range col_Range;
	row_Range.start = rowIndex * sigma_sqrtImg.rows / mm;
	row_Range.end = (rowIndex + 1) * sigma_sqrtImg.rows / mm;
	col_Range.start = colIndex * sigma_sqrtImg.cols / nn;
	col_Range.end = (colIndex + 1) * sigma_sqrtImg.cols / nn;
	*subImg = src_Img(row_Range, col_Range);

	cv::Mat sub = src_Img(row_Range, col_Range);
	sub.convertTo(sub, CV_8U, 1.0);
    // imwrite("./temp/subImg.bmp", sub);
	// 取中间
	for (row = 0; row < 1; row++)
	{
		for (col = 0; col < nn; col++)
		{
			score[row][col] = 0;
		}
	}
	for (row = mm-1; row < mm; row++)
	{
		for (col = 0; col < nn; col++)
		{
			score[row][col] = 0;
		}
	}
	for (row = 0; row < mm; row++)
	{
		for (col = 0; col < 2; col++)
		{
			score[row][col] = 0;
		}
	}
	for (row = 0; row < mm; row++)
	{
		for (col = nn-2; col < nn; col++)
		{
			score[row][col] = 0;
		}
	}
	// 取最大的三块
	double S1, S2, S3;
	S1 = 0.0;
	S2 = 0.0;
	S3 = 0.0;
	// 1-最大
	maxValue = 0.0;
	rowIndex = 0;
	colIndex = 0;
	for (row = 0; row < mm; row++)
	{
		for (col = 0; col < nn; col++)
		{
			if (score[row][col]>maxValue)
			{
				maxValue = score[row][col];
				rowIndex = row;
				colIndex = col;
			}
		}
	}
	S1 = score[rowIndex][colIndex];
	score[rowIndex][colIndex] = 0.0;
	cv::Range row_Range_1;
	cv::Range col_Range_1;
	row_Range_1.start = rowIndex * sigma_sqrtImg.rows / mm;
	row_Range_1.end = (rowIndex + 1) * sigma_sqrtImg.rows / mm;
	col_Range_1.start = colIndex * sigma_sqrtImg.cols / nn;
	col_Range_1.end = (colIndex + 1) * sigma_sqrtImg.cols / nn;
	*subImg_1 = src_Img(row_Range_1, col_Range_1);

	cv::Mat sub_1 = src_Img(row_Range_1, col_Range_1);
	sub_1.convertTo(sub_1, CV_8U, 1.0);
	// imwrite("./temp/subImg_1.bmp", subImg_1);
	// 2-第二大
	maxValue = 0.0;
	rowIndex = 0;
	colIndex = 0;
	for (row = 0; row < mm; row++)
	{
		for (col = 0; col < nn; col++)
		{
			if (score[row][col]>maxValue)
			{
				maxValue = score[row][col];
				rowIndex = row;
				colIndex = col;
			}
		}
	}
	S2 = score[rowIndex][colIndex];
	score[rowIndex][colIndex] = 0.0;
	cv::Range row_Range_2;
	cv::Range col_Range_2;
	row_Range_2.start = rowIndex * sigma_sqrtImg.rows / mm;
	row_Range_2.end = (rowIndex + 1) * sigma_sqrtImg.rows / mm;
	col_Range_2.start = colIndex * sigma_sqrtImg.cols / nn;
	col_Range_2.end = (colIndex + 1) * sigma_sqrtImg.cols / nn;
	*subImg_2 = src_Img(row_Range_2, col_Range_2);

	cv::Mat sub_2 = src_Img(row_Range_2, col_Range_2);
	sub_2.convertTo(sub_2, CV_8U, 1.0);
	// imwrite("./temp/subImg_2.bmp", subImg_2);
	// 3-第三大
	maxValue = 0.0;
	rowIndex = 0;
	colIndex = 0;
	for (row = 0; row < mm; row++)
	{
		for (col = 0; col < nn; col++)
		{
			if (score[row][col]>maxValue)
			{
				maxValue = score[row][col];
				rowIndex = row;
				colIndex = col;
			}
		}
	}
	S3 = score[rowIndex][colIndex];
	score[rowIndex][colIndex] = 0.0;
	cv::Range row_Range_3;
	cv::Range col_Range_3;
	row_Range_3.start = rowIndex * sigma_sqrtImg.rows / mm;
	row_Range_3.end = (rowIndex + 1) * sigma_sqrtImg.rows / mm;
	col_Range_3.start = colIndex * sigma_sqrtImg.cols / nn;
	col_Range_3.end = (colIndex + 1) * sigma_sqrtImg.cols / nn;
	*subImg_3 = src_Img(row_Range_3, col_Range_3);

	cv::Mat sub_3 = src_Img(row_Range_3, col_Range_3);
	sub_3.convertTo(sub_3, CV_8U, 1.0);
	// imwrite("./temp/subImg_3.bmp", subImg_3);

	*lvValue = (S1 + S2 + S3) / 3;
	/*
	cv::Mat gaussianFilter_ImgTemp;
	GaussianBlur(src_Img, gaussianFilter_ImgTemp, cv::Size(gaussian_kernelSize, gaussian_kernelSize), gaussian_kernelSigma, gaussian_kernelSigma, cv::BORDER_CONSTANT);
	//cv::Mat gaussianFilter_Img = gaussianFilter_ImgTemp(Range(gaussian_kernelSize / 2, gaussianFilter_ImgTemp.rows - gaussian_kernelSize / 2), Range(gaussian_kernelSize / 2, gaussianFilter_ImgTemp.cols - gaussian_kernelSize / 2));
	cv::Mat gaussianFilter_ImgSqAll = gaussianFilter_ImgTemp.mul(gaussianFilter_ImgTemp);
	cv::Mat gaussianFilter_ImgSq = gaussianFilter_ImgSqAll(cv::Range(gaussian_kernelSize / 2, gaussianFilter_ImgTemp.rows - gaussian_kernelSize / 2), cv::Range(gaussian_kernelSize / 2, gaussianFilter_ImgTemp.cols - gaussian_kernelSize / 2));


	cv::Mat gaussianFilter_ImgSigmaTemp;
	GaussianBlur(src_Img.mul(src_Img), gaussianFilter_ImgSigmaTemp, cv::Size(gaussian_kernelSize, gaussian_kernelSize), gaussian_kernelSigma, gaussian_kernelSigma, cv::BORDER_CONSTANT);
	cv::Mat gaussianFilter_ImgSigma = gaussianFilter_ImgSigmaTemp(cv::Range(gaussian_kernelSize / 2, gaussianFilter_ImgTemp.rows - gaussian_kernelSize / 2), cv::Range(gaussian_kernelSize / 2, gaussianFilter_ImgTemp.cols - gaussian_kernelSize / 2)) - gaussianFilter_ImgSq;
	*lvValue = cv::mean(gaussianFilter_ImgSigma)[0];

	cv::Mat sigma_Img = gaussianFilter_ImgSigmaTemp - gaussianFilter_ImgSqAll;

	cv::Range rowRange;
	cv::Range colRange;

	*featureVar = compute_FeatureVar(sigma_Img, &rowRange, &colRange);
	cv::Mat subImg = src_Img(rowRange, colRange);
	imwrite("./temp/subImg.bmp", subImg);
	*/
}