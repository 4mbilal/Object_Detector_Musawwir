/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/
#include "HSG_HIK.h"
#include "NMS.h"
#include "HSG004_HIK_LUT.h"		//{ 54, 18 }  9 bin
#include "HSG005_HIK_LUT.h"		//{ 78, 30 } 9 bin
#include "HSG006_HIK_LUT.h"		//{ 90, 36 }  9 bin	
#include "HSG007_HIK_LUT.h"		//{ 102, 36 } 9 bin

//										HSG Parameters
int HSG_Descriptor_WinSize[2] = { 102, 36 };//{ 126, 62 };//{ 78, 30 };//{ 90, 36 };// { 54, 18 }; // 
const int HSG_Hist_Cell_Size[2] = { 6, 6 };
const int HSG_Hist_Bins = 9;
const int HSG_Aux_Bins = 1;
const int tot_bins = (HSG_Hist_Bins + HSG_Aux_Bins);
const int HSG_hist_stride[2] = { 3, 3 };
const int HSG_GradMag_Q = 16;										//Gradient Magnitude quantization factor
float SVM_Score_Th = 0;
float svm_score_offset = 0;
float min_object_height = 102;//51;//	78;// 90;//54;// 
float Scale_Step = pow(2, (1 / 8.f));
unsigned char Frame_Padding[4] = { 8, 8, 8, 8 }; // 	{ 0, 0, 0, 0 }; // { 24, 24, 24, 24 };// { 32, 32, 32, 32 }; // { 64, 64, 64, 64 };// 				//Padding {Top, Bottom, Left, Right} at each scale to detect objects towards the edges
unsigned char Detect_Win_Stride_Offset[2] = { 1, 1 };	//Each step is a multiple of HSG_Hist_Stride
enum nms_method_types { GroupRectangles_OCV, MeanShift_OCV, Custom_NMS, None };
nms_method_types nms_method = Custom_NMS;//     MeanShift_OCV;//   		GroupRectangles_OCV;//		None;// 
	/*
	const char* svm_file_name = "SVM_Data\\HSG000.svm";//{ 54, 18 }
	double Soft_SVM_C = 0.00022;
	int feature_size = 595;

	const char* svm_file_name = "SVM_Data\\HSG001.svm";//{ 78, 30 }
	double Soft_SVM_C = 0.00015;
	int feature_size = 1575;

	const char* svm_file_name = "SVM_Data\\HSG002.svm";//{ 90, 36 }
	double Soft_SVM_C = 0.00005;
	int feature_size = 2233;

	const char* svm_file_name = "SVM_Data\\HSG003.svm";//{ 102, 36 }
	double Soft_SVM_C = 0.00006;
	int feature_size = 2541; //3630;

	const char* svm_file_name = "SVM_Data\\HSG004.svm";//{ 54, 18 }
	double Soft_SVM_C = 306e-6;
	int feature_size = 850;

	const char* svm_file_name = "SVM_Data\\HSG005.svm";//{ 78, 30 }
	double Soft_SVM_C = 50e-6;
	int feature_size = 2250;

	const char* svm_file_name = "SVM_Data\\HSG006.svm";//{ 90, 36 }
	double C_Neg = 175e-6;
	double C_Pos = 47e-6;
	double Soft_SVM_C = C_Neg;
	double Soft_SVM_C_ratio = C_Pos / C_Neg;
	int feature_size = 3190;

	const char* svm_file_name = "SVM_Data\\HSG007.svm";//{ 102, 36 }
	double C_Pos = 40e-6;
	double C_Neg = 48e-6;
	double Soft_SVM_C = C_Neg;
	double Soft_SVM_C_ratio = C_Pos / C_Neg;
	int feature_size = 3630;

	*/
//#define Orient
#ifdef Orient
const float HSG_bin_size = 2 * pi / HSG_Hist_Bins;					//Orientation (0-360)
#else
const float HSG_bin_size = CV_PI / HSG_Hist_Bins;						//No Orientation (0-180)
#endif

#ifndef Training_Mode
template <size_t xx, size_t yy>
void HSG_Window_Scan(Mat& frame, unsigned char* frame_Hists, int rows, int cols, vector<Rect>& BB_Rects, vector<double>& BB_Scores, double scale, vector<double>& BB_Scales, bool training_mode_local, int* WinSize, float(&HIK_LUT)[xx][yy], float hik_bias, float padding_scale, int* cascade_indices, float* cascade_stage_th, int cascade_stage_cnt);
#else
void HSG_Window_Scan(Mat& frame, unsigned char* frame_Hists, int rows, int cols, vector<Rect>& BB_Rects, vector<double>& BB_Scores, double scale, vector<double>& BB_Scales, bool training_mode_local, int* WinSize, double** HIK_LUT, double hik_bias, float padding_scale);
#endif
void HSG_SingleScale_Detector_Approx(Mat& frame, unsigned char* frame_Hists, int cols, int rows, vector<Rect>& BB_Rects, vector<double>& BB_Scores, double scale, vector<double>& BB_Scales);
//int cascade_indices[726] = { 4, 7, 4, 6, 4, 3, 30, 7, 31, 4, 1, 6, 0, 5, 30, 3, 12, 0, 5, 8, 31, 3, 17, 9, 5, 2, 1, 4, 7, 0, 8, 0, 7, 10, 31, 5, 29, 7, 30, 8, 1, 5, 1, 7, 31, 6, 3, 6, 30, 5, 11, 0, 4, 4, 29, 3, 5, 7, 30, 4, 5, 3, 30, 6, 29, 5, 3, 7, 3, 3, 10, 10, 9, 0, 3, 5, 18, 9, 31, 7, 11, 10, 3, 4, 1, 3, 6, 10, 22, 5, 12, 1, 2, 7, 2, 4, 4, 5, 10, 0, 17, 1, 14, 1, 2, 3, 2, 6, 9, 10, 12, 10, 8, 10, 24, 8, 29, 8, 11, 9, 31, 8, 0, 6, 29, 6, 21, 5, 29, 2, 24, 7, 12, 2, 4, 8, 15, 1, 23, 7, 23, 5, 20, 2, 29, 4, 16, 9, 10, 9, 11, 8, 22, 9, 25, 8, 6, 1, 12, 8, 13, 1, 23, 4, 20, 5, 30, 2, 25, 9, 17, 8, 13, 0, 15, 9, 0, 4, 7, 1, 18, 8, 11, 1, 16, 1, 12, 9, 2, 5, 2, 0, 26, 8, 24, 1, 9, 1, 4, 2, 6, 0, 5, 4, 3, 0, 18, 1, 24, 3, 28, 5, 13, 6, 31, 2, 14, 9, 19, 8, 15, 0, 24, 9, 14, 5, 5, 10, 25, 3, 23, 8, 11, 2, 16, 0, 28, 2, 10, 1, 24, 2, 20, 1, 23, 1, 3, 10, 21, 6, 4, 10, 14, 0, 23, 2, 13, 9, 20, 8, 12, 3, 14, 2, 21, 9, 13, 5, 2, 8, 15, 2, 24, 5, 6, 9, 19, 1, 8, 1, 5, 6, 22, 8, 23, 3, 26, 2, 17, 10, 26, 3, 28, 7, 23, 6, 14, 6, 27, 7, 1, 8, 22, 6, 27, 2, 5, 1, 27, 8, 25, 5, 28, 3, 19, 9, 21, 8, 28, 6, 16, 5, 11, 7, 25, 7, 21, 2, 11, 3, 16, 8, 22, 1, 24, 4, 7, 9, 31, 9, 25, 4, 1, 2, 22, 4, 27, 3, 15, 5, 15, 8, 26, 7, 23, 9, 20, 6, 20, 3, 28, 8, 5, 5, 30, 9, 27, 6, 21, 1, 25, 2, 26, 9, 12, 7, 10, 7, 22, 7, 6, 2, 3, 8, 25, 1, 15, 10, 17, 5, 19, 5, 17, 0, 19, 2, 13, 8, 15, 6, 20, 7, 1, 0, 10, 8, 29, 1, 5, 0, 31, 1, 8, 9, 9, 9, 10, 6, 10, 4, 24, 6, 13, 10, 21, 7, 28, 9, 20, 9, 5, 9, 28, 4, 17, 2, 27, 4, 13, 2, 9, 2, 30, 1, 14, 10, 27, 9, 21, 3, 28, 1, 29, 9, 10, 2, 21, 4, 6, 5, 16, 4, 24, 0, 18, 10, 16, 10, 10, 5, 19, 0, 27, 5, 2, 2, 27, 1, 0, 10, 13, 4, 13, 7, 20, 4, 23, 10, 1, 10, 25, 6, 20, 0, 26, 1, 26, 4, 24, 10, 9, 7, 14, 8, 22, 2, 9, 5, 12, 6, 3, 2, 8, 7, 2, 10, 19, 7, 6, 8, 16, 2, 9, 6, 14, 4, 18, 2, 26, 6, 12, 5, 7, 5, 0, 7, 25, 0, 9, 4, 16, 6, 4, 0, 25, 10, 1, 1, 11, 4, 10, 3, 26, 5, 8, 4, 8, 2, 8, 8, 7, 6, 17, 6, 15, 4, 12, 4, 23, 0, 2, 1, 13, 3, 15, 7, 11, 6, 19, 6, 26, 0, 22, 10, 14, 7, 11, 5, 18, 7, 7, 8, 17, 4, 0, 0, 22, 3, 15, 3, 18, 0, 8, 6, 8, 3, 16, 7, 3, 1, 18, 6, 17, 7, 18, 5, 4, 1, 19, 3, 9, 8, 14, 3, 19, 4, 32, 5, 1, 9, 16, 3, 4, 9, 7, 4, 22, 0, 0, 3, 7, 7, 17, 3, 7, 2, 9, 3, 27, 0, 19, 10, 21, 0, 20, 10, 0, 9, 3, 9, 2, 9, 8, 5, 6, 4, 0, 8, 27, 10, 32, 6, 0, 1, 6, 6, 6, 3, 28, 10, 29, 0, 32, 4, 30, 0, 6, 7, 21, 10, 18, 3, 26, 10, 28, 0, 18, 4, 0, 2, 31, 0, 32, 8, 7, 3, 31, 10, 32, 7, 32, 9, 29, 10, 32, 2, 32, 1, 32, 3, 30, 10, 32, 10, 32, 0 };
//int cascade_indices[726] = { 32, 0, 32, 10, 0, 5, 1, 5, 5, 2, 32, 7, 4, 7, 0, 6, 5, 8, 32, 3, 4, 3, 32, 1, 4, 6, 32, 4, 30, 10, 0, 4, 32, 2, 32, 8, 5, 1, 1, 4, 0, 0, 7, 10, 5, 7, 6, 1, 1, 6, 8, 10, 29, 10, 5, 3, 2, 0, 32, 9, 0, 2, 6, 2, 8, 0, 6, 8, 0, 7, 32, 6, 2, 1, 31, 10, 3, 3, 6, 4, 1, 7, 0, 8, 0, 10, 7, 3, 3, 2, 31, 0, 6, 3, 18, 3, 2, 9, 7, 8, 3, 10, 0, 1, 4, 2, 3, 6, 2, 10, 9, 10, 6, 7, 6, 6, 6, 9, 26, 10, 0, 3, 18, 1, 7, 9, 7, 0, 9, 0, 17, 1, 31, 4, 18, 4, 30, 7, 5, 6, 27, 0, 28, 0, 7, 2, 19, 10, 15, 7, 3, 9, 22, 4, 27, 10, 3, 7, 20, 4, 1, 10, 14, 8, 19, 0, 5, 9, 30, 3, 31, 3, 14, 2, 21, 10, 19, 8, 22, 0, 17, 9, 30, 0, 21, 2, 1, 0, 12, 0, 0, 9, 3, 0, 3, 1, 29, 0, 15, 10, 4, 5, 7, 1, 4, 4, 31, 5, 1, 9, 17, 2, 19, 2, 20, 0, 29, 7, 28, 10, 13, 8, 1, 1, 23, 4, 8, 9, 30, 5, 20, 2, 15, 8, 30, 8, 14, 3, 27, 5, 19, 3, 2, 3, 21, 0, 7, 4, 30, 4, 26, 0, 21, 4, 22, 3, 1, 3, 24, 6, 17, 3, 13, 2, 15, 4, 31, 6, 8, 5, 2, 6, 20, 8, 32, 5, 6, 10, 12, 1, 20, 10, 14, 10, 27, 3, 11, 10, 8, 1, 8, 6, 15, 3, 11, 9, 14, 7, 6, 0, 22, 7, 22, 9, 11, 0, 18, 0, 12, 8, 28, 6, 14, 9, 11, 2, 10, 10, 21, 6, 5, 5, 15, 5, 28, 7, 27, 4, 9, 3, 1, 8, 29, 3, 16, 5, 16, 6, 7, 7, 21, 3, 28, 5, 29, 2, 18, 9, 25, 1, 3, 4, 10, 0, 2, 2, 12, 2, 21, 5, 16, 10, 4, 10, 23, 2, 30, 6, 19, 4, 18, 2, 4, 1, 17, 4, 21, 7, 28, 4, 4, 9, 4, 0, 18, 5, 20, 6, 22, 1, 18, 8, 23, 0, 11, 1, 29, 5, 31, 7, 27, 6, 22, 5, 12, 10, 15, 0, 1, 2, 23, 7, 15, 2, 16, 4, 10, 9, 26, 6, 24, 7, 24, 8, 5, 10, 31, 8, 25, 2, 2, 7, 18, 10, 3, 5, 3, 8, 17, 7, 15, 1, 23, 1, 26, 3, 13, 0, 12, 4, 22, 6, 19, 6, 19, 9, 5, 4, 13, 9, 15, 6, 28, 8, 7, 6, 20, 3, 11, 8, 16, 2, 2, 4, 23, 10, 29, 6, 14, 4, 19, 7, 18, 7, 22, 8, 27, 1, 16, 3, 23, 8, 29, 1, 19, 1, 23, 6, 13, 1, 11, 5, 12, 9, 9, 8, 8, 2, 22, 2, 15, 9, 24, 5, 4, 8, 5, 0, 29, 8, 16, 7, 8, 8, 26, 5, 17, 6, 16, 1, 28, 3, 21, 8, 24, 10, 10, 2, 18, 6, 25, 10, 12, 6, 13, 10, 30, 2, 30, 9, 17, 5, 8, 3, 10, 1, 17, 0, 11, 4, 25, 7, 25, 6, 16, 8, 6, 5, 26, 4, 13, 5, 26, 7, 14, 1, 29, 4, 13, 3, 12, 3, 25, 0, 9, 6, 11, 6, 24, 4, 20, 9, 26, 1, 13, 4, 25, 8, 7, 5, 20, 7, 14, 6, 23, 3, 17, 8, 22, 10, 26, 8, 20, 1, 10, 3, 30, 1, 16, 9, 11, 3, 12, 5, 17, 10, 27, 9, 9, 4, 25, 3, 26, 9, 9, 9, 26, 2, 24, 1, 23, 5, 21, 9, 24, 2, 2, 8, 31, 1, 24, 3, 12, 7, 27, 7, 21, 1, 9, 5, 23, 9, 8, 4, 9, 1, 19, 5, 31, 2, 27, 8, 27, 2, 25, 4, 10, 8, 25, 5, 14, 0, 13, 6, 20, 5, 29, 9, 16, 0, 8, 7, 28, 2, 14, 5, 24, 0, 28, 9, 28, 1, 9, 7, 13, 7, 24, 9, 10, 6, 25, 9, 10, 4, 10, 5, 2, 5, 10, 7, 31, 9, 9, 2, 11, 7 };

unsigned char r0 = 1, c0 = 1;											//Set boundary for rows and columns (Start processing from these values instead of 0,0)

//---------------------------------------------------------------------------------------------------------
void HSG_SingleScale_Detector(Mat& frame, vector<Rect>& BB_Rects, vector<double>& BB_Scores, double scale, vector<double>& BB_Scales) {
	//	ftime(&t_start);
	//---------------------------------------------------------------------------------------------------------
	//				Insert border around the frame to detect partially occluded objects on the sides
	//---------------------------------------------------------------------------------------------------------
	copyMakeBorder(frame, frame, Frame_Padding[0], Frame_Padding[1], Frame_Padding[2], Frame_Padding[3], BORDER_CONSTANT, Scalar(255, 255, 255));
	int rows = frame.rows;
	int cols = frame.cols;
	int no_horiz_hists = ((cols - 2) / HSG_hist_stride[1]);
	int no_vert_hists = ((rows - 2) / HSG_hist_stride[0]);
	//---------------------------------------------------------------------------------------------------------
	//									Display frame info: row, col etc.
	//---------------------------------------------------------------------------------------------------------
/*	cout << "\nRows = " << rows;	cout << ", Cols = " << cols;	cout << ", horiz hists = " << no_horiz_hists;	cout << ", vert hists = " << no_vert_hists;	cout << "\nscale = " << scale;	getchar();*/
	//---------------------------------------------------------------------------------------------------------
	//									Memory Allocation & Initialization
	//---------------------------------------------------------------------------------------------------------
	unsigned char* frame_Hists;
	int mem_siz = (no_vert_hists)* (no_horiz_hists)*(HSG_Hist_Bins + HSG_Aux_Bins);
	frame_Hists = new unsigned char[mem_siz];			//No. of bins
	for (int i = 0; i < mem_siz; i++)
		frame_Hists[i] = 0;		//initialize hists to zero*/
//	ftime(&t_end);
	//---------------------------------------------------------------------------------------------------------
	//									Calculate Features (Histograms + Color info) for the whole image
	//---------------------------------------------------------------------------------------------------------
//	ftime(&t_start);
	HSG_Hist(frame, frame_Hists);
	//	HSG_Feature(frame, frame_Hists);
	//	HSG_Feature_V2(frame, frame_Hists);
	//	HSG_Feature_V3(frame, frame_Hists);
//	ftime(&t_end);
	//---------------------------------------------------------------------------------------------------------
	//											Scanning Window
	//---------------------------------------------------------------------------------------------------------
#ifdef Training_Mode
	HSG_Window_Scan(frame, frame_Hists, rows, cols, BB_Rects, BB_Scores, scale, BB_Scales, training_mode, HSG_Descriptor_WinSize, Kernel_LUT, b, 1);
#else
//ftime(&t_start);
		if (scale>0.6)
		HSG_Window_Scan(frame, frame_Hists, rows, cols, BB_Rects, BB_Scores, scale, BB_Scales, training_mode, WinSize_4, Kernel_LUT_4, b_4, 1, cascade_indices_4, cascade_stage_th_4, 5);
		HSG_Window_Scan(frame, frame_Hists, rows, cols, BB_Rects, BB_Scores, scale, BB_Scales, training_mode, WinSize_5, Kernel_LUT_5, b_5, 1, cascade_indices_5, cascade_stage_th_5, 6);
		HSG_Window_Scan(frame, frame_Hists, rows, cols, BB_Rects, BB_Scores, scale, BB_Scales, training_mode, WinSize_6, Kernel_LUT_6, b_6, 1, cascade_indices_6, cascade_stage_th_6, 6);
		HSG_Window_Scan(frame, frame_Hists, rows, cols, BB_Rects, BB_Scores, scale, BB_Scales, training_mode, WinSize_7, Kernel_LUT_7, b_7, 1, cascade_indices_7, cascade_stage_th_7, 6);
	//ftime(&t_end);
//	if ((floor(rows*scale) > (HSG_Descriptor_WinSize[0] + 2)) && floor(cols*scale) > (HSG_Descriptor_WinSize[1] + 2)) {
		//if (scale<=1 && scale > 0.5){
//	if((abs(scale-1)<0.001))
//		if (((scale<1.001) && (scale>0.4999))) {
//		ftime(&t_start);
//		HSG_SingleScale_Detector_Approx(frame, frame_Hists, cols, rows, BB_Rects, BB_Scores, scale, BB_Scales);
//		ftime(&t_end);
//			printf("\n%.3f", scale);
//		}
#endif
	//---------------------------------------------------------------------------------------------------------
	//											Release Memory
	//---------------------------------------------------------------------------------------------------------
//	ftime(&t_start);
	delete frame_Hists;
//	ftime(&t_end);
}

#ifndef Training_Mode
template <size_t xx, size_t yy>
void HSG_Window_Scan(Mat& frame, unsigned char* frame_Hists, int rows, int cols, vector<Rect>& BB_Rects, vector<double>& BB_Scores, double scale, vector<double>& BB_Scales, bool training_mode_local, int* WinSize, float(&HIK_LUT)[xx][yy], float hik_bias, float padding_scale, int* cascade_indices, float* cascade_stage_th, int cascade_stage_cnt) {
#else
void HSG_Window_Scan(Mat& frame, unsigned char* frame_Hists, int rows, int cols, vector<Rect>& BB_Rects, vector<double>& BB_Scores, double scale, vector<double>& BB_Scales, bool training_mode_local, int* WinSize, double** HIK_LUT, double hik_bias, float padding_scale) {
#endif
	//	ftime(&t_start);
	int no_horiz_hists = ((cols - 2) / HSG_hist_stride[1]);
	int no_vert_hists = ((rows - 2) / HSG_hist_stride[0]);
	int hi_max = (WinSize[0] / HSG_hist_stride[0]) - 1;
	int hj_max = (WinSize[1] / HSG_hist_stride[1]) - 1;
	int hij_max = (hj_max)*(hi_max);
	int hjk_max = hj_max*tot_bins;
	int horiz_elements = no_horiz_hists*tot_bins;//2180 for 640x480 frame

	int mem_siz = (HSG_Hist_Bins + HSG_Aux_Bins)*(no_vert_hists)*(no_horiz_hists);
	for (int x = 0; x < mem_siz; x++) {
		int Q_Val = frame_Hists[x];
		if (Q_Val >(Kernel_LUT_Q - 1))
			frame_Hists[x] = Kernel_LUT_Q - 1;
	}

	Mutex mtx;
	int wind_cnt = 0, oper_cnt = 0;
	int nwindows = (no_vert_hists - hi_max)*(no_horiz_hists - hj_max);
	//	printf("\nNo. of windows = %d", nwindows); getchar();

	//Scan 27117 Windows for 640x480 frame (HSG_007)
	int nwindowsX = no_horiz_hists - hj_max;
#ifdef Use_P4
	//	parallel_for(size_t(0), size_t(nwindows), size_t(1), [&](size_t n) {
	//	if ((n % Detect_Win_Stride_Offset[1]) != 0) return;
	//	if (((n / nwindowsX) % Detect_Win_Stride_Offset[0]) != 0) return;
	//		int v_adr_i = n / nwindowsX;
	//		int h_adr_i = n - nwindowsX*v_adr_i;
	parallel_for(size_t(0), size_t(no_vert_hists - hi_max), size_t(Detect_Win_Stride_Offset[0]), [&](size_t v_adr_i) {
		unsigned char* frame_Hists_row_ptr = frame_Hists + v_adr_i*horiz_elements;
		parallel_for(size_t(0), size_t(no_horiz_hists - hj_max), size_t(Detect_Win_Stride_Offset[1]), [&](size_t h_adr_i) {
#else
	//	for (int n = 0; n < nwindows; n++) {//Inspired from HOG OCV code but slightly slower
	//		if ((n % Detect_Win_Stride_Offset[1]) != 0) continue;
	//		if (((n/nwindowsX) % Detect_Win_Stride_Offset[0]) != 0) continue;
	//		int v_adr_i = n / nwindowsX;
	//		int h_adr_i = n - nwindowsX*v_adr_i;
	for (int v_adr_i = 0; v_adr_i < no_vert_hists - hi_max; v_adr_i += Detect_Win_Stride_Offset[0]) {
		unsigned char* frame_Hists_row_ptr = frame_Hists + v_adr_i*horiz_elements;
		for (int h_adr_i = 0; h_adr_i < no_horiz_hists - hj_max; h_adr_i += Detect_Win_Stride_Offset[1]) {
#endif
			
			float score = 0;
			int feature_indx = 0;
			//wind_cnt++;
			unsigned char* frame_Hists_win_ptr = frame_Hists_row_ptr + h_adr_i * tot_bins;

#ifdef Use_Cascade
			int c = 0;
			for (int cascade_stage = 0; cascade_stage < cascade_stage_cnt; cascade_stage++) {//Need to add more stages. Collect stats to get further insight 
				//stats[cascade_stage] = stats[cascade_stage] + 1;
				for (int ic=0; ic < 15; ic++) {
					int hi = cascade_indices[c * 2];
					int hj = cascade_indices[c * 2 + 1];
					int a1 = hi*horiz_elements;
					int f1 = hi*hjk_max;
					int a2 = hj * tot_bins;
					int f2 = f1 + a2;
					for (int hk = 0; hk < tot_bins; hk++) {
						score += HIK_LUT[frame_Hists_win_ptr[a1 + a2 + hk]][f2 + hk];
					}
					c++;
				}
				if (score + cascade_stage_th[cascade_stage] < hik_bias)
					goto end_cascade;// break;
			}
			for (; c < hi_max*hj_max; c++) {
				int hi = cascade_indices[c * 2];
				int hj = cascade_indices[c * 2 + 1];
				int a1 = hi*horiz_elements;
				int f1 = hi*hjk_max;
				int a2 = hj * tot_bins;
				int f2 = f1 + a2;
				for (int hk = 0; hk < tot_bins; hk++) {
					score += HIK_LUT[frame_Hists_win_ptr[a1 + a2 + hk]][f2 + hk];
				}
			}
#else
			for (int hi=0; hi < hi_max; hi++) {
				int a1 = hi*horiz_elements;
				int f1 = hi*hjk_max;
				for (int hj = 0; hj < hj_max; hj++) {
					int a2 = hj * tot_bins;
					int f2 = f1 + a2;
					for (int hk = 0; hk < tot_bins; hk++) {
						score += HIK_LUT[frame_Hists_win_ptr[a1 + a2 + hk]][f2 + hk];
					}
				}
			}

#endif			

end_cascade:
			score = score - hik_bias + svm_score_offset;
			//printf("\n%.2f", score); getchar();

			if (score > SVM_Score_Th) {
				int i = h_adr_i*HSG_hist_stride[1] + 1;
				int j = v_adr_i*HSG_hist_stride[0] + 1;

				Rect detected_bb;
				bool exist = false;
				/*if (training_mode_local) {
					Rect clip_window;
					char clip_file[450];
					clip_window.x = i - 1;
					clip_window.y = j - 1;
					clip_window.width = WinSize[1] + 2;
					clip_window.height = WinSize[0] + 2;
					if (clip_window.x > 0 && clip_window.y > 0) {
						if (((clip_window.x + clip_window.width) < (cols - 1))&((clip_window.y + clip_window.height) < (rows - 1))) {
							sprintf(clip_file, "%s%s%s_S%04d_%04d_%04d.png", Dataset_Path, train_neg_dir, curr_img.c_str(), (int)(scale * 1000), i - 16, j - 16);
							Mat temp = imread(clip_file);   // Read the file
							Mat fp;
							frame(clip_window).copyTo(fp);
							GammaCorrection(fp, fp, 2.0);	//reverse gamma correction
							if (!temp.data) {//file does not exist
											 //printf("\n%s", clip_file);
								imwrite(clip_file, fp);//Does not work if project settings changed in C++ Code Generation. This works: Multi-threaded DLL/MD
													   //imshow("clip", fp);
													   //waitKey(0);
							}
							else
								exist = true;
							fp.release();
							temp.release();
						}
					}
				}
				else {*/
					//		Original Detected BB in padded and scaled frame
					detected_bb.x = (float)i;
					detected_bb.y = (float)j;
					detected_bb.width = (float)WinSize[1];
					detected_bb.height = (float)WinSize[0];
					//		Deal with objects towards edges of the frame (detected due to padding)
					detected_bb.x = detected_bb.x - Frame_Padding[2] * padding_scale;
					detected_bb.y = detected_bb.y - Frame_Padding[0] * padding_scale;
					if (detected_bb.x < 0) {
						detected_bb.width = detected_bb.width + detected_bb.x;
						detected_bb.x = 0;
					}
					if (detected_bb.y < 0) {
						detected_bb.height = detected_bb.height + detected_bb.y;
						detected_bb.y = 0;
					}
					/*					if (detected_bb.x + detected_bb.width>((cols - 1 - Frame_Padding[2] - Frame_Padding[3]))){
					detected_bb.width = ((cols)-1 - Frame_Padding[2] - Frame_Padding[3]) - detected_bb.x;
					}

					if (detected_bb.y + detected_bb.height>((rows - 1 - Frame_Padding[0] - Frame_Padding[1]))){
					detected_bb.height = ((rows)-1 - Frame_Padding[0] - Frame_Padding[1]) - detected_bb.y;
					}*/

					//		Add scaling effect
					detected_bb.x = (detected_bb.x / scale);
					detected_bb.y = (detected_bb.y / scale);
					detected_bb.width = (detected_bb.width / scale);
					detected_bb.height = (detected_bb.height / scale);
				//}

				mtx.lock();
				if (training_mode) {
					if (exist) persistent_hard_ex_cnt++;
					else hard_ex_cnt_stage++;
				}
				else {
					BB_Rects.push_back(detected_bb);
					BB_Scores.push_back((double)score);
					BB_Scales.push_back(scale * (float)HSG_Descriptor_WinSize[0] / (float)WinSize[0]);
					//cout << "\nDetected = " << BB_Scores.size() << endl;
				}
				mtx.unlock();
			}
#ifdef Use_P4
		});
	});
#else	
		}
	}
#endif
	//	ftime(&t_end);
	//	printf("\nWindwos = %d, Operations = %d", wind_cnt, oper_cnt);	getchar();
}


void HSG_Hist(Mat frame, unsigned char* frame_Hists){
//	ftime(&t_start);
	int rows = frame.rows;
	int cols = frame.cols;
	int no_horiz_hists = ((cols - 2) / HSG_hist_stride[1]);
	int no_vert_hists = ((rows - 2) / HSG_hist_stride[0]);
	int tot_bins = HSG_Hist_Bins + HSG_Aux_Bins;
	int horiz_elements = no_horiz_hists*tot_bins;
	unsigned char *frame_data_ptr = (unsigned char*)(frame.data);
	
	float* Grad_Mag;
	unsigned char* Grad_Or;
	Grad_Mag = new float[rows*cols];
	Grad_Or = new unsigned char[rows*cols];

//	ftime(&t_start);
	Mat luv;
	cvtColor(frame, luv, CV_BGR2Luv);	//Uses Parallel Loopbody, 
	unsigned char *luv_ptr = (unsigned char*)(luv.data);
	//RGB2LUV_ACF(frame_data_ptr, luv_ptr, rows*cols, 3, 2);
//	ftime(&t_end);

//	ftime(&t_start);
//	Calc_Grad_Img(luv, Grad_Mag, Grad_Or);

	//Calc_Grad_Img(frame, Grad_Mag, Grad_Or);
	computeGradient(frame, Grad_Mag, Grad_Or);
//	ftime(&t_end);

	const unsigned char bin_lut[2][9] = {{ 8,0,1,2,3,4,5,6,7},{ 1,2,3,4,5,6,7,8,0}};

//	ftime(&t_start);
	//Leave one-pixel boundary around the frame where gradient data is not available
#ifdef Use_P4
	parallel_for(size_t(r0), size_t(rows - HSG_Hist_Cell_Size[0]), size_t(HSG_hist_stride[0]), [&](size_t x) {
#else
	for (unsigned int x = r0; x < rows - HSG_Hist_Cell_Size[0]; x += HSG_hist_stride[0]) {
#endif
		int a1 = int((x - 1) / HSG_hist_stride[0])*horiz_elements;
		unsigned char* row_ptr = frame_Hists + a1;
		unsigned char* luv_row = luv_ptr + x*cols * 3;
		float* Grad_Mag_row = Grad_Mag + x*cols;
		unsigned char* Grad_Or_row = Grad_Or + x*cols;
		for (unsigned int y = c0; y < cols - HSG_Hist_Cell_Size[1]; y += HSG_hist_stride[1]) {
			int a2 = int((y - 1) / HSG_hist_stride[1]);
			unsigned char* hist_ptr = row_ptr + a2*tot_bins;
			unsigned char* luv_block = luv_row + y * 3;
			float* Grad_Mag_block = Grad_Mag_row + y;
			unsigned char* Grad_Or_block = Grad_Or_row + y;

			int luv_U_avg = 0;
			//int luv_V_avg = 0;
			for (int i = 0; i < HSG_Hist_Cell_Size[0]; i++) {
				unsigned char* luv_block_row = luv_block + i*cols * 3;
				float* Grad_Mag_block_row = Grad_Mag_block + i*cols;
				unsigned char* Grad_Or_block_row = Grad_Or_block + i*cols;
				for (int j = 0; j < HSG_Hist_Cell_Size[1]; j++) {
					float grad_mag = Grad_Mag_block_row[j];
					luv_U_avg += luv_block_row[j * 3 + 1];
					if ((int)grad_mag > 0) {
						//---------- 3 bin votes 1-2-1 -------------------
						unsigned char b = Grad_Or_block_row[j];
						hist_ptr[b] += 2;
						hist_ptr[bin_lut[0][b]] += 1;
						hist_ptr[bin_lut[1][b]] += 1;
					}
				}
			}
			luv_U_avg = luv_U_avg / (HSG_Hist_Cell_Size[0] * HSG_Hist_Cell_Size[1]);
			hist_ptr[HSG_Hist_Bins] = (int)(luv_U_avg / 4);
			//luv_V_avg = luv_V_avg / (HSG_Hist_Cell_Size[0] * HSG_Hist_Cell_Size[1]);
			//frame_Hists[adr_aux + HSG_Hist_Bins] = (int)(luv_V_avg / 4);
		}
#ifdef Use_P4
	});
#else
	}
#endif
//	getchar();

//	ftime(&t_end);
//	Release allocated memory
	delete Grad_Mag;
	delete Grad_Or;
//	ftime(&t_end);
}

void HSG_Training_Window(Mat& frame, float* hsg_feature_vec){
	GammaCorrection(frame, frame, 0.5);
	int rows = frame.rows;
	int cols = frame.cols;
	int no_horiz_hists = ((cols - 2) / HSG_hist_stride[1]);
	int no_vert_hists = ((rows - 2) / HSG_hist_stride[0]);
	/*cout << "\nRows = " << rows;
	cout << ", Cols = " << cols;
	cout << ", horiz hists = " << no_horiz_hists;
	cout << ", vert hists = " << no_vert_hists;
	getchar();*/
	//cout << "\n" << scale;

	unsigned char* frame_Hists;
	int mem_siz = (HSG_Hist_Bins + HSG_Aux_Bins)*(no_vert_hists)*(no_horiz_hists);
	frame_Hists = new unsigned char[mem_siz];			//No. of bins
	for (int i = 0; i < mem_siz; i++)
		frame_Hists[i] = 0;		//initialize hists to zero*/
	//Calculate Histograms for the whole image
	HSG_Hist(frame, frame_Hists);
	//HSG_Feature_V2(frame, frame_Hists);

	Mutex mtx;
	for (int j = r0; j < rows - HSG_Descriptor_WinSize[0] + 1; j = j + (Detect_Win_Stride_Offset[0]* HSG_hist_stride[0])){
		for (int i = c0; i < cols - HSG_Descriptor_WinSize[1] + 1; i = i + (Detect_Win_Stride_Offset[1] * HSG_hist_stride[1])){	//boundary pixels are not used
			float score = 0;
			int feature_indx = 0;
//			int Q_Val;
			int v_adr_i = int((j - 1) / HSG_hist_stride[0]);
			int h_adr_i = int((i - 1) / HSG_hist_stride[1]);
			
			for (int hi = 0; hi< (HSG_Descriptor_WinSize[0] / HSG_hist_stride[0])-1; hi++){
				int v_adr = v_adr_i + hi;
				for (int hj = 0; hj < (HSG_Descriptor_WinSize[1] / HSG_hist_stride[1])-1; hj++){
					int h_adr = h_adr_i + hj;
					for (int hk = 0; hk < HSG_Hist_Bins + HSG_Aux_Bins; hk++){
						hsg_feature_vec[feature_indx++] = (float)frame_Hists[v_adr*no_horiz_hists*(HSG_Hist_Bins + HSG_Aux_Bins) + h_adr*(HSG_Hist_Bins + HSG_Aux_Bins) + hk];
					}
				}
			}
		}//);
	}//);
	//	Release Memory
	delete frame_Hists;
	//getchar();
}

class MultiScale_Detector_Parallel_Process : public ParallelLoopBody
{
private:
	Mat frame;
	vector<Rect>* BB_Rects;
	vector<double>* BB_Scores;
	vector<double>* BB_Scales;
	vector<double>* Scales;
	Mutex* mtx;

public:
	MultiScale_Detector_Parallel_Process(const Mat _frame, vector<Rect>* _BB_Rects, vector<double>* _BB_Scores, vector<double>* _BB_Scales, vector<double>* _Scales, Mutex* _mtx)
	{
		frame = _frame;
		BB_Rects = _BB_Rects;
		BB_Scores = _BB_Scores;
		BB_Scales = _BB_Scales;
		Scales = _Scales;
		mtx = _mtx;
	}

	void operator()(const Range& range) const
	{
		int i, i1 = range.start, i2 = range.end;

		for (i = i1; i < i2; i++)
		{
			Mat frame_resized, frame_blurred;
			Size sz(cvRound(frame.cols*Scales->at(i)), cvRound(frame.rows*Scales->at(i)));
			resize(frame, frame_resized, sz);// , CV_INTER_CUBIC);

											 //if (Scales->at(i) < 0.75)
											 //GaussianBlur(frame, frame_blurred, Size(7, 7), 0, 0);
											 //else
											 //frame.copyTo(frame_blurred);
											 //resize(frame_blurred, frame_resized, sz);// , CV_INTER_CUBIC);

			vector<Rect> BB_Rects_i;		//Detected BBs for the current scale indexed by i
			vector<double> BB_Scores_i;		//Corresponding scores for the current scale indexed by i
			vector<double> BB_Scales_i;		//Corresponding scales for the current scale indexed by i
			SingleScale_Detector(frame_resized, BB_Rects_i, BB_Scores_i, Scales->at(i), BB_Scales_i);

			mtx->lock();
			for (int ii = 0; ii < BB_Scores_i.size(); ii++) {
				BB_Scores->push_back(BB_Scores_i[ii]);
				BB_Rects->push_back(BB_Rects_i[ii]);
				//				BB_Scales->push_back(1 / Scales->at(i));
				BB_Scales->push_back(1 / BB_Scales_i[ii]);
				//printf("\n%.2f", BB_Scores_i[ii]);
			}
			BB_Rects_i.clear();
			BB_Scores_i.clear();
			mtx->unlock();
		}
	}

};


void MultiScale_Detector(Mat& Frame, vector<Rect>& BB_Rects, vector<double>& BB_Scores) {
	//Pre-processing
	//GaussianBlur(cur_frame, cur_frame, Size(7, 7), 0,0);
	//medianBlur(cur_frame, cur_frame, 5);
	blur(Frame, Frame, Size(3, 3), Point(-1, -1));
	GammaCorrection(Frame, Frame, 0.5);

	//Clear Data structures if they are not already
	BB_Rects.clear();
	BB_Scores.clear();
	vector<double> BB_Scales;		//Detected Scales, used with BB_Rects and BB_Scores

	int c1 = 0;
	//1- Setup detection Scales
	vector<double> Scales;
	float scale = (float)HSG_Descriptor_WinSize[0] / min_object_height;
	while ((floor(Frame.rows*scale) > (HSG_Descriptor_WinSize[0] + 2)) && floor(Frame.cols*scale) > (HSG_Descriptor_WinSize[1] + 2)) {//Check for minimum scale allowed constraint by Descriptor Window Size
		c1++;
		//c1 = 1;
		switch (c1) {
		case 1:
		case 2:
		case 7:
		case 8:
		case 12:
		case 13:
		case 18:
		case 19:
		{
			Scales.push_back(scale);
			//cout << endl << scale;
			break;
		}
		}

#ifdef Scale_Linear
		scale = scale - Scale_Step;
#else
		scale = scale / Scale_Step;
#endif
	}

	//2- Spawn parallel processing of all scales
	//cout << "\n\tNumber of scales = " << Scales.size();
	//getchar();
	Mutex mtx;
	parallel_for_(cv::Range(0, Scales.size()), MultiScale_Detector_Parallel_Process(Frame, &BB_Rects, &BB_Scores, &BB_Scales, &Scales, &mtx));
	//getchar();

	//3- Non-Maxima Suppression
	if (nms_method == GroupRectangles_OCV) {
		vector<int> levels(BB_Rects.size(), 0);
		groupRectangles(BB_Rects, levels, BB_Scores, 1, 0.2);
	}
	else if (nms_method == MeanShift_OCV) {
		NMS_DetectedBB_MeanShift(BB_Rects, BB_Scores, BB_Scales, SVM_Score_Th, Size(HSG_Descriptor_WinSize[1], HSG_Descriptor_WinSize[0]));
	}
	else if (nms_method == Custom_NMS) {
		NMS_Custom(BB_Rects, BB_Scores, BB_Scales, SVM_Score_Th, Size(23, 56));// 20, 48));// 39, 96));
	}
}

void SingleScale_Detector(Mat& Frame, vector<Rect>& BB_Rects, vector<double>& BB_Scores, double scale, vector<double>& BB_Scales) {
	//GammaCorrection(Frame, Frame, 0.5);
	//	ftime(&t_start);
	HSG_SingleScale_Detector(Frame, BB_Rects, BB_Scores, scale, BB_Scales);
	//	ftime(&t_end);
}


void computeGradient(const Mat& img, float* Grad_Mag, unsigned char* Grad_Or) {
	Size gradsize(img.cols, img.rows);
	Size wholeSize;
	Point roiofs;
	img.locateROI(wholeSize, roiofs);
	const float pi = 3.141593;

	int i, y;
	bool gammaCorrection = 0;
	float fGamma = 0.5;
	int width = gradsize.width;
	int nbins = 9;
	float angleScale = (float)(nbins / CV_PI);

	Mat_<float> _lut(1, 256);
	const float* const lut = &_lut(0, 0);
#ifdef Use_SSE
	const int indeces[] = { 0, 1, 2, 3 };
	__m128i idx = _mm_loadu_si128((const __m128i*)indeces);		//This instruction not translated in SSE2NEON.h
	__m128i ifour = _mm_set1_epi32(4);

	float* const _data = &_lut(0, 0);
	if (gammaCorrection)
		for (i = 0; i < 256; i += 4)
		{
			_mm_storeu_ps(_data + i, _mm_sqrt_ps(_mm_cvtepi32_ps(idx)));
			idx = _mm_add_epi32(idx, ifour);
		}
	else
		for (i = 0; i < 256; i += 4)
		{
			_mm_storeu_ps(_data + i, _mm_cvtepi32_ps(idx));
			idx = _mm_add_epi32(idx, ifour);
		}
#else
	if (gammaCorrection)
		for (i = 0; i < 256; i++)
			_lut(0, i) = saturate_cast<uchar>(pow((float)(i / 255.0), fGamma) * 255.0f); //std::sqrt((float)i);
	else
		for (i = 0; i < 256; i++)
			_lut(0, i) = (float)i;
#endif

	// x- & y- derivatives for the whole row
#ifdef Use_P4
	parallel_for(size_t(1), size_t(gradsize.height - 1), size_t(1), [&](size_t y) {
#endif
		AutoBuffer<float> _dbuf(width * 4);			//this will contain Dx, Dy, Mag and Angle for the whole row, hence width*4 size.
		float* const dbuf = _dbuf;
		Mat Dx(1, width, CV_32F, dbuf);
		Mat Dy(1, width, CV_32F, dbuf + width);
		Mat Mag(1, width, CV_32F, dbuf + width * 2);
		Mat Angle(1, width, CV_32F, dbuf + width * 3);
#ifndef Use_P4
		for (y = 1; y < gradsize.height - 1; y++) {		//iterate over rows	
#endif
			const uchar* imgPtr = img.ptr(y);					//address of the first pixel in this row (y)
			const uchar* prevPtr = img.data + img.step*(y + 1);	//these have been interchanged because of the way HSG was trained.
			const uchar* nextPtr = img.data + img.step*(y - 1);

			float* gradPtr = Grad_Mag + y*width;
			unsigned char* qanglePtr = Grad_Or + y*width;

			int x = 1;
#ifdef Use_SSE		//With and without SSE give slightly different results, Investigate!
			for (; x <= width - 4; x += 4)
			{
				int x0 = x * 3, x1 = (x + 1) * 3, x2 = (x + 2) * 3, x3 = (x + 3) * 3;
				typedef const uchar* const T;
				T p02 = imgPtr + (x + 1) * 3, p00 = imgPtr + (x - 1) * 3;
				T p12 = imgPtr + (x + 2) * 3, p10 = imgPtr + x * 3;
				T p22 = imgPtr + (x + 3) * 3, p20 = p02;
				T p32 = imgPtr + (x + 4) * 3, p30 = p12;

				//To-Do 1- _mm_loadu_si128 2- _mm_sub_epi8, _mm_sub_epi16

				//The instructions below are too slow!! Because they use _mm_set_ps to move scalars (interleaved rgb values) into sse registers
				//Take a look at the assembly and you will see a lot of instructions being generted against these. 
				//So planar rgb is best for SSE as done by ACF.
				__m128 _dx0 = _mm_sub_ps(_mm_set_ps(lut[p32[0]], lut[p22[0]], lut[p12[0]], lut[p02[0]]),
					_mm_set_ps(lut[p30[0]], lut[p20[0]], lut[p10[0]], lut[p00[0]]));
				__m128 _dx1 = _mm_sub_ps(_mm_set_ps(lut[p32[1]], lut[p22[1]], lut[p12[1]], lut[p02[1]]),
					_mm_set_ps(lut[p30[1]], lut[p20[1]], lut[p10[1]], lut[p00[1]]));
				__m128 _dx2 = _mm_sub_ps(_mm_set_ps(lut[p32[2]], lut[p22[2]], lut[p12[2]], lut[p02[2]]),
					_mm_set_ps(lut[p30[2]], lut[p20[2]], lut[p10[2]], lut[p00[2]]));

				__m128 _dy0 = _mm_sub_ps(_mm_set_ps(lut[nextPtr[x3]], lut[nextPtr[x2]], lut[nextPtr[x1]], lut[nextPtr[x0]]),
					_mm_set_ps(lut[prevPtr[x3]], lut[prevPtr[x2]], lut[prevPtr[x1]], lut[prevPtr[x0]]));
				__m128 _dy1 = _mm_sub_ps(_mm_set_ps(lut[nextPtr[x3 + 1]], lut[nextPtr[x2 + 1]], lut[nextPtr[x1 + 1]], lut[nextPtr[x0 + 1]]),
					_mm_set_ps(lut[prevPtr[x3 + 1]], lut[prevPtr[x2 + 1]], lut[prevPtr[x1 + 1]], lut[prevPtr[x0 + 1]]));
				__m128 _dy2 = _mm_sub_ps(_mm_set_ps(lut[nextPtr[x3 + 2]], lut[nextPtr[x2 + 2]], lut[nextPtr[x1 + 2]], lut[nextPtr[x0 + 2]]),
					_mm_set_ps(lut[prevPtr[x3 + 2]], lut[prevPtr[x2 + 2]], lut[prevPtr[x1 + 2]], lut[prevPtr[x0 + 2]]));

				__m128 _mag0 = _mm_add_ps(_mm_mul_ps(_dx0, _dx0), _mm_mul_ps(_dy0, _dy0));
				__m128 _mag1 = _mm_add_ps(_mm_mul_ps(_dx1, _dx1), _mm_mul_ps(_dy1, _dy1));
				__m128 _mag2 = _mm_add_ps(_mm_mul_ps(_dx2, _dx2), _mm_mul_ps(_dy2, _dy2));

				__m128 mask = _mm_cmpgt_ps(_mag2, _mag1);
				_dx2 = _mm_or_ps(_mm_and_ps(_dx2, mask), _mm_andnot_ps(mask, _dx1));
				_dy2 = _mm_or_ps(_mm_and_ps(_dy2, mask), _mm_andnot_ps(mask, _dy1));

				mask = _mm_cmpgt_ps(_mm_max_ps(_mag2, _mag1), _mag0);
				_dx2 = _mm_or_ps(_mm_and_ps(_dx2, mask), _mm_andnot_ps(mask, _dx0));
				_dy2 = _mm_or_ps(_mm_and_ps(_dy2, mask), _mm_andnot_ps(mask, _dy0));

				_mm_storeu_ps(dbuf + x, _dx2);
				_mm_storeu_ps(dbuf + x + width, _dy2);
			}
#endif
			for (; x < width - 1; x++)
			{
				int x1 = x * 3;
				float dx0, dy0, dx, dy, mag0, mag;
				const uchar* p2 = imgPtr + (x + 1) * 3;
				const uchar* p0 = imgPtr + (x - 1) * 3;

				dx0 = lut[p2[2]] - lut[p0[2]];
				dy0 = lut[nextPtr[x1 + 2]] - lut[prevPtr[x1 + 2]];
				mag0 = dx0*dx0 + dy0*dy0;		//Magnitude for Red

				dx = lut[p2[1]] - lut[p0[1]];
				dy = lut[nextPtr[x1 + 1]] - lut[prevPtr[x1 + 1]];
				mag = dx*dx + dy*dy;			//Magnitude for Green
				if (mag0 < mag) {
					dx0 = dx;
					dy0 = dy;
					mag0 = mag;
				}

				dx = lut[p2[0]] - lut[p0[0]];
				dy = lut[nextPtr[x1]] - lut[prevPtr[x1]];
				mag = dx*dx + dy*dy;		//Magnitude for Blue
				if (mag0 < mag) {
					dx0 = dx;
					dy0 = dy;
					mag0 = mag;
				}

				dbuf[x] = dx0;				//dx for max mag
				dbuf[x + width] = dy0;		//dy for max mag
			}
			// computing angles and magnidutes
			cartToPolar(Dx, Dy, Mag, Angle, false);

			// filling the result matrix
			x = 0;
#ifdef Use_SSE
			__m128 fzero = _mm_setzero_ps();
			__m128 _angleScale = _mm_set1_ps(angleScale), fone = _mm_set1_ps(1.0f);
			__m128i _nbins = _mm_set1_epi32(nbins), izero = _mm_setzero_si128();
			__m128 Q = _mm_set1_ps(0.0625f);
			__m128i store[3];
			int ans[4];

			int n = 1;
			for (; x <= width - n * 4; x += n * 4)
			{
				for (int p = 0; p < 2; p++) {
					int a = p * 4;
					__m128 mag = _mm_loadu_ps(dbuf + x + a + (width << 1));				//float mag = dbuf[x+width*2]
					__m128 mag_q = _mm_mul_ps(mag, Q);								//mag_q = mag*0.0625 (mag/16), use "_mm_cvtps_epi32" or "_mm_cvttps_epi32" for rounding/truncating to int
					_mm_storeu_ps(gradPtr + x + a, mag_q);								//store the quantized mag (mag_q)
					__m128 _angle = _mm_loadu_ps(dbuf + x + a + width * 3);				//float angle = dbuf[x+width*3]
					_angle = _mm_mul_ps(_angleScale, _angle);	//angle*=angleScale;
					__m128 sign = _mm_and_ps(fone, _mm_cmplt_ps(_angle, fzero));
					__m128i _hidx = _mm_cvttps_epi32(_angle);						//int hidx = cvFloor(angle);
					_hidx = _mm_sub_epi32(_hidx, _mm_cvtps_epi32(sign));			//?
					_angle = _mm_sub_ps(_angle, _mm_cvtepi32_ps(_hidx));			// angle -= hidx;
					__m128i mask0 = _mm_sub_epi32(izero, _mm_srli_epi32(_hidx, 31));
					__m128i it0 = _mm_and_si128(mask0, _nbins);
					mask0 = _mm_cmplt_epi32(_hidx, _nbins);
					__m128i it1 = _mm_andnot_si128(mask0, _nbins);
					_hidx = _mm_add_epi32(_hidx, _mm_sub_epi32(it0, it1));
					store[p] = _hidx;
				}

				//__m128i or_16 = _mm_packs_epi16(_mm_packs_epi32(store[0], store[1]), _mm_packs_epi32(store[2], store[3]));
				//_mm_storel_epi64((__m128i*)(qanglePtr + x), or_16);

				//_mm_storeu_si128((__m128i*)(qanglePtr + x), or_16);

				_mm_store_si128((__m128i*)ans, store[0]);
				qanglePtr[x] = ans[0];
				qanglePtr[x + 1] = ans[1];
				qanglePtr[x + 2] = ans[2];
				qanglePtr[x + 3] = ans[3];
			}
#endif
			for (; x < width; x++)
			{
				gradPtr[x] = dbuf[x + width * 2] / 16;
				/*float angle = dbuf[x + width * 3];
				if (angle > CV_PI) angle -= CV_PI;
				if (angle < 0) angle += CV_PI;
				angle *= angleScale;
				angle = cvFloor(angle);
				qanglePtr[x] = (uchar)angle;*/
				//------
				float angle = dbuf[x + width * 3] * angleScale;
				int hidx = cvFloor(angle);
				angle -= hidx;
				if (hidx < 0)
					hidx += nbins;
				else if (hidx >= nbins)
					hidx -= nbins;
				qanglePtr[x] = (uchar)hidx;
			}
#ifdef Use_P4
		});
#else
	}
#endif
}

void GammaCorrection(Mat& src, Mat& dst, float fGamma) {
	unsigned char lut[256];
	for (int i = 0; i < 256; i++)
		lut[i] = saturate_cast<uchar>(pow((float)(i / 255.0), fGamma) * 255.0f);

	dst = src.clone();
	MatIterator_<Vec3b> it, end;
	for (it = dst.begin<Vec3b>(), end = dst.end<Vec3b>(); it != end; it++)
	{
		(*it)[0] = lut[((*it)[0])];
		(*it)[1] = lut[((*it)[1])];
		(*it)[2] = lut[((*it)[2])];
	}
	/*Mat correctGamma(Mat& img, double gamma) {
	double inverse_gamma = 1.0 / gamma;

	Mat lut_matrix(1, 256, CV_8UC1);
	uchar * ptr = lut_matrix.ptr();
	for (int i = 0; i < 256; i++)
	ptr[i] = (int)(pow((double)i / 255.0, inverse_gamma) * 255.0);

	Mat result;
	LUT(img, lut_matrix, result);

	return result;
	}*/
}

void Test_HSG_HIK_Detector(string DirPath) {
	Mat frame, temp;
	vector<Rect> found;
	vector<double> scores;
	float fps = 15, proc_t = 66;
	char str[20];

	if (DirPath.empty()) {
		VideoCapture capture;
		//capture.open("D:\\RnD\\Current_Projects\\Musawwir\\Object_Detection\\Frameworks\\SW\\Dataset\\Person\\others\\camera1.mov");	
		//capture.open("D:\\RnD\\Current_Projects\\Musawwir\\Frameworks\\SW\\Dataset\\Face\\Vidz\\DATA (18)_x264.mp4");
		capture.open(0);
		//	capture.set(CV_CAP_PROP_FRAME_WIDTH, 1280);
		//	capture.set(CV_CAP_PROP_FRAME_HEIGHT, 720);
		if (!capture.isOpened()) { printf("--(!)Error opening video capture\n"); return; }
		while (capture.read(frame)) {
			//resize(frame, frame, Size(1920, 1080));
			frame.copyTo(temp);
			ftime(&t_start);
			MultiScale_Detector(temp, found, scores);
			ftime(&t_end);
			float msec = int((t_end.time - t_start.time) * 1000 + (t_end.millitm - t_start.millitm));
			proc_t = 0.99*proc_t + 0.01*msec;
			printf("\nfps = %.1f\ttime = %.1f", 1000 / msec, proc_t);
			fps = 1000 / proc_t;
			sprintf(str, "%.1f", fps);
			cv::putText(frame, str, Point(20, 20), CV_FONT_HERSHEY_COMPLEX, 1, Scalar(255, 0, 0), 1, 8);
			DrawRectangles(frame, found, scores);
			imshow("Live Video", frame);
			waitKey(1);
		}
	}
	else {
		cout << "\n\n\t Processing images in directory: \n\t" << DirPath;
		int file_cnt = 0;
		intptr_t hFile;
		_finddata_t fd;
		_chdir(DirPath.c_str());
		hFile = _findfirst("*.*", &fd);
			do {
				if (fd.name != string(".") && fd.name != string("..")) {
					printf("\r%.2f", (double)file_cnt / 12.18);
					file_cnt++;
					printf("\nReading file ->%s", fd.name);
					frame = imread(fd.name);
					frame.copyTo(temp);
					//resize(frame, frame, Size(1920, 1080));
					frame.copyTo(temp);
					ftime(&t_start);
					MultiScale_Detector(temp, found, scores);
					ftime(&t_end);
					float msec = int((t_end.time - t_start.time) * 1000 + (t_end.millitm - t_start.millitm));
					proc_t = 0.99*proc_t + 0.01*msec;
					printf("\nfps = %.1f\ttime = %.1f", 1000 / msec, proc_t);
					fps = 1000 / proc_t;
					sprintf(str, "%.1f", fps);
					cv::putText(frame, str, Point(20, 20), CV_FONT_HERSHEY_COMPLEX, 1, Scalar(255, 0, 0), 1, 8);
					DrawRectangles(frame, found, scores);
					imshow("Still Images", frame);
					waitKey(1);
					//imshow("Still Images", frame);
					//char output_img_file[200];
					//sprintf(output_img_file, "%s%s\\%s", Dataset_Path, detect_offline_output_img_dir, fd.name);
					//imwrite(output_img_file, frame);
					//waitKey(1);
				}
			} while (_findnext(hFile, &fd) == 0);
			_findclose(hFile);
		}
	
}

void DrawRectangles(Mat& Frame, vector<Rect>& BB_Rects, vector<double>& BB_Scores) {
	for (int x = 0; x < BB_Scores.size(); x++) {
		if (BB_Scores[x] < SVM_Score_Th) continue;
		Rect detected_bb;
		char str[50];
		sprintf(str, "%.02f", BB_Scores[x]);
		detected_bb = BB_Rects[x];
		cv::putText(Frame, str, Point(detected_bb.x, detected_bb.y), CV_FONT_HERSHEY_COMPLEX, 1, Scalar(255, 0, 0), 1, 8);
		rectangle(Frame, detected_bb, Scalar(255, 0, 0), 4);
	}
}

void HSG_Feature(Mat frame, unsigned char * feature_vec)
{
	int fRows = frame.rows;
	int fCols = frame.cols;
	int No_horiz_hists = ((fCols - 2) / HSG_hist_stride[1]);
	int No_vert_hists = ((fRows - 2) / HSG_hist_stride[0]);
	unsigned char *frame_data_ptr = (unsigned char*)(frame.data);
	Mat luv;
	cvtColor(frame, luv, CV_BGR2Luv);
	unsigned char *luv_data_ptr = (unsigned char*)(luv.data);

	int* temp;
	temp = new int[No_horiz_hists*No_vert_hists];
	for (int x = 0; x < (No_horiz_hists*No_vert_hists); x++)
		temp[x] = 0;
		
	float Grad_Mag;
	float Grad_Or;

	//Parallel FOr shold theoretically give wrong results because feature vector could be accessed by multiple unrolled loops
	//and would lead to lesser count in the bins. BUT ironcially it gives better results!!!
//	parallel_for(size_t(1), size_t(fRows - 1), size_t(1), [&](size_t j) {
//		parallel_for(size_t(1), size_t(fCols - 1), size_t(1), [&](size_t i) {
	for (int j = 1; j < fRows - 1; j++){			//leave boundary pixels
		for (int i = 1; i < fCols - 1; i++){		//leave boundary pixels
			int dx[3], dy[3];
			int k, temp_mag1, temp_mag2;
			int jcols3 = j*fCols * 3;
			int cols3 = fCols * 3;
			int i3 = i * 3;
			int a1 = jcols3 + i3;
			dx[0] = frame_data_ptr[a1 + 3] - frame_data_ptr[a1 - 3];
			dx[1] = frame_data_ptr[a1 + 4] - frame_data_ptr[a1 - 2];
			dx[2] = frame_data_ptr[a1 + 5] - frame_data_ptr[a1 - 1];
			dy[0] = -frame_data_ptr[a1 + cols3] + frame_data_ptr[a1 - cols3];
			dy[1] = -frame_data_ptr[a1 + cols3 + 1] + frame_data_ptr[a1 - cols3 + 1];
			dy[2] = -frame_data_ptr[a1 + cols3 + 2] + frame_data_ptr[a1 - cols3 + 2];

			temp_mag1 = (dx[0] * dx[0] + dy[0] * dy[0]);
			k = 0;//greatest mag. value index
			temp_mag2 = (dx[1] * dx[1] + dy[1] * dy[1]);
			if (temp_mag2 > temp_mag1) {
				temp_mag1 = temp_mag2;
				k = 1;
			}
			temp_mag2 = (dx[2] * dx[2] + dy[2] * dy[2]);
			if (temp_mag2 > temp_mag1) {
				temp_mag1 = temp_mag2;
				k = 2;
			}
			Grad_Mag = int(sqrtf(float(temp_mag1)));
			float Grad_Or_temp = atan2f(float(dy[k]), float(dx[k]));
#ifdef Orient
			if (Grad_Or_temp<0) Grad_Or_temp += twopi;							//Orientation (0-360), add 2pi
#else
			if (Grad_Or_temp<0) Grad_Or_temp += CV_PI;									//No Orientation (0-180), add pi
#endif
			Grad_Or = Grad_Or_temp;

			/*
			---------------------------------------------------------------------------------------------------------------------------------
			Grad_Mag and Grad_Or are ready to populate hists
			---------------------------------------------------------------------------------------------------------------------------------
			*/
			int h1, h2, h3, h4, v1, v2, v3, v4, adr1, adr2, adr3, adr4, b0,b1,b2;
			h4 = (i-1) / HSG_hist_stride[1];
			v4 = (j-1) / HSG_hist_stride[0];
			h3 = h4 - 1;
			v3 = v4;
			h2 = h4;
			v2 = v4 - 1;
			h1 = h4 - 1;
			v1 = v4 - 1;

			feature_vec[h4] = Grad_Mag;
			feature_vec[v4] = Grad_Or;


			Grad_Mag = int(Grad_Mag / HSG_GradMag_Q);
			if (Grad_Mag > 0) {
				b1 = (int)floor(Grad_Or / HSG_bin_size);
				if (b1 < 0)	b1 = HSG_Hist_Bins - 1;
				if (b1 - 1 >= 0)
					b0 = b1 - 1;
				else
					b0 = HSG_Hist_Bins - 1;
				if (b1 + 1 <= HSG_Hist_Bins - 1)
					b2 = b1 + 1;
				else
					b2 = 0;
			}

			int luv_U = luv_data_ptr[a1 + 1];

			if ((h4 < No_horiz_hists - 1) & (v4 < No_vert_hists - 1)) {
				adr4 = v4*No_horiz_hists*(HSG_Hist_Bins + HSG_Aux_Bins) + h4*(HSG_Hist_Bins + HSG_Aux_Bins);
				if (Grad_Mag > 0) {
					feature_vec[adr4 + b1] += 2;
					feature_vec[adr4 + b0] += 1;
					feature_vec[adr4 + b2] += 1;
				}
				//frame_Hists[adr4 + HSG_Hist_Bins] += luv_U;
				temp[v4*No_horiz_hists + h4] += luv_U;
			}

			if ((h3 >= 0) & (v3 < No_vert_hists - 1)) {
				adr3 = v3*No_horiz_hists*(HSG_Hist_Bins + HSG_Aux_Bins) + h3*(HSG_Hist_Bins + HSG_Aux_Bins);
				if (Grad_Mag > 0) {
					feature_vec[adr3 + b1] += 2;
					feature_vec[adr3 + b0] += 1;
					feature_vec[adr3 + b2] += 1;
				}
				//				frame_Hists[adr3 + HSG_Hist_Bins] += luv_U;
				temp[v3*No_horiz_hists + h3] += luv_U;
			}
			if ((h2 < No_horiz_hists - 1) & (v2 >= 0)) {
				adr2 = v2*No_horiz_hists*(HSG_Hist_Bins + HSG_Aux_Bins) + h2*(HSG_Hist_Bins + HSG_Aux_Bins);
				if (Grad_Mag > 0) {
					feature_vec[adr2 + b1] += 2;
					feature_vec[adr2 + b0] += 1;
					feature_vec[adr2 + b2] += 1;
				}
				//				frame_Hists[adr2 + HSG_Hist_Bins] += luv_U;
				temp[v2*No_horiz_hists + h2] += luv_U;
			}
			if ((h1 >= 0) & (v1 >= 0)) {
				adr1 = v1*No_horiz_hists*(HSG_Hist_Bins + HSG_Aux_Bins) + h1*(HSG_Hist_Bins + HSG_Aux_Bins);
				if (Grad_Mag > 0) {
					feature_vec[adr1 + b1] += 2;
					feature_vec[adr1 + b0] += 1;
					feature_vec[adr1 + b2] += 1;
				}
				//				frame_Hists[adr1 + HSG_Hist_Bins] += luv_U;
				temp[v1*No_horiz_hists + h1] += luv_U;
			}
		}//);
	}//);

	for (int i = 0; i < No_vert_hists; i++) {
		for (int j = 0; j < No_horiz_hists; j++) {
			int adr = i*No_horiz_hists*(HSG_Hist_Bins + HSG_Aux_Bins) + j*(HSG_Hist_Bins + HSG_Aux_Bins) + HSG_Hist_Bins;
			int U = temp[i*No_horiz_hists + j];
			feature_vec[adr] = U / (HSG_Hist_Cell_Size[0] * HSG_Hist_Cell_Size[1] * 4);
		}
	}
	//	Release allocated memory
	delete temp;
}

void HSG_Feature_V2(Mat frame, unsigned char * feature_vec)
{
	int fRows = frame.rows;
	int fCols = frame.cols;
	int No_horiz_hists = ((fCols - 2) / HSG_hist_stride[1]);
	int No_vert_hists = ((fRows - 2) / HSG_hist_stride[0]);
	unsigned char *frame_data_ptr = (unsigned char*)(frame.data);
	Mat luv;
	cvtColor(frame, luv, CV_BGR2Luv);
//	cvtColor(frame, luv, CV_BGR2YCrCb);
	unsigned char *luv_data_ptr = (unsigned char*)(luv.data);

	int* temp;
	temp = new int[No_horiz_hists*No_vert_hists];
	for (int x = 0; x < (No_horiz_hists*No_vert_hists); x++)
		temp[x] = 0;

	float Grad_Mag, Grad_Or;
	float* Grad_Mag_;
	float* Grad_Or_;
	Grad_Mag_ = new float[fRows*fCols];
	Grad_Or_ = new float[fRows*fCols];
//	Calc_Grad_Img(frame, Grad_Mag_, Grad_Or_);
//	Calc_Grad_Img(luv, Grad_Mag_, Grad_Or_);

	//Parallel FOr shold theoretically give wrong results because feature vector could be accessed by multiple unrolled loops
	//and would lead to lesser count in the bins. BUT ironcially it gives better results!!!
		//parallel_for(size_t(1), size_t(fRows - 1), size_t(1), [&](size_t j) {
			//parallel_for(size_t(1), size_t(fCols - 1), size_t(1), [&](size_t i) {
	for (int j = 1; j < fRows - 1; j++) {			//leave boundary pixels
		for (int i = 1; i < fCols - 1; i++) {		//leave boundary pixels
			int dx[3], dy[3];
			int k, temp_mag1, temp_mag2;
			int jcols3 = j*fCols * 3;
			int cols3 = fCols * 3;
			int i3 = i * 3;
			int a1 = jcols3 + i3;
/*			dx[0] = frame_data_ptr[a1 + 3] - frame_data_ptr[a1 - 3];
			dx[1] = frame_data_ptr[a1 + 4] - frame_data_ptr[a1 - 2];
			dx[2] = frame_data_ptr[a1 + 5] - frame_data_ptr[a1 - 1];
			dy[0] = -frame_data_ptr[a1 + cols3] + frame_data_ptr[a1 - cols3];
			dy[1] = -frame_data_ptr[a1 + cols3 + 1] + frame_data_ptr[a1 - cols3 + 1];
			dy[2] = -frame_data_ptr[a1 + cols3 + 2] + frame_data_ptr[a1 - cols3 + 2];

			temp_mag1 = (dx[0] * dx[0] + dy[0] * dy[0]);
			k = 0;//greatest mag. value index
			temp_mag2 = (dx[1] * dx[1] + dy[1] * dy[1]);
			if (temp_mag2 > temp_mag1) {
				temp_mag1 = temp_mag2;
				k = 1;
			}
			temp_mag2 = (dx[2] * dx[2] + dy[2] * dy[2]);
			if (temp_mag2 > temp_mag1) {
				temp_mag1 = temp_mag2;
				k = 2;
			}
			Grad_Mag = int(sqrtf(float(temp_mag1)));
			float Grad_Or_temp = atan2f(float(dy[k]), float(dx[k]));
#ifdef Orient
			if (Grad_Or_temp<0) Grad_Or_temp += twopi;							//Orientation (0-360), add 2pi
#else
			if (Grad_Or_temp<0) Grad_Or_temp += pi;									//No Orientation (0-180), add pi
#endif
			Grad_Or = Grad_Or_temp;
*/
			/*
			---------------------------------------------------------------------------------------------------------------------------------
			Grad_Mag and Grad_Or are ready to populate hists
			---------------------------------------------------------------------------------------------------------------------------------
			*/
			Grad_Or = Grad_Or_[j*fCols + i];
			Grad_Mag = Grad_Mag_[j*fCols + i];

			int h1, h2, h3, h4, v1, v2, v3, v4, adr1, adr2, adr3, adr4, b0, b1, b2;
			h4 = (i - 1) / HSG_hist_stride[1];
			v4 = (j - 1) / HSG_hist_stride[0];
			h3 = h4 - 1;
			v3 = v4;
			h2 = h4;
			v2 = v4 - 1;
			h1 = h4 - 1;
			v1 = v4 - 1;


			Grad_Mag = int(Grad_Mag / HSG_GradMag_Q);
			if (Grad_Mag > 0) {
				b1 = (int)floor(Grad_Or / HSG_bin_size);
				if (b1 < 0)	b1 = HSG_Hist_Bins - 1;
				if (b1 - 1 >= 0)
					b0 = b1 - 1;
				else
					b0 = HSG_Hist_Bins - 1;
				if (b1 + 1 <= HSG_Hist_Bins - 1)
					b2 = b1 + 1;
				else
					b2 = 0;
			}

			int luv_U = luv_data_ptr[a1 + 1];

			if ((h4 < No_horiz_hists - 1) & (v4 < No_vert_hists - 1)) {
				adr4 = v4*No_horiz_hists*(HSG_Hist_Bins + HSG_Aux_Bins) + h4*(HSG_Hist_Bins + HSG_Aux_Bins);
				if (Grad_Mag > 0) {
					feature_vec[adr4 + b1] += 2;
					feature_vec[adr4 + b0] += 1;
					feature_vec[adr4 + b2] += 1;
				}
				//frame_Hists[adr4 + HSG_Hist_Bins] += luv_U;
				temp[v4*No_horiz_hists + h4] += luv_U;
			}

			if ((h3 >= 0) & (v3 < No_vert_hists - 1)) {
				adr3 = v3*No_horiz_hists*(HSG_Hist_Bins + HSG_Aux_Bins) + h3*(HSG_Hist_Bins + HSG_Aux_Bins);
				if (Grad_Mag > 0) {
					feature_vec[adr3 + b1] += 2;
					feature_vec[adr3 + b0] += 1;
					feature_vec[adr3 + b2] += 1;
				}
				//				frame_Hists[adr3 + HSG_Hist_Bins] += luv_U;
				temp[v3*No_horiz_hists + h3] += luv_U;
			}
			if ((h2 < No_horiz_hists - 1) & (v2 >= 0)) {
				adr2 = v2*No_horiz_hists*(HSG_Hist_Bins + HSG_Aux_Bins) + h2*(HSG_Hist_Bins + HSG_Aux_Bins);
				if (Grad_Mag > 0) {
					feature_vec[adr2 + b1] += 2;
					feature_vec[adr2 + b0] += 1;
					feature_vec[adr2 + b2] += 1;
				}
				//				frame_Hists[adr2 + HSG_Hist_Bins] += luv_U;
				temp[v2*No_horiz_hists + h2] += luv_U;
			}
			if ((h1 >= 0) & (v1 >= 0)) {
				adr1 = v1*No_horiz_hists*(HSG_Hist_Bins + HSG_Aux_Bins) + h1*(HSG_Hist_Bins + HSG_Aux_Bins);
				if (Grad_Mag > 0) {
					feature_vec[adr1 + b1] += 2;
					feature_vec[adr1 + b0] += 1;
					feature_vec[adr1 + b2] += 1;
				}
				//				frame_Hists[adr1 + HSG_Hist_Bins] += luv_U;
				temp[v1*No_horiz_hists + h1] += luv_U;
			}
		}//);
	}//);

	for (int i = 0; i < No_vert_hists; i++) {
		for (int j = 0; j < No_horiz_hists; j++) {
			int adr = i*No_horiz_hists*(HSG_Hist_Bins + HSG_Aux_Bins) + j*(HSG_Hist_Bins + HSG_Aux_Bins) + HSG_Hist_Bins;
			int U = temp[i*No_horiz_hists + j];
			feature_vec[adr] = U / (HSG_Hist_Cell_Size[0] * HSG_Hist_Cell_Size[1] * 4);
		}
	}
	//	Release allocated memory
	delete temp;
	delete Grad_Mag_;
	delete Grad_Or_; 
}

void HSG_Feature_V3(Mat frame, unsigned char * feature_vec)
{
	r0 = 7;
	c0 = 7;
	int fRows = frame.rows;
	int fCols = frame.cols;
	int No_horiz_hists = ((fCols - 2) / HSG_hist_stride[1]);
	int No_vert_hists = ((fRows - 2) / HSG_hist_stride[0]);
	unsigned char *frame_data_ptr = (unsigned char*)(frame.data);
	Mat luv;
	cvtColor(frame, luv, CV_BGR2Luv);
	unsigned char *luv_data_ptr = (unsigned char*)(luv.data);

	int* temp;
	temp = new int[No_horiz_hists*No_vert_hists];
	for (int x = 0; x < (No_horiz_hists*No_vert_hists); x++)
		temp[x] = 0;

	float Grad_Mag, Grad_Or;
	float* Grad_Mag_;
	float* Grad_Or_;
	Grad_Mag_ = new float[fRows*fCols];
	Grad_Or_ = new float[fRows*fCols];
//	Calc_Grad_Img(frame, Grad_Mag_, Grad_Or_);

	//Mat Y;
	//cvtColor(frame, Y, CV_RGB2YCrCb);
//	Calc_Grad_Img(luv, Grad_Mag_, Grad_Or_);

	//Parallel FOr shold theoretically give wrong results because feature vector could be accessed by multiple unrolled loops
	//and would lead to lesser count in the bins. BUT ironcially it gives better results!!!
	//parallel_for(size_t(1), size_t(fRows - 1), size_t(1), [&](size_t j) {
	//parallel_for(size_t(1), size_t(fCols - 1), size_t(1), [&](size_t i) {
	for (int j = r0; j < fRows - r0; j++) {			//leave boundary pixels
		for (int i = c0; i < fCols - c0; i++) {		//leave boundary pixels
			int dx[3], dy[3];
			int k, temp_mag1, temp_mag2;
			int jcols3 = j*fCols * 3;
			int cols3 = fCols * 3;
			int i3 = i * 3;
			int a1 = jcols3 + i3;

			Grad_Or = Grad_Or_[j*fCols + i];
			Grad_Mag = Grad_Mag_[j*fCols + i];

			int h1, h2, h3, h4, v1, v2, v3, v4, adr1, adr2, adr3, adr4, b0, b1, b2;
			h4 = (i - 1) / HSG_hist_stride[1];
			v4 = (j - 1) / HSG_hist_stride[0];
			h3 = h4 - 1;
			v3 = v4;
			h2 = h4;
			v2 = v4 - 1;
			h1 = h4 - 1;
			v1 = v4 - 1;

			int luv_U = luv_data_ptr[a1 + 1];

			adr4 = v4*No_horiz_hists*(HSG_Hist_Bins + HSG_Aux_Bins) + h4*(HSG_Hist_Bins + HSG_Aux_Bins);
			adr3 = v3*No_horiz_hists*(HSG_Hist_Bins + HSG_Aux_Bins) + h3*(HSG_Hist_Bins + HSG_Aux_Bins);
			adr2 = v2*No_horiz_hists*(HSG_Hist_Bins + HSG_Aux_Bins) + h2*(HSG_Hist_Bins + HSG_Aux_Bins);
			adr1 = v1*No_horiz_hists*(HSG_Hist_Bins + HSG_Aux_Bins) + h1*(HSG_Hist_Bins + HSG_Aux_Bins);

			temp[v4*No_horiz_hists + h4] += luv_U;
			temp[v3*No_horiz_hists + h3] += luv_U;
			temp[v2*No_horiz_hists + h2] += luv_U;
			temp[v1*No_horiz_hists + h1] += luv_U;

			Grad_Mag = int(Grad_Mag / HSG_GradMag_Q);
			if (Grad_Mag > 0) {
				b1 = (int)floor(Grad_Or / HSG_bin_size);
				if (b1 < 0)	b1 = HSG_Hist_Bins - 1;
				if (b1 - 1 >= 0)
					b0 = b1 - 1;
				else
					b0 = HSG_Hist_Bins - 1;
				if (b1 + 1 <= HSG_Hist_Bins - 1)
					b2 = b1 + 1;
				else
					b2 = 0;

					feature_vec[adr4 + b1] += 2;
					feature_vec[adr4 + b0] += 1;
					feature_vec[adr4 + b2] += 1;
					feature_vec[adr3 + b1] += 2;
					feature_vec[adr3 + b0] += 1;
					feature_vec[adr3 + b2] += 1;
					feature_vec[adr2 + b1] += 2;
					feature_vec[adr2 + b0] += 1;
					feature_vec[adr2 + b2] += 1;
					feature_vec[adr1 + b1] += 2;
					feature_vec[adr1 + b0] += 1;
					feature_vec[adr1 + b2] += 1;
			}
		}//);
	}//);

	for (int i = 0; i < No_vert_hists; i++) {
		for (int j = 0; j < No_horiz_hists; j++) {
			int adr = i*No_horiz_hists*(HSG_Hist_Bins + HSG_Aux_Bins) + j*(HSG_Hist_Bins + HSG_Aux_Bins) + HSG_Hist_Bins;
			int U = temp[i*No_horiz_hists + j];
			feature_vec[adr] = U / (HSG_Hist_Cell_Size[0] * HSG_Hist_Cell_Size[1] * 4);
		}
	}
	//	Release allocated memory
	delete temp;
	delete Grad_Mag_;
	delete Grad_Or_;
}

void HSG_SingleScale_Detector_Approx(Mat& frame, unsigned char* frame_Hists, int cols, int rows, vector<Rect>& BB_Rects, vector<double>& BB_Scores, double scale, vector<double>& BB_Scales) {
	int no_horiz_hists = ((cols - 2) / HSG_hist_stride[1]);
	int no_horiz_hists_half = ((cols / 2 - 2) / HSG_hist_stride[1]);
	int no_vert_hists_half = ((rows / 2 - 2) / HSG_hist_stride[0]);
	unsigned char* frame_Hists_half;
	int mem_siz_half = (HSG_Hist_Bins + HSG_Aux_Bins)*(no_horiz_hists_half)*(no_vert_hists_half);
	frame_Hists_half = new unsigned char[mem_siz_half];			//No. of bins
	for (int i = 0; i < mem_siz_half; i++)
		frame_Hists_half[i] = 0;		//initialize hists to zero*/

	unsigned char h1, h2, h3, h4;
	for (int hi = 0; hi < no_vert_hists_half - 1; hi++) {
		for (int hj = 0; hj < no_horiz_hists_half; hj++) {
			for (int hk = 0; hk < (HSG_Hist_Bins + HSG_Aux_Bins); hk++) {
				h1 = frame_Hists[2 * hi*(no_horiz_hists)*(HSG_Hist_Bins + HSG_Aux_Bins) + 2 * hj*(HSG_Hist_Bins + HSG_Aux_Bins) + hk];
				h2 = frame_Hists[2 * hi*(no_horiz_hists)*(HSG_Hist_Bins + HSG_Aux_Bins) + 2 * (hj + 1)*(HSG_Hist_Bins + HSG_Aux_Bins) + hk];
				h3 = frame_Hists[2 * (hi + 1)*(no_horiz_hists)*(HSG_Hist_Bins + HSG_Aux_Bins) + 2 * hj*(HSG_Hist_Bins + HSG_Aux_Bins) + hk];
				h4 = frame_Hists[2 * (hi + 1)*(no_horiz_hists)*(HSG_Hist_Bins + HSG_Aux_Bins) + 2 * (hj + 1)*(HSG_Hist_Bins + HSG_Aux_Bins) + hk];
				frame_Hists_half[hi*(no_horiz_hists_half)*(HSG_Hist_Bins + HSG_Aux_Bins) + hj*(HSG_Hist_Bins + HSG_Aux_Bins) + hk] = (int)((float)(((int)h1 + (int)h2 + (int)h3 + (int)h4) / 4) * 1);
			}
		}
	}

#ifndef Training_Mode
	//	HSG_Window_Scan(frame, frame_Hists_half, rows / 2, cols / 2, BB_Rects, BB_Scores, scale / 2, BB_Scales, training_mode, WinSize_1, Kernel_LUT_1, b_1, 0.5);
#endif
	delete frame_Hists_half;
}

