#ifndef _GlobalDefines_
#define _GlobalDefines_
//		OpenCV
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "hog_mod.h"
#include "HSG_HIK.h"
#include "Musawwir.h"


//		System
#include <fstream>
#include <sys/timeb.h>
#include <time.h>
#include <iostream>
#include <stdio.h>
#include <cmath>
#include <algorithm>

//		Exclusive to Windows environment
#ifndef __linux__
#include <Windows.h>
#include <ppl.h>
#include <direct.h>
#include <io.h>
#include <xmmintrin.h>
#include <emmintrin.h>
using namespace concurrency;
#else
#include "SSE2NEON.h"
#endif

using namespace std;
using namespace cv;


/*
	This file contains Global Variables and Defines
*/

#ifndef __linux__

#endif

//#define Training_Mode
#ifndef Training_Mode

#define Vid_Detect
#endif


	//extern string curr_img;
	extern 	unsigned long int hard_ex_cnt_stage;
	extern int hard_ex_cnt_total;
	extern int persistent_hard_ex_cnt;
	extern int height_hist[10000];
	//SVM Model parameters
	extern int Model_Kernel_Type;		//0-Linear, 4-HIK
	extern double Soft_SVM_C;			//C parameter
	extern double Soft_SVM_C_ratio;		//C ratio for +ve examples
	extern const char* svm_file_name;
	extern const char* svm_liner_wts_file_name;
	extern const char* svm_kernel_lut_file_name;
	extern int sv_num;
	extern double** supvec;
	extern double* alpha;
	extern double* asv;
	extern double b;
	extern double** Kernel_LUT;
	extern int Kernel_LUT_Q;		//	Size of Table, maximum possible value of integer feature is Q-1 (0 to Q-1)
	//DET curve file path
	extern const char* csv_file_name;
	//Training & Classification Dataset variables
	extern bool training_mode;
	extern int examples_count;
	extern double** examples;
	extern double* labels;
	//DET Curve parameters
	extern const float dist_range;
	extern const float dist_step;
	extern float** DET;
	extern timeb t_start, t_end;
	extern float fps, t_elapsed;
	extern long int stats[7];

	extern string MainDir;
#endif