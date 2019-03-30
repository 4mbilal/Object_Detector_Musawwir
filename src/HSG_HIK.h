#ifndef _HSG_HIK_H__
#define _HSG_HIK_H__
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <sys/timeb.h>

//Hardware Optimizations (Platform dependent)
#define Use_P4			//Enable Parallel For Loops
#define Use_SSE			//Enable SSE
//Algorithmic Optimization
#define Use_Cascade		//Enable SVM Cascading

//		Exclusive to Windows environment
#ifndef __linux__
#include <ppl.h>
//#include <xmmintrin.h>
//#include <emmintrin.h>
using namespace concurrency;
#else
#include "SSE2NEON.h"
#endif

using namespace cv;
using namespace std;


int HSG_HIK_Test(void);

struct HSGDetector
{
//private:
public:
	//Pre-Processing Parameters
	float Gamma = 0.5;
	Size pre_filter;

	//HSG Feature Parameters
	Size WinSize;											//{ 102, 36 };//{ 126, 62 };//{ 78, 30 };//{ 90, 36 };// { 54, 18 }; // 
	Size CellSize; 
	unsigned char CellStride[2];							//[0]-Rows, [1]-Cols
	unsigned char OrBins;									//Number of Gradient orientation bins
	unsigned char ColorBins;								//Number of Color histogram bins
	int GradMagQ;											//Gradient Magnitude quantization factor

	//Full Scale Detector Parameters
	float SVM_Score_Th;
	int min_object_height;									//	102;//51;//	78;// 90;//54;// 
	float ScaleStride; 
	unsigned char Frame_Padding[4];							//Padding {Top, Bottom, Left, Right} at each scale to detect objects towards the edges
	unsigned char Detect_Win_Stride[2];						//Each step is a multiple of CellStride
	unsigned char Kernel_LUT_Q;								//HIK LUT range [0 Kernel_LUT_Q]
	enum nms_methods { GroupRectangles_OCV, MeanShift_OCV, Custom_NMS, None };
	nms_methods nms; 

	//Timing Variables
	timeb t_start, t_end;

	//Default constructor
	HSGDetector() : Gamma(0.5f), pre_filter(3, 3), WinSize(36, 102), CellSize(6, 6), OrBins(9), ColorBins(1), CellStride{ 3, 3 },
		GradMagQ(16), SVM_Score_Th(0.0f), min_object_height(102), ScaleStride(pow(2, (1 / 8.f))),
		Frame_Padding{ 8, 8, 8, 8 }, Detect_Win_Stride{ 1,1 }, Kernel_LUT_Q(64), nms(Custom_NMS)
	{}

	void GammaCorrection(Mat& src, Mat& dst, float fGamma);
	void computeGradient(const Mat& img, float* Grad_Mag, unsigned char* Grad_Or);

//public:
	void HSG_Feature(Mat frame, unsigned char* frame_features);
	void MultiScale_Detector(Mat& Frame, vector<Rect>& BB_Rects, vector<double>& BB_Scores);
	void SingleScale_Detector(Mat& Frame, vector<Rect>& BB_Rects, vector<double>& BB_Scores, double scale, vector<double>& BB_Scales);
	void DrawRectangles(Mat& Frame, vector<Rect>& BB_Rects, vector<double>& BB_Scores, float SVM_Score_Th);
};

#endif