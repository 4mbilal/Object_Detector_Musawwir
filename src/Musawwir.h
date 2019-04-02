#ifndef _MUSAWWIR__
#define __MUSAWWIR____
#include "Defines.h"

struct Musawwir_Obj_Detector
{
public:
	enum { Face, Pedestrian, Car };
	enum { Ped_INRIA, Ped_ETH, Ped_TUDBrussels, Ped_USA, Ped_USA_Train, Ped_USA_Test };
	enum { HSG, HOG_OpenCV, HOG_OpenCV_Mod };
	enum { SVM_Linear, SVM_Poly, SVM_RBF, SVM_Sigmoid, SVM_HIK };		//These numbers correspond to those listed in SVM-Light library
	HOGDescriptor_Mod *HOG_OpenCV_Mod_Obj;
	HOGDescriptor *HOG_OpenCV_Obj;
	HSGDetector *HSG_Obj;
	int Active_Detector_Obj;
	int Active_Obj_Type;
	int SVM_Type;
	float Detection_Threshold;
	float Scale_Stride;
	Size Spatial_Stride;
	Size Padding;
	float** SVM_Wts_LUT;
	vector<float> SVM_Wts_vec;
	int Feature_Vec_Length;
	float SVM_bias;
	int Kernel_LUT_Q;
	int Dataset;

	//Dataset evaluations
	float SVM_training_error;
	bool load_show_annotations=0;
	vector<double> FP_scores;
	vector<double> TP_scores;
	int frame_cnt = 0;
	int total_detections = 0;
	int total_objects = 0;
	int* ftp;
	float* fp;
	float* tp;
	float lamr = 0;

	virtual void Fill_SVM_Wts(string SVM_Model_FilePath);
	virtual void Fill_SVM_Wts_LUT();
	virtual void Process_Test_Datasets(string Exp);
};

#endif