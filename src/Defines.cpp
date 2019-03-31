#include "Defines.h"
/*
								Global Variables and Defines
*/
string MainDir = "E:\\RnD\\Current_Projects\\Musawwir\\Frameworks\\SW\\Dataset\\Person\\";

//Training Specific Variables
	bool training_mode = false;
	//string curr_img;								//current training image file name 
	unsigned long int hard_ex_cnt_stage = 0;		//Number of hard examples added in this stage
	int hard_ex_cnt_total = 0;							//Number of hard examples added in total
	int persistent_hard_ex_cnt = 0;						//Number of hard examples already in the pool

	//SVM Model
	int Model_Kernel_Type = 0;		//0-Linear, 4-HIK
	//const char* svm_file_name = "SVM_Data\\PD_HIK_INRIA(Q_24_NoAvg_c=0.0001).svm";
	const char* svm_liner_wts_file_name = "E:\\RnD\\Current_Projects\\Musawwir\\Frameworks\\SW\\Musawwir_SW_VS\\Object_Detector\\src\\SVM_Linear_Wts.h";
	const char* svm_kernel_lut_file_name = "src\\SVM_HIK_LUT.h";
	int sv_num;
	double** supvec;
	double* alpha;
	double* asv;
	double b;
	double** Kernel_LUT;		//LUT for discrete Kernel evaluation

	const char* svm_file_name = "SVM_Data\\HOG_HIK.svm";//{ 102, 36 }
	double C_Pos = 10000e-6;
	double C_Neg = 10000e-6;
	double Soft_SVM_C = C_Neg;
	double Soft_SVM_C_ratio = C_Pos / C_Neg;

	int Kernel_LUT_Q = 64;		//	Size of Kernel LUT, maximum possible value of integer feature is Q-1 (0 to Q-1)
	float feature_scale = 1;

	//Training & Classification Dataset variables
	int examples_count = 0;
	double** examples = 0;
	double* labels = 0;

	timeb t_start, t_end;
	float fps = 0, t_elapsed = 0;
	//Annotations Statistics
	int height_hist[10000];

	long int stats[7] = { 0,0,0,0,0,0,0 };

