#include "Defines.h"
#include "TwoD_Filters.h"
#include "hog_mod.h"
#ifndef __linux__
//#include "SVM_Demo.h"
#include "Dataset_Trainings.h"
#endif
#include "SVM_Wts_Matlab.h"
void float_companding_test(void);

int main(void)
{
	//Yolo_V3_test();
	printf("\n*-*-*-*-*-*-*--- Object Detection Framework ---*-*-*-*-*-*-*\n");
//	-----------------------------------------------------------------------------------------------------------------
//	Step 1- Fill the full scale detector specs 
	Musawwir_Obj_Detector Obj_Det;
	Obj_Det.Scale_Stride = 1.05;	//pow(2.0, (1.0 / 8.0)); //
	Obj_Det.Spatial_Stride = Size(8, 8);
	Obj_Det.Padding = Size(24, 24);
	Obj_Det.Detection_Threshold = 0.0;

//	-----------------------------------------------------------------------------------------------------------------
//	Step 2- Select the Feature (or the NN Framework e.g. YOLO)
	Obj_Det.Active_Detector_Obj = Obj_Det.CNN_YOLO;		//Options are HSG, HOG_OpenCV, HOG_OpenCV_Mod, CNN_YOLO

	if (Obj_Det.Active_Detector_Obj == Obj_Det.HOG_OpenCV) {
		Obj_Det.HOG_OpenCV_Obj = new HOGDescriptor;
		/*(*Obj_Det.HOG_OpenCV_Obj).winSize = Size(64,128);		
		(*Obj_Det.HOG_OpenCV_Obj).cellSize = Size(8,16);
		(*Obj_Det.HOG_OpenCV_Obj).blockSize = Size(16,32);
		(*Obj_Det.HOG_OpenCV_Obj).blockStride = Size(8,16);
		(*Obj_Det.HOG_OpenCV_Obj).nbins = 12;*/
	}
	else if (Obj_Det.Active_Detector_Obj == Obj_Det.HOG_OpenCV_Mod) {
		Obj_Det.HOG_OpenCV_Mod_Obj = new HOGDescriptor_Mod;
		/*	(*Obj_Det.HOG_OpenCV_Mod_Obj).winSize = Size(64, 128);
		//	(*Obj_Det.HOG_OpenCV_Mod_Obj).cellSize = Size(4, 4);*/
	}
	else if (Obj_Det.Active_Detector_Obj == Obj_Det.HSG) {
		Obj_Det.HSG_Obj = new HSGDetector;	//instantiate an HSG object with default settings. 
	}
	else if (Obj_Det.Active_Detector_Obj == Obj_Det.CNN_YOLO) {
		Obj_Det.CNN_YOLO_Obj = new Net;
	}

//	-----------------------------------------------------------------------------------------------------------------
//  Step 3- Select the Classifier Type  (or the NN configuration and weights)
	Obj_Det.Active_Obj_Type = Obj_Det.Pedestrian;	//Not used yet, for expansion into multiple objects

	if (Obj_Det.Active_Detector_Obj == Obj_Det.HOG_OpenCV) {
		Obj_Det.Feature_Vec_Length = (*Obj_Det.HOG_OpenCV_Obj).getDescriptorSize();
		Obj_Det.SVM_Type = Obj_Det.SVM_Linear;	//Only Linear supported by default OCV implementation
		(*Obj_Det.HOG_OpenCV_Obj).setSVMDetector(HOGDescriptor_Mod::HOG_Optimal_64_128());		//Retrained model
//		(*Obj_Det.HOG_OpenCV_Obj).setSVMDetector(Matlab_SVM_Model());	//Matlab model
		//(*Obj_Det.HOG_OpenCV_Obj).setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());	//OCV default model
	}
	else if (Obj_Det.Active_Detector_Obj == Obj_Det.HOG_OpenCV_Mod) {
		Obj_Det.Feature_Vec_Length = (*Obj_Det.HOG_OpenCV_Mod_Obj).getDescriptorSize();
		Obj_Det.SVM_Type = Obj_Det.SVM_Linear;	//Options are SVM_Linear, SVM_Poly, SVM_RBF, SVM_Sigmoid, SVM_HIK 
		(*Obj_Det.HOG_OpenCV_Mod_Obj).setSVMDetector(HOGDescriptor_Mod::HOG_Optimal_64_128());	//Retrained model
		(*Obj_Det.HOG_OpenCV_Mod_Obj).SVM_Eval_Method = HOGDescriptor_Mod::SVM_LUT;		//Options are SVM_Dot_Product, SVM_LUT
		Obj_Det.Kernel_LUT_Q = 32;		//Only valid for SVM_LUT
	}
	else if (Obj_Det.Active_Detector_Obj == Obj_Det.HSG) {
		//Default Settings
	}
	else if (Obj_Det.Active_Detector_Obj == Obj_Det.CNN_YOLO) {
		// Load the network
		String modelConfiguration = "E:\\RnD\\Current_Projects\\Musawwir\\Frameworks\\SW\\CNN\\yolov3-tiny.cfg";
		String modelWeights = "E:\\RnD\\Current_Projects\\Musawwir\\Frameworks\\SW\\CNN\\yolov3-tiny.weights";
//		String modelConfiguration = "E:\\RnD\\Current_Projects\\Musawwir\\Frameworks\\SW\\CNN\\yolov3.cfg";
//		String modelWeights = "E:\\RnD\\Current_Projects\\Musawwir\\Frameworks\\SW\\CNN\\yolov3.weights";
		(*Obj_Det.CNN_YOLO_Obj) = readNetFromDarknet(modelConfiguration, modelWeights);
		(*Obj_Det.CNN_YOLO_Obj).setPreferableBackend(DNN_BACKEND_OPENCV);
		(*Obj_Det.CNN_YOLO_Obj).setPreferableTarget(DNN_TARGET_CPU);
		//(*Obj_Det.CNN_YOLO_Obj).setPreferableTarget(DNN_TARGET_OPENCL);	//Only Intel GPU's are supported as yet. So this gives error
		// Load names of classes
		string classesFile = "E:\\RnD\\Current_Projects\\Musawwir\\Frameworks\\SW\\CNN\\coco.names";
		ifstream ifs(classesFile.c_str());
		string line;
		while (getline(ifs, line)) Obj_Det.CNN_Classes.push_back(line);
	}

//	-----------------------------------------------------------------------------------------------------------------
//  Step 4- Training
//	printf("\n\tFeature Vector Size = %d", Obj_Det.Feature_Vec_Length);
//	Training_Master(Obj_Det);
	//	goto finish;

//	-----------------------------------------------------------------------------------------------------------------
//  Step 5- Grid Search for Optimal SVM values.
//	Grid_Search_SVM_Parameters(Obj_Det);

//	-----------------------------------------------------------------------------------------------------------------
//  Step 6- Testing on Datasets etc.
//	Obj_Det.load_show_annotations = 1;
	Obj_Det.monitor_detections = 1;
	//Obj_Det.Img_Dir_Test("E:\\RnD\\Current_Projects\\Musawwir\\Frameworks\\SW\\Dataset\\Person\\Caltech\\code\\data-INRIA\\images\\set00\\V000\\");
//	Obj_Det.Img_Dir_Test("E:\\RnD\\Current_Projects\\Musawwir\\Frameworks\\SW\\Dataset\\Person\\test_images\\2\\");
//	Obj_Det.Vid_Test("E:\\RnD\\Current_Projects\\Musawwir\\Frameworks\\SW\\Dataset\\Person\\others\\2012_10_11_143333_Starting.avi");

	Obj_Det.Dataset = Obj_Det.Ped_INRIA;		//288 frames	8274(USA) 4024(USA-Test) 4250(USA-Train)
//	Obj_Det.Process_Test_Datasets("YOLO");

	Obj_Det.Dataset = Obj_Det.Ped_TUDBrussels;	//508 frames
//	Obj_Det.Process_Test_Datasets("YOLO");

	Obj_Det.Dataset = Obj_Det.Ped_ETH;			//1804 frames
//	Obj_Det.Process_Test_Datasets("YOLO");

	Obj_Det.Dataset = Obj_Det.Ped_USA;			//1804 frames
//	Obj_Det.Process_Test_Datasets("YOLO");

//	goto finish;

//	Code dumps below, cleanup when have time
//	hog_mod_test();

//	Test_HSG_HIK_Detector("");// E:\\RnD\\Current_Projects\\Musawwir\\Frameworks\\SW\\Dataset\\Person\\Caltech\\code\\data - TudBrussels\\images\\set00\\V000\\");
//	printf("\n%d", getNumThreads());
//	getchar();
//	setNumThreads(0);
//	setUseOptimized(0);
	//vid_reader();
//	
	
//	Detect();

	//AdaBoost_Demo();
	//SVM_Demo();
	//HSG_Test();
	//NMS_test();
//	LUT_CSV();
//	hog_mod_test();


//	printf("\n%d, %d, %d, %d, %d, %d, %d", stats[0], stats[1], stats[2], stats[3], stats[4], stats[5], stats[6]);
//	printf("\n%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f", 100*(float)stats[0]/(float)stats[0], 100 * (float)stats[1] / (float)stats[0], 100 * (float)stats[2] / (float)stats[0], 100 * (float)stats[3] / (float)stats[0], 100 * (float)stats[4] / (float)stats[0], 100 * (float)stats[5] / (float)stats[0], 100 * (float)stats[6] / (float)stats[0]);


/*	std::cout << cv::checkHardwareSupport(CV_CPU_SSE2);
	cout << endl << useOptimized();
	setUseOptimized(0);
	cout << endl << useOptimized();
	std::cout << cv::checkHardwareSupport(CV_CPU_SSE2);*/
finish:
	cout << "\n\n...finished!";
	getchar();	return 0;
}

//---------- Read SVM Model ----------------
/*HOGDescriptor *hog = (HOGDescriptor *)DetectorObj;
char svm_file_path[100];
sprintf(svm_file_path, "%s%s", MainDir.c_str(), svm_file_name);
vector<double> linear_model;
linear_model.clear();
load_SVM(svm_file_path, feature_size, sv_num, supvec, alpha, b, asv);
for (int x = 0; x < 1980; x++)
linear_model.push_back(asv[x]);
linear_model.push_back(-b);
(*hog).setSVMDetector(linear_model);

Kernel_LUT = new double*[Kernel_LUT_Q];
for (int x = 0; x<Kernel_LUT_Q; x++)
Kernel_LUT[x] = new double[feature_size];
printf("\n\tBuilding Kernel LUT....");
Build_Kernel_LUT();*/

//hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
//hog.setSVMDetector(HOGDescriptor_Mod::getTrainedDetector());
/*
//Model_Kernel_Type = 0;
if (Model_Kernel_Type == 4) {
load_SVM(svm_file_path, feature_size, sv_num, supvec, alpha, b, asv);// , win_R, win_C, grayscale);
Kernel_LUT = new double*[Kernel_LUT_Q];
for (int x = 0; x<Kernel_LUT_Q; x++)
Kernel_LUT[x] = new double[feature_size];
printf("\n\tBuilding Kernel LUT....");
Build_Kernel_LUT();
//filter_LP_LUT();
printf("done\n");
}*/

void float_companding_test(void) {
#define MAX   (8031.0/8192.0)
#define BIAS  (33.0/8192.0)
#define MAGIC (0x16f)
	float x;
	long  mu;
	long  *x_as_long;
	/* limit the sample between +/- max and take absolute value */
	for (float input_sample = -1; input_sample < 1; input_sample += 0.01) {
		x = input_sample;
		if ((x < -MAX) || (x > MAX))
			x = MAX;
		else if (x < 0)
			x = -x;
		/* add bias */
		x += BIAS;
		/* Extract the segment and quantization bits from the exponent and the
		mantissa.  Since we have limited the range of the signal already, the
		exponent will be well restricted. The pointer "x_as_long" is used to
		avoid unwanted type conversion.  In assembly, this is easy. */
		x_as_long = (long*)&x;
		mu = (*x_as_long >> 19) & 0xff;
		/* Unfortunately, mu needs a slight (but magical) adjustment */
		mu = MAGIC - mu;
		/* All that remains is to splice in the sign bit */
		if (input_sample >= 0)
			mu |= 0x80;

		printf("\n%.2f == %ld", input_sample, mu);
	}
}