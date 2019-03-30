#include "SVM_Demo.h"


//#define SVM_KERNEL_LINEAR
#define SVM_KERNEL_HIK
//#define SVM_KERNEL_RBF

void SVM_Demo(void){
	Vec3b green(0, 255, 0), red(0, 0, 255), black(0, 0, 0), white(255, 255, 255);
	const int examples_count = 500;
	feature_size = 2;
	double labels[examples_count];
	double **trainingData = new double*[examples_count];
	double scale = 1;
	double dist = 0;
	int width = 512, height = 512;
	Mat image = Mat::ones(height, width, CV_8UC3);
	float Data[2];
	int d = 128;


	for (int i = 0; i<examples_count; i++)
		trainingData[i] = new double[feature_size];

	for (int i = 0; i < examples_count / 2; i++){
		trainingData[i][0] = i * d;
		trainingData[i][1] = int(140 - (0.05*i*i * d) + (0.005*i*i*i * d) / 2);
		labels[i] = 1.0;
	}
	for (int i = 0; i<examples_count/2; i++){
		trainingData[i + examples_count / 2][0] = i * d;
		trainingData[i + examples_count / 2][1] = int(60 - (0.05*i*i * d) + (0.005*i*i*i * d) / 2);
		labels[i + examples_count / 2] = -1.0;
	}

	/*
	for (int i = 0; i<examples_count / 2; i++){
		trainingData[i][0] = (int(32 * (float(rand()) / float(RAND_MAX) - 0.5)) + 128)*scale;
		trainingData[i][1] = (int(32 * (float(rand()) / float(RAND_MAX) - 0.5)) + 128)*scale;
		//printf("\nPositive Example[%d]-> x = %f, y = %f",i,trainingData[i][0],trainingData[i][1]);
		labels[i] = 1.0;
	}

	for (int i = 0; i<examples_count / 4; i++){
		trainingData[i][0] = (int(16 * (float(rand()) / float(RAND_MAX) - 0.5)) + 256)*scale;
		trainingData[i][1] = (int(16 * (float(rand()) / float(RAND_MAX) - 0.5)) + 256)*scale;
		//printf("\nPositive Example[%d]-> x = %f, y = %f",i,trainingData[i][0],trainingData[i][1]);
		labels[i] = 1.0;
	}

	for (int i = examples_count*0.4; i<examples_count / 2; i++){
		trainingData[i][0] = (int(6 * (float(rand()) / float(RAND_MAX) - 0.5)) + 435)*scale;
		trainingData[i][1] = (int(6 * (float(rand()) / float(RAND_MAX) - 0.5)) + 174)*scale;
		//printf("\nPositive Example[%d]-> x = %f, y = %f",i,trainingData[i][0],trainingData[i][1]);
		labels[i] = 1.0;
	}
	

	for (int i = examples_count / 2; i<examples_count; i++){
//		trainingData[i][0] = (int(32 * (float(rand()) / float(RAND_MAX) - 0.5)) + 384)*scale;
	//	trainingData[i][1] = (int(32 * (float(rand()) / float(RAND_MAX) - 0.5)) + 150)*scale;
		trainingData[i][0] = (int(32 * (float(rand()) / float(RAND_MAX) - 0.5)) + 30)*scale;
		trainingData[i][1] = (int(32 * (float(rand()) / float(RAND_MAX) - 0.5)) + 150)*scale;
		//printf("\nNegative Example[%d]-> x = %f, y = %f",i,trainingData[i][0],trainingData[i][1]);
		labels[i] = -1.0;
	}
	

	for (int i = examples_count*0.75; i<examples_count; i++){
		trainingData[i][0] = (int(8 * (float(rand()) / float(RAND_MAX) - 0.5)) + 400)*scale;
		trainingData[i][1] = (int(8 * (float(rand()) / float(RAND_MAX) - 0.5)) + 500)*scale;
		//printf("\nPositive Example[%d]-> x = %f, y = %f",i,trainingData[i][0],trainingData[i][1]);
		labels[i] = -1.0;
	}

	for (int i = examples_count*0.9; i<examples_count; i++){
		trainingData[i][0] = (int(2 * (float(rand()) / float(RAND_MAX) - 0.5)) + 13)*scale;
		trainingData[i][1] = (int(2 * (float(rand()) / float(RAND_MAX) - 0.5)) + 13)*scale;
		//printf("\nNegative Example[%d]-> x = %f, y = %f",i,trainingData[i][0],trainingData[i][1]);
		labels[i] = -1.0;
	}
	*/
	/*





	for(int i=examples_count*0.75;i<examples_count;i++){
	trainingData[i][0] = (int(64*(float(rand())/float(RAND_MAX)-0.5))+25)*scale;
	trainingData[i][1] = (int(64*(float(rand())/float(RAND_MAX)-0.5))+300)*scale;
	//printf("\nPositive Example[%d]-> x = %f, y = %f",i,trainingData[i][0],trainingData[i][1]);
	labels[i] = -1.0;
	}
	*/

	for (int i = 0; i < examples_count / 2; i++)
		circle(image, Point(int(trainingData[i][0] / scale), int(trainingData[i][1] / scale)), 8, Scalar(255, 255, 255), -1, 3);

	for (int i = examples_count / 2; i<examples_count; i++)
		circle(image, Point(int(trainingData[i][0] / scale), int(trainingData[i][1] / scale)), 8, Scalar(0, 0, 0), -1, 3);

#ifdef SVM_KERNEL_LINEAR
	printf("\nLinear");
	Model_Kernel_Type = 0;
#else
	printf("\nNon-Linear");
	Model_Kernel_Type = 4;
	Kernel_LUT_Q = 512;
	Soft_SVM_C = 0.1;
#endif

	char svm_file_path[150];
	sprintf(svm_file_path, "%s%s", Dataset_Path, svm_file_name);

	for (int cnt = 0; cnt < 1; cnt++){
		SVM_Train(svm_file_path, trainingData, labels, feature_size, examples_count);
		cout << endl << "iteration no"<<cnt;
		//getchar();
	}

	printf("\n\tLoading SVM Model....");
	load_SVM(svm_file_path, feature_size, sv_num, supvec, alpha, b, asv);// , win_R, win_C, grayscale);
	Kernel_LUT = new double*[Kernel_LUT_Q];
	for (int x = 0; x<Kernel_LUT_Q; x++)
		Kernel_LUT[x] = new double[feature_size];
	printf("\n\tBuilding Kernel LUT....");
	Build_Kernel_LUT();
	//filter_LP_LUT();

	printf("\n\nNo. of support vectors = %d", sv_num - 1);

	for (int i = 0; i < image.rows; ++i)
	for (int j = 0; j < image.cols; ++j)
	{
		Data[0] = float(j)*scale;
		Data[1] = float(i)*scale;
		//printf("\n%d,%d",int(Data[0]/scale),int(Data[1]/scale));
		/*
		dist = 0;
		for(int k=0;k<feature_size;k++)
		dist += ASV[k]*Data[k];
		dist = dist - b;
		*/
		//dist = Kernel_LUT_Dist(Data);
#ifdef SVM_KERNEL_LINEAR
		dist = svm_dist_f(Data);
#else
		//dist = Kernel_Dist(Data);
		dist = Kernel_LUT_Dist(Data);

#endif
		//printf("\nDist1 = %f, Dist2 = %f", dist,Kernel_Dist(Data));
		if ((image.at<Vec3b>(i, j) != white)&(image.at<Vec3b>(i, j) != black)){
			if (dist>0)
				image.at<Vec3b>(i, j) = green;
			else
				image.at<Vec3b>(i, j) = red;
		}
	}

	for (int i = 0; i < sv_num - 1; i++)
		circle(image, Point(int(supvec[i][0] / scale), int(supvec[i][1] / scale)), 4, Scalar(255, 0, 255), -1, 8);

	do{
		imshow("Output", image);
	} while (waitKey(1) != 'n');


	//getchar();


	delete supvec;
	delete alpha;
	delete asv;


	//OpenCV Latent SVM
	/*
	Mat labelsMat(examples_count, 1, CV_32FC1);
	float *label_ptr = (float*)(labelsMat.data);
	Mat trainingDataMat(examples_count, feature_size, CV_32FC1);
	float *train_ptr = (float*)(trainingDataMat.data);

	for(int i=0;i<examples_count/2;i++){
	train_ptr[i*2] = trainingData[i][0];
	train_ptr[i*2+1] = trainingData[i][1];
	label_ptr[i] = 1.0;
	}
	for(int i=examples_count/2;i<examples_count;i++){
	train_ptr[i*2] = trainingData[i][0];
	train_ptr[i*2+1] = trainingData[i][1];
	label_ptr[i] = -1.0;
	}

	CvSVMParams params;
	params.svm_type    = CvSVM::C_SVC;
	params.kernel_type = 0;
	//params.C           = 0.01;
	params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 10000, 1e-6);
	CvSVM SVM;
	SVM.train(trainingDataMat, labelsMat, Mat(), Mat(), params);

	for (int i = 0; i < image.rows; ++i)
	for (int j = 0; j < image.cols; ++j)
	{
	Mat sampleMat = (Mat_<float>(1,2) << j,i);
	float response = SVM.predict(sampleMat);

	if (response == 1)
	image.at<Vec3b>(i,j)  = green;
	else if (response == -1)
	image.at<Vec3b>(i,j)  = red;
	}

	for (int i = 0; i < examples_count/2; i++)
	circle(image,Point(int(trainingData[i][0]/scale), int(trainingData[i][1]/scale)),   8,  Scalar(255, 255, 255), -1, 3);
	for(int i=examples_count/2;i<examples_count;i++)
	circle(image,Point(int(trainingData[i][0]/scale), int(trainingData[i][1]/scale)),   8,  Scalar(0, 0, 0), -1, 3);

	//SVM.save("g:\\out.txt");
	int c     = SVM.get_support_vector_count();
	printf("\n\nNo. of Support Vectors = %d",c);

	for (int i = 0; i < c; ++i)
	{
	const float* v = SVM.get_support_vector(i);
	circle( image,  Point( (int) v[0], (int) v[1]),   4,  Scalar(255, 0, 255), -1, 8);
	}
	imshow("Output OpenCV", image);
	waitKey(1000);
	getchar();
	*/
}

void test_boost(void){
	// Training data
	float labels[11] = { 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0 };
	Mat labelsMat(11, 1, CV_32FC1, labels);

	float trainingData[11][2] = {
		{ 501, 10 }, { 508, 15 },
		{ 255, 10 }, { 501, 255 }, { 10, 501 }, { 10, 501 }, { 11, 501 }, { 9, 501 }, { 10, 502 }, { 10, 511 }, { 10, 495 } };
	Mat trainingDataMat(11, 2, CV_32FC1, trainingData);

	// Set up SVM's parameters
//	CvSVMParams params;
	//params.svm_type = CvSVM::C_SVC;
	//params.kernel_type = CvSVM::LINEAR;
	//params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

	// Train a SVM classifier
	//CvSVM SVM;
	//SVM.train(trainingDataMat, labelsMat, Mat(), Mat(), params);

	// Train a boost classifier
	Ptr<Boost> boost = Boost::create();
//	CvBoost boost;

	boost->train(trainingDataMat,
		ROW_SAMPLE,
		labelsMat);

	// Test the classifiers
	//Mat testSample1 = (Mat_<float>(1, 2) << 251, 5);
	Mat testSample2 = (Mat_<float>(1, 2) << 502, 11);

	//float svmResponse1 = SVM.predict(testSample1);
	//float svmResponse2 = SVM.predict(testSample2);
	cout << "\ntrained";

	float test[2] = { 502, 11 };
	Mat testSample1(1, 2, CV_32FC1, test);
	Mat out(1, 2, CV_32FC1, test);
	float boostResponse1  = boost->predict(testSample1, out, 1);
	//StatModel
	float boostResponse2 = 0;// boost->predict(testSample2);
	//boost->predict()

//	std::cout << "SVM:   " << svmResponse1 << " " << svmResponse2 << std::endl;
	std::cout << "BOOST: " << boostResponse1 << " " << boostResponse2 << std::endl;

	// Output:
	//  > SVM:   -1 1
	//  > BOOST: -1 1
}


void AdaBoost_Demo(void){
	//test_boost();
	//getchar();
	Vec3b green(0, 255, 0), red(0, 0, 255), black(0, 0, 0), white(255, 255, 255);
	double scale = 1;
	int width = 512, height = 512;
	Mat image = Mat::ones(height, width, CV_8UC3);

	const int examples_count = 11;
	/*
	float labels[examples_count];
	float **trainingData = new float*[examples_count];
	for (int i = 0; i<examples_count; i++)
		trainingData[i] = new float[feature_size];

	trainingData[0][0] = 100;	trainingData[0][1] = 100;	labels[0] = 1.0;
	trainingData[1][0] = 100;	trainingData[1][1] = 250;	labels[1] = 1.0;
	trainingData[2][0] = 100;	trainingData[2][1] = 400;	labels[2] = 1.0;
	trainingData[3][0] = 250;	trainingData[3][1] = 100;	labels[3] = 1.0;
	trainingData[4][0] = 400;	trainingData[4][1] = 100;	labels[4] = -1.0;
	trainingData[5][0] = 250;	trainingData[5][1] = 250;	labels[5] = 1.0;
	trainingData[6][0] = 250;	trainingData[6][1] = 400;	labels[6] = 1.0;
	trainingData[7][0] = 400;	trainingData[7][1] = 250;	labels[7] = -1.0;
	trainingData[8][0] = 400;	trainingData[8][1] = 400;	labels[8] = -1.0;
	trainingData[9][0] = 400;	trainingData[9][1] = 400;	labels[9] = -1.0;
	trainingData[10][0] = 400;	trainingData[10][1] = 400;	labels[10] = -1.0;
	*/
	
	float trainingData[11][2] = {
		{ 100, 100 }, { 100, 250 }, { 100, 400 }, { 250, 100 }, { 400, 100 }, 
		{ 250, 250 }, { 250, 400 }, { 400, 250 }, { 400, 400 }, { 50, 50 }, { 350, 350 } };
	float labels[11] = { 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0 };
	

	for (int i = 0; i < examples_count; i++){
	if (labels[i]>0)
		circle(image, Point(int(trainingData[i][0] / scale), int(trainingData[i][1] / scale)), 8, Scalar(255, 255, 255), -1, 3);
	else
		circle(image, Point(int(trainingData[i][0] / scale), int(trainingData[i][1] / scale)), 8, Scalar(0, 0, 0), -1, 3);
	}
	for (int i = 0; i < examples_count; i++){
		//trainingData[i][0] = trainingData[i][0] - 300;
		//trainingData[i][1] = trainingData[i][1] - 250;
	}
	//-----------------------------
	Mat labelsMat(11, 1, CV_32FC1, labels);
	Mat trainingDataMat(11, 2, CV_32FC1, trainingData);
	Ptr<Boost> boost = Boost::create();
	//boost->setMaxDepth(4);
	//boost->setBoostType(Boost::REAL);
	boost->setWeakCount(30);
	boost->setWeightTrimRate(0);
	//boost->setMaxDepth(2);
	//boost->setUseSurrogates(false);
	//boost->setPriors(Mat());
	boost->train(trainingDataMat, ROW_SAMPLE, labelsMat);
	boost->save("c:\\model.xml");
	cout << "trained";
//----------------------------

	for (int i = 0; i < image.rows; ++i)
	for (int j = 0; j < image.cols; ++j)
	{
		float dist = 0;
		/*
		if (i>175 & j>175)
			dist = -1;
		else
			dist = 1;
		*/
		//float test[2] = { 502, 11 };
		//Mat testSample1(1, 2, CV_32FC1, test);
		
		//float boostResponse1 = boost->predict(testSample1, out, 1);

		Mat testSample = (Mat_<float>(1, 2) << j, i);
		Mat out = (Mat_<float>(1, 2) << i, j); //(1, 2, CV_32FC1, testSample2);
		dist = boost->predict(testSample, out, 1); 
		//cout << dist<<", ";

		if ((image.at<Vec3b>(i, j) != white)&(image.at<Vec3b>(i, j) != black)){
			if (dist>0)
				image.at<Vec3b>(i, j) = green;
			else
				image.at<Vec3b>(i, j) = red;
		}
	}

	namedWindow("Output", WINDOW_NORMAL);
	imshow("Output", image);
	waitKey(0);


}

