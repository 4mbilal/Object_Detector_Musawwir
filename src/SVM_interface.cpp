#include "SVM_interface.h"
#include "SVM_Linear_Wts.h"

void SVM_Training(Musawwir_Obj_Detector &MOD, string TrainPosDirPath, string TrainNegDirPath, string SVM_Model_FilePath)
{
	printf("\n\n.....................................................................");
	printf("\n\n\t->Training Started...");
	Mat img1;
	Mat img2;
	//img2.create(win_R, win_C, CV_8UC3);
	/*for (int i = 0; i<10000; i++)
	height_hist[i] = 0;*/

	intptr_t hFile;
	_finddata_t fd;
	int i, j;
	int cp = 0, cn = 0;	//positive and negative examples count
	vector<float> FeatureVector;
	int FeatureVectorLength = MOD.Feature_Vec_Length;

	//------------------Determine the total number of +ve and -ve example images---------------------------------
	if (_chdir(TrainPosDirPath.c_str()) == -1) {
		printf("\n%s Directory not found", TrainPosDirPath.c_str());
		getchar();
		return;
	}

	hFile = _findfirst("*.*", &fd);
	do {
		if (fd.name != string(".") && fd.name != string("..")) {
			//printf("\n%s",fd.name);
			cp++;
		}
	} while (_findnext(hFile, &fd) == 0);
	_findclose(hFile);

	if (_chdir(TrainNegDirPath.c_str()) == -1) {
		printf("\n%s Directory not found", TrainNegDirPath.c_str());
		getchar();
		return;
	}
	hFile = _findfirst("*.*", &fd);
	do {
		if (fd.name != string(".") && fd.name != string("..")) {
			//printf("\n%s", fd.name);
			cn++;
		}
	} while (_findnext(hFile, &fd) == 0);
	_findclose(hFile);
	printf("\n\t\t%d Positive and %d Negative examples found.\n", cp, cn);
	//	getchar();

	//------------------Create the examples data structures---------------------------------
	examples_count = cp + cn;

	examples = new double*[examples_count];
	for (i = 0; i < examples_count; i++)
		examples[i] = new double[FeatureVectorLength];

	labels = new double[examples_count];
	//------------------Populate the examples data structures---------------------------------
	j = 0;
	_chdir(TrainPosDirPath.c_str());
	hFile = _findfirst("*.*", &fd);
	do {
		if (fd.name != string(".") && fd.name != string("..")) {
			img2 = imread(fd.name);
			vector<Point> locations;
			static const Size trainingPadding_HOG = Size(0, 0);
			static const Size winStride_HOG = Size(8, 8);
			//compute function in HOG implementation does not input cellsize. It has to be modified to allow this setting.
			FeatureVector.clear();
			switch (MOD.Active_Detector_Obj)
			{
			case Musawwir_Obj_Detector::HOG_OpenCV_Mod:
				(*MOD.HOG_OpenCV_Mod_Obj).compute(img2, FeatureVector, winStride_HOG, trainingPadding_HOG, locations);
				FeatureVectorLength = (*MOD.HOG_OpenCV_Mod_Obj).getDescriptorSize();
				break;
			case Musawwir_Obj_Detector::HOG_OpenCV:
				(*MOD.HOG_OpenCV_Obj).compute(img2, FeatureVector, winStride_HOG, trainingPadding_HOG, locations);
				FeatureVectorLength = (*MOD.HOG_OpenCV_Obj).getDescriptorSize();
				break;
			case Musawwir_Obj_Detector::HSG:
				break;
			default:
				break;
			}
			for (i = 0; i < FeatureVectorLength; i++) {
				examples[j][i] = FeatureVector[i];// int(FeatureVector[i] * MOD.Kernel_LUT_Q);//
				//height_hist[int(featureVector_HOG[i]*10000)]++;
			}

			labels[j] = 1;		//1 for positive examples
			j++;
		}
		printf("\r\t->Calculating features .. %d%% completed", 100 * j / (cp + cn));
	} while (_findnext(hFile, &fd) == 0);
	_findclose(hFile);


	_chdir(TrainNegDirPath.c_str());
	hFile = _findfirst("*.*", &fd);
	do {
		if (fd.name != string(".") && fd.name != string("..")) {
			img2 = imread(fd.name);
			vector<Point> locations;
			static const Size trainingPadding_HOG = Size(0, 0);
			static const Size winStride_HOG = Size(8, 8);
			FeatureVector.clear();
			switch (MOD.Active_Detector_Obj)
			{
			case Musawwir_Obj_Detector::HOG_OpenCV_Mod:
				(*MOD.HOG_OpenCV_Mod_Obj).compute(img2, FeatureVector, winStride_HOG, trainingPadding_HOG, locations);
				break;
			case Musawwir_Obj_Detector::HOG_OpenCV:
				(*MOD.HOG_OpenCV_Obj).compute(img2, FeatureVector, winStride_HOG, trainingPadding_HOG, locations);
				break;
			case Musawwir_Obj_Detector::HSG:
				break;
			default:
				break;
			}
			for (i = 0; i<FeatureVectorLength; i++) {
				examples[j][i] = FeatureVector[i];// int(FeatureVector[i] * MOD.Kernel_LUT_Q);//
				//height_hist[int(featureVector_HOG[i] * 10000)]++;
			}
			labels[j] = -1;		//-1 for negative examples
			j++;
		}
		printf("\r\t->Calculating features .. %d%% completed", 100 * j / (cp + cn));
	} while (_findnext(hFile, &fd) == 0);
	_findclose(hFile);

	printf("\n\t->SVM Training started");
	printf("\nExample Count = %d", examples_count);
	printf("\nFeatureVectorLength = %d", FeatureVectorLength);

	//----------------------------------------------------------------------------------------
	//Write features on disk for evaluation by Matlab
	/*ofstream feature_file("E:\\RnD\\Current_Projects\\Musawwir\\Frameworks\\SW\\Dataset\\Person\\train\\features_dump.dat", std::ios::out | std::ios::binary);
	float temp;
	temp = examples_count;
	feature_file.write((const char *)&temp, sizeof(float));
	temp = FeatureVectorLength;
	feature_file.write((const char *)&temp, sizeof(float));
	for (int j = 0; j < examples_count; j++) {
		for (int k = 0; k < FeatureVectorLength; k++) {
			temp = examples[j][k];
			feature_file.write((const char *)&temp, sizeof(float));
		}
		temp = labels[j];
		feature_file.write((const char *)&temp, sizeof(float));
	}
	feature_file.close();*/
	//----------------------------------------------------------------------------------------
	//SVM_Param_Grid_Search(SVM_Model_FilePath.c_str(), examples, labels, FeatureVectorLength, examples_count);
	MOD.SVM_training_error = SVM_Train(SVM_Model_FilePath.c_str(), examples, labels, FeatureVectorLength, examples_count);// , win_R, win_C, grayscale);
	printf("\n\t->SVM Training finished!");
	//printf("\n\t\t%.2f", MOD.SVM_training_error);
	printf("\n\n.....................................................................");
	//-------------- Release memory ---------------------------------------------------------------

	for (i = 0; i<examples_count; i++)
		delete examples[i];
	delete examples;

	delete labels;
	/*ofstream stats_file;
	stats_file.open("E:\\RnD\\Current_Projects\\Musawwir\\Frameworks\\SW\\Dataset\\Person\\SVM_Data\\feature_stats.csv");
	for (int x = 0; x < 10000; x++){
	stats_file << x << ",\t" << height_hist[x]<<endl;
	}
	stats_file.close();*/
}

/*void SVM_Param_Grid_Search(const char* svm_file_name, double** examples, double* labels, long totwords, long totdoc) {
	//This function is not very useful for the following reasons.
	//1- Xi-Alpha is not reliable at all as the error metric.
	//2- LOOC is reliable and gives some promising optimal point but is TOO time consuming.
	//3- A better way is to write the features to disk and evaluate in Matlab (Matlab_SVM_Interface.m)
	float e1 = 0, e2 = 0, e3 = 0;
	Soft_SVM_C = 20000e-6;
	e1 = SVM_Train(svm_file_name, examples, labels, totwords, totdoc);
	Soft_SVM_C = 1000e-6;
	e2 = SVM_Train(svm_file_name, examples, labels, totwords, totdoc);
	Soft_SVM_C = 100e-6;
	e3 = SVM_Train(svm_file_name, examples, labels, totwords, totdoc);

	printf("\n\n\tE1 = %.2f, E2 = %.2f, E3 = %.2f", e1, e2, e3);
}*/

void Musawwir_Obj_Detector::Fill_SVM_Wts(string SVM_Model_FilePath) {
	switch (Active_Detector_Obj) {
	case HOG_OpenCV:
		printf("\nLoading HOG_OpenCV Model..");
		SVM_Wts_vec.clear();
		Feature_Vec_Length = (*HOG_OpenCV_Obj).getDescriptorSize();
		load_SVM(SVM_Model_FilePath.c_str(), Feature_Vec_Length, sv_num, supvec, alpha, b, asv);
		for (int x = 0; x < Feature_Vec_Length; x++)
			SVM_Wts_vec.push_back(asv[x]);
		SVM_Wts_vec.push_back(-b);
		(*HOG_OpenCV_Obj).setSVMDetector(SVM_Wts_vec);
		break;

	case HOG_OpenCV_Mod:
		if (SVM_Type == SVM_Linear) {
			if ((*HOG_OpenCV_Mod_Obj).SVM_Eval_Method == (*HOG_OpenCV_Mod_Obj).SVM_Dot_Product) {
				printf("\nLoading HOG_OpenCV_Mod Model..");
				SVM_Wts_vec.clear();
				Feature_Vec_Length = (*HOG_OpenCV_Mod_Obj).getDescriptorSize();
				load_SVM(SVM_Model_FilePath.c_str(), Feature_Vec_Length, sv_num, supvec, alpha, b, asv);
				for (int x = 0; x < Feature_Vec_Length; x++)
					SVM_Wts_vec.push_back(asv[x]);
				SVM_Wts_vec.push_back(-b);
				(*HOG_OpenCV_Mod_Obj).setSVMDetector(SVM_Wts_vec);
			}
			if ((*HOG_OpenCV_Mod_Obj).SVM_Eval_Method == (*HOG_OpenCV_Mod_Obj).SVM_LUT) {
				Feature_Vec_Length = (*HOG_OpenCV_Mod_Obj).getDescriptorSize();
				load_SVM(SVM_Model_FilePath.c_str(), Feature_Vec_Length, sv_num, supvec, alpha, b, asv);
				SVM_Wts_LUT = new float*[Kernel_LUT_Q];
				for (int x = 0; x<Kernel_LUT_Q; x++)
					SVM_Wts_LUT[x] = new float[Feature_Vec_Length];
				Fill_SVM_Wts_LUT();
				(*HOG_OpenCV_Mod_Obj).svmDetectorLUT = SVM_Wts_LUT;
				(*HOG_OpenCV_Mod_Obj).svmDetectorLUT_Q = Kernel_LUT_Q;
				(*HOG_OpenCV_Mod_Obj).svmDetectorBias = -b;
			}
		}
		break;
	case HSG:
		break;
	default:
		break;
	}
}

void Musawwir_Obj_Detector::Fill_SVM_Wts_LUT() {
	printf("\n\tBuilding SVM Weights LUT....");
	float temp;
	printf("\n");
	for (int k = 0; k<Kernel_LUT_Q; k++) {
		printf("\r\t\t->%d%% completed", 100 * k / Kernel_LUT_Q);
		for (int j = 0; j<Feature_Vec_Length; j++) {
			temp = 0;
			for (int i = 0; i<sv_num - 1; i++) {
				//Feature value [0 <-> 1], Q_Val [0 <-> Kernel_LUT_Q]. Feature value should not be exactly '1' because LUT can hold b/w 0 and Kernel_LUT_Q-1
				int Q_Val = int(supvec[i][j]);// *Kernel_LUT_Q);
				if (Q_Val>Kernel_LUT_Q - 1)		//Saturate
					Q_Val = Kernel_LUT_Q - 1;

				float feature_value = float(k);// / float(Kernel_LUT_Q);
				//feature_value = pow(feature_value, 4);

				//Linear
				//temp += supvec[i][j] * alpha[i] * (feature_value);

				//HIK
				if (Q_Val<k)
					temp += supvec[i][j] * alpha[i];
				else
					temp += (feature_value)*alpha[i];
				//temp += (float(k))*alpha[i];
				
				//SAD
				//temp += (128 - abs((supvec[i][j]) - ((float(k)*feature_scale))))*alpha[i];
				//SSE
				/*float ss = supvec[i][j] - float(k)*feature_scale;
				ss = sqrt(ss*ss);
				temp += (512 - ss)*alpha[i];*/
			}
			SVM_Wts_LUT[k][j] = temp;
		}
	}
	printf("\n\n");
	//-------------------------------------- Write Kernel_LUT to file--------------------
	ofstream file;
	double temp2;
	char buff[25] = "";
	file.open(svm_kernel_lut_file_name);

	file << "\nint WinSize_1[2] = { " << 102 << ", " << 36 << " };";

	file << "\n\nfloat Kernel_LUT_1[" << Kernel_LUT_Q << "][" << Feature_Vec_Length << "] = \n{\n";
	for (int k = 0; k<Kernel_LUT_Q; k++) {
		file << "\t{";
		for (int j = 0; j < Feature_Vec_Length; j++) {
			temp2 = SVM_Wts_LUT[k][j];
			sprintf(buff, "%.6f", temp2);
			file << buff;
			if (j<Feature_Vec_Length - 1)
				file << ", ";
		}
		if (k<Kernel_LUT_Q - 1)
			file << "},\n";
		else
			file << "}\n";
	}
	file << "};";
	file << "\n\nfloat b_1 = ";
	sprintf(buff, "%.6f", double(b));
	file << buff;
	file << ";";

	file.close();

}


/*void LUT_CSV(){
	char svm_file_path[100];
	sprintf(svm_file_path, "%s%s", Dataset_Path, svm_file_name);
	load_SVM(svm_file_path, feature_size, sv_num, supvec, alpha, b, asv);

	Kernel_LUT = new double*[Kernel_LUT_Q];
	for (int x = 0; x<Kernel_LUT_Q; x++)
		Kernel_LUT[x] = new double[feature_size];

	printf("\n\tBuilding Kernel LUT....");
	Build_Kernel_LUT();

	//filter_LP_LUT();

	ofstream svm_file;
	svm_file.open("C:\\HIK_LUT.csv");

	printf("\nWriting %d x %d LUT...", Kernel_LUT_Q, feature_size);
	for (int k = 0; k<Kernel_LUT_Q; k++){
		for (int j = 0; j<feature_size; j++){
			svm_file << Kernel_LUT[k][j] << ", ";
		}
		svm_file << "\n";
	}
	svm_file.close();
}*/

void Build_Kernel_LUT(){
	float temp;
	int Q_Val;
	int feature_size = 3630;
	printf("\n");
	for(int k=0;k<Kernel_LUT_Q;k++){
	printf("\r\t\t->%d%% completed", 100*k/Kernel_LUT_Q);
	int feature_size = 3630;
		for(int j=0;j<feature_size;j++){
			temp = 0;
			for(int i=0;i<sv_num-1;i++){  

				//if(supvec[i][j]>0.5)
					//supvec[i][j] = 0.5;

				Q_Val = int(supvec[i][j]*Kernel_LUT_Q);		//Feature value [0 <-> 1], Q_Val [0 <-> Kernel_LUT_Q]. Feature value should not be exactly '1' because LUT can hold b/w 0 and Kernel_LUT_Q-1
				//Q_Val = int(supvec[i][j] / feature_scale);
				if(Q_Val>Kernel_LUT_Q-1)
					Q_Val=Kernel_LUT_Q-1;

				//Linear
				//temp += supvec[i][j] * alpha[i] * (float(k)*feature_scale);
				//temp += supvec[i][j] * alpha[i] * (float(k)/float(Kernel_LUT_Q));

				//HIK
				if(Q_Val<k)
					temp += supvec[i][j] * alpha[i];
				else
					temp += (float(k) / float(Kernel_LUT_Q))*alpha[i];
					//temp += (float(k)*feature_scale)*alpha[i];
				
				//SAD
				//temp += (128 - abs((supvec[i][j]) - ((float(k)*feature_scale))))*alpha[i];
				//SSE
				/*float ss = supvec[i][j] - float(k)*feature_scale;
				ss = sqrt(ss*ss);
				temp += (512 - ss)*alpha[i];*/
			}

			Kernel_LUT[k][j] = temp;
		}
	}
	printf("\n\n");
//-------------------------------------- Write Kernel_LUT to file--------------------
	ofstream file;
	double temp2;
	char buff[25] = "";
	file.open(svm_kernel_lut_file_name);

	file << "\nint WinSize_1[2] = { " << 102 << ", " << 36 << " };";

	file << "\n\nfloat Kernel_LUT_1[" << Kernel_LUT_Q << "][" << feature_size << "] = \n{\n";
	for (int k = 0; k<Kernel_LUT_Q; k++){
		file << "\t{";
		for (int j = 0; j < feature_size; j++){
			temp2 = Kernel_LUT[k][j];
			sprintf(buff, "%.6f", temp2);
			file << buff;
			if (j<feature_size-1)
				file << ", ";
		}
		if (k<Kernel_LUT_Q - 1)
			file << "},\n";
		else
			file << "}\n";
	}
	file << "};";
	file << "\n\nfloat b_1 = ";
	sprintf(buff, "%.6f", double(b));
	file << buff;
	file << ";";

	file.close();

}


float Kernel_LUT_Dist(float* ex){
	float dist = 0;
	int Q_Val;
	int feature_size = 3630;
		for(int j=0;j<feature_size;j++){

				//if(ex[j]>0.5)
					//ex[j] = 0.5;

			//Q_Val = int(ex[j]*Kernel_LUT_Q);
			Q_Val = int(ex[j]);
			//Q_Val = int(Q_Val/4)*4;

			if(Q_Val>Kernel_LUT_Q-1)
				Q_Val=Kernel_LUT_Q-1;
			dist += Kernel_LUT[Q_Val][j];
			//dist += Kernel_LUT2[Q_Val][j];
		}
		
	return(dist-b);
}

double Kernel_Dist(float *Data){
	double dist=0;
	double sum = 0;
	double k = 0;
	int feature_size = 3630;
	//HIK
	for(int j=0;j<feature_size;j++){
		k = 0;
		for(int i=0;i<sv_num-1;i++) {
			if(supvec[i][j]<Data[j])
				k += supvec[i][j]*alpha[i];	
			else
				k += Data[j]*alpha[i];	
		}
			dist+= k;
	}
	
	
	//AD
	/*
	for(int j=0;j<feature_size;j++){
		k = 0;
		for(int i=0;i<sv_num-1;i++) {
			k += (511-abs(supvec[i][j]-Data[j]))*alpha[i];	
		}
			dist+= k;
	}
	*/
	
	/*
	//SSE
	for(int i=0;i<sv_num-1;i++) {
		k = 0;
		for(int j=0;j<feature_size;j++){
			k += (abs(supvec[i][j]-Data[j])*abs(supvec[i][j]-Data[j]));	
		}
			//dist += (723-sqrt(k))*alpha[i];

			dist += (exp(-0.05*sqrt(k)))*alpha[i];		//RBF
		//dist += k*alpha[i];
	}
	*/

	//Linear
	/*
	for(int j=0;j<feature_size;j++){
		k = 0;
		for(int i=0;i<sv_num-1;i++) {
			k += supvec[i][j]*Data[j]*alpha[i];	
		}
			dist+= k;
	}
	*/
	//printf("\nDist = %f",dist-b);
  return(dist-b);
}

void filter_LP_LUT(){
	double f;
	int n = 16;
	double ** Kernel_LUT_LPF = new double*[Kernel_LUT_Q];
	int feature_size = 3630;
	for(int x=0;x<Kernel_LUT_Q;x++)
		Kernel_LUT_LPF[x] = new double[feature_size];

	for(int j=0;j<feature_size;j++){
		for(int k=0;k<Kernel_LUT_Q;k++){
			Kernel_LUT_LPF[k][j] = Kernel_LUT[k][j];
		}
	}
	
	for(int j=0;j<feature_size;j++){
		for(int k=n;k<Kernel_LUT_Q-n;k++){
			f = 0;
			for(int l=-n;l<=n;l++){
				f += Kernel_LUT[k+l][j];
			}
			f = f/double(2*n + 1);
			Kernel_LUT_LPF[k][j] = f;
		}
	}
	/*
	double alpha = 0.45;
	for(int j=0;j<feature_size;j++){
		f = Kernel_LUT[0][j];
		for(int k=0;k<Kernel_LUT_Q;k++){
			Kernel_LUT_LPF[k][j] = Kernel_LUT[k][j]*(1-alpha) + alpha*f;
			f = Kernel_LUT_LPF[k][j];
		}
	}
	*/
	for (int j = 0; j<feature_size; j++){
		for (int k = 0; k<Kernel_LUT_Q; k++){
			Kernel_LUT[k][j] = Kernel_LUT_LPF[k][j];
		}
	}

	//-------------------------------------- Write Kernel_LUT to file--------------------
	ofstream file;
	double temp2;
	char buff[25] = "";
	file.open(svm_kernel_lut_file_name);
	file << "\n\nfloat Kernel_LUT_2[" << Kernel_LUT_Q << "][" << feature_size << "] = \n{\n";
	for (int k = 0; k<Kernel_LUT_Q; k++){
		file << "\t{";
		for (int j = 0; j < feature_size; j++){
			temp2 = Kernel_LUT[k][j];
			sprintf(buff, "%.6f", temp2);
			file << buff;
			if (j<feature_size - 1)
				file << ", ";
		}
		if (k<Kernel_LUT_Q - 1)
			file << "},\n";
		else
			file << "}\n";
	}
	file << "};";
	file << "\n\nfloat b_2 = ";
	sprintf(buff, "%.6f", double(b));
	file << buff;
	file << ";";

	file.close();

	delete Kernel_LUT;
	Kernel_LUT = Kernel_LUT_LPF;
}

void support_vec_stats(){
	int sv_num;
	double** supvec;
	double* alpha;
	double* asv;
	double b;
	float hist[1000];
	int feature_size = 3630;
	load_SVM(svm_file_name, feature_size, sv_num, supvec, alpha, b, asv);// , win_R, win_C, grayscale);

	for (int i = 0; i<1000; i++)
		hist[i] = 0;

	for (int i = 0; i<sv_num - 1; i++){
		/*
		for(int j=0;j<1000;j++)
		hist[j] = 0;
		*/
		for (int j = 0; j<feature_size; j++){
			hist[int(supvec[i][j] * 1000)] += 0.01 / sv_num;
		}
	}
	plot(hist);


	for (int i = 0; i < sv_num - 1; i++)
		delete supvec[i];
	delete supvec;
	delete alpha;
	delete asv;
}



void save_SVM(MODEL *model, const char* filename)// , int win_R, int win_C, char grayscale)
{
	mWORD * mw;
	int temp;
	double temp2;
//	char abc;
	char buff[25]="";

	ofstream file;
	file.open (filename,ios::binary);
	int feature_size = model->totwords;

	double** data = new double*[model->sv_num];
	for(int i=0;i<model->sv_num;i++)
		data[i] = new double[feature_size];

	file.write(reinterpret_cast<char*>( &feature_size ), sizeof feature_size);		//write feature size in each sv
	temp = int(model->sv_num);
	file.write(reinterpret_cast<char*>( &temp ), sizeof temp);		//write number of sv's

	for(int i=1;i<model->sv_num;i++) {		//iterate through all sv's
		mw = model->supvec[i]->fvec->words;
		for(int j=0;j<feature_size;j++){
			temp2 = double(mw->weight);
			file.write(reinterpret_cast<char*>( &temp2 ), sizeof temp2);		//write elements within current sv
			mw++;
		}
	}
 
	for(int i=1;i<model->sv_num;i++) {		
		temp2 = double(model->alpha[i]);
		file.write(reinterpret_cast<char*>( &temp2 ), sizeof temp2);		//write elements within alpha
	}

		temp2 = double(model->b);
		file.write(reinterpret_cast<char*>( &temp2 ), sizeof temp2);		//write b
		//printf("\n VAlue of b = %f", temp2);
//-------------------------------------------------------------------------------------------------
	for(int i=1;i<model->sv_num;i++) {		//iterate through all sv's
		mw = model->supvec[i]->fvec->words;
		for(int j=0;j<feature_size;j++){
			data[i][j] = double(mw->weight);
			mw++;
		}
	}
		for(int j=0;j<feature_size;j++){
			temp2 = 0;
			for(int i=1;i<model->sv_num;i++) {
				temp2 += data[i][j]*model->alpha[i];
			}
		file.write(reinterpret_cast<char*>( &temp2 ), sizeof temp2);		//write elements of ASV
	}
//-------------------------------------------------------------------------------------------------
	//file.write(reinterpret_cast<char*>( &win_R ), sizeof win_R);				//write window height
	//file.write(reinterpret_cast<char*>( &win_C ), sizeof win_C);				//write window width
	//file.write(reinterpret_cast<char*>( &grayscale ), sizeof grayscale);		//write weether model uses grayscale or color

	file.close();
//---------------------------------------Write ASV and b in a C File---------------------------------
	file.open(svm_liner_wts_file_name);
//		file.open ("G:\\SVM.h");
		file<<"\n\nfloat SVM_f ["<<feature_size<<"]= {";
		for(int j=0;j<feature_size;j++){
			temp2 = 0;
			for(int i=1;i<model->sv_num;i++) {
				temp2 += data[i][j]*model->alpha[i];
			}
			sprintf(buff,"%.08ff",temp2);
			file<<buff;
			if(j<feature_size-1)
				file<<", ";
		}
		file<<"};";
		file<<"\n\nint SVM_i ["<<feature_size<<"]= {";
		for(int j=0;j<feature_size;j++){
			temp2 = 0;
			for(int i=1;i<model->sv_num;i++) {
				temp2 += data[i][j]*model->alpha[i];
			}
			file<<int(temp2*100000);
			if(j<feature_size-1)
				file<<", ";
		}
		file<<"};";
		file<<"\n\nfloat b_f = ";
		sprintf(buff,"%.20f",double(model->b));
		file<<buff;
		file<<";";
		file<<"\n\nint b_i = ";
		file<<int(model->b*100000);
		file<<";";
		file<<"\n\nint feature_size_i = "<<feature_size<<";";
		file<<"\n\n//Number of Support Vectors = "<<(model->sv_num)-1;

		file.close();

/*
#define Print_SVs
#ifdef Print_SVs
		file.open ("G:\\SV.txt");

#endif
		*/
	for(int i=0;i<model->sv_num;i++)
		delete data[i];
	delete data;
}

void load_SVM(const char* filename, int &feature_size, int &sv_num, double**& supvec, double*& alpha, double &b, double*& ASV)//, int &win_R, int &win_C, char &grayscale)
{
	int temp;
	double temp2;
	printf("\n\tLoading SVM Model....%s\n", filename);

	ifstream file;
	file.open(filename, ios::binary);
	if (!file){
		printf("\n\n\tSVM Model not found!!!!!");
		getchar();
		exit(0);
	}

	file.read( reinterpret_cast<char*>( &feature_size ), sizeof feature_size );		//read feature size in each sv
	file.read( reinterpret_cast<char*>( &temp ), sizeof temp );						//read number of sv's
	sv_num = temp;

	supvec = new double* [sv_num-1];		// -1 because 1st sv is always zero, why-?
	for(int i = 0; i < sv_num-1; i++)
	    supvec[i] = new double[feature_size];
	alpha = new double [sv_num-1];		// -1 because 1st element in alpah is always zero, why-?


	for(int i=0;i<sv_num-1;i++) {		//iterate through all sv's
		for(int j=0;j<feature_size;j++){
			file.read( reinterpret_cast<char*>( &temp2 ), sizeof temp2 );		//read feature size in each sv
			supvec[i][j] = temp2;												//read elements within current sv
		}
	}
 
	for(int i=0;i<sv_num-1;i++) {		
			file.read( reinterpret_cast<char*>( &temp2 ), sizeof temp2 );
			alpha[i] = temp2;		//read elements within alpha
	}
	
	file.read( reinterpret_cast<char*>( &temp2 ), sizeof temp2 );	//read 'b'
	b = temp2;

	ASV = new double[feature_size];
	for(int i=0;i<feature_size;i++) {		
			file.read( reinterpret_cast<char*>( &temp2 ), sizeof temp2 );
			ASV[i] = temp2;		//read elements within ASV
	}
	
	//file.read( reinterpret_cast<char*>( &win_R ), sizeof win_R );	//read win_R
	//file.read( reinterpret_cast<char*>( &win_C ), sizeof win_C );	//read win_C
	//file.read( reinterpret_cast<char*>( &grayscale ), sizeof grayscale );	//read grayscale
	
  file.close();

  /*
  char buff[25]="";

	ofstream fileo;
	fileo.open ("G:\\Alpha.txt",ios::binary);
	for(int i=0;i<sv_num-1;i++) {		
		fileo<<alpha[i]<<"\n";		//read elements within alpha
	}
  fileo.close();
  */
}


int svm_dist_i(int *Data)
{
	int dist=0;
	int feature_size = 3630;
	for(int j=0;j<feature_size;j++)
		dist+= SVM_i[j]*Data[j];

	dist = dist/1000;

return(dist-b_i);
}

float svm_dist_f(float *Data){
	float dist=0;
	/*
	//Attempt to prune non-significant svm coefficients
	int cnt = 0;
	for(int j=0;j<feature_size_i;j++){
		if((abs(SVM_f[j])<0.035)&(abs(SVM_f[j])>0.03)){
			SVM_f[j] = 0;
			cnt++;
		}
	}
	*/
	//printf("\n\n\nZeroed = %d",cnt);
	//getchar();
	int feature_size = 3630;
	for(int j=0;j<feature_size;j++)
		dist+= SVM_f[j]*Data[j];

return(dist-b_f);
}

float svm_dist_f(double *Data){	//overloaded function for double inputs
	float dist=0;

	/*
	//Attempt to prune non-significant svm coefficients
	int cnt = 0;
	for(int j=0;j<feature_size_i;j++){
		if((abs(SVM_f[j])<0.03)){//&(abs(SVM_f[j])>0.03)){
			SVM_f[j] = 0;
			cnt++;
		}
	}
	//printf("\nZeroed = %d",cnt);
	//getchar();
	*/
	int feature_size;
	for(int j=0;j<feature_size;j++)
		dist+= SVM_f[j]*Data[j];

	return(dist-b_f);
}

float SVM_Train(const char* svm_file_name, double** examples, double* labels, long totwords,long totdoc)//, int win_R, int win_C, char grayscale)
{
	DOC **docs;  /* training examples */
	double *alpha_in=NULL;		//What is this?
	KERNEL_CACHE *kernel_cache;
	LEARN_PARM learn_parm;
	KERNEL_PARM kernel_parm;
	MODEL model;
	kernel_cache_statistic = 0;

	set_input_parameters(&learn_parm,&kernel_parm);

	ConvToDOC(examples, &docs, totwords, totdoc);
	
	//Release examples memory
	/*
	for(int i=0;i<totdoc;i++)
		 delete examples[i];
	delete examples;
	*/
	if(kernel_parm.kernel_type == LINEAR) /* don't need the cache */
		kernel_cache=NULL;
	else 
		kernel_cache=kernel_cache_init(totdoc,learn_parm.kernel_cache_size);

		/* Always get a new kernel cache. It is not possible to use the
		same cache for two different training runs */
	printf("\n\nStarting learning....");

	svm_learn_classification(docs,labels,totdoc,totwords,&learn_parm,&kernel_parm,kernel_cache,&model,alpha_in);

	if(kernel_cache) {
		/* Free the memory used for the cache. */
		kernel_cache_cleanup(kernel_cache);
	}

	//write_model("G:\\svm.txt", &model);								//original SVM complete model (ASCII)
	save_SVM(&model, svm_file_name);// , win_R, win_C, grayscale);		//relevant svm values (Binary)
	float training_error = model.xa_error;

	//Free memory
	for (int i = 0; i < totdoc; i++) {
		free_example((docs)[i], 1);
	}
	return training_error;
}

void ConvToDOC(double** examples, DOC ***docs, long int totwords, long int totdoc)
{
  int i,j;
  int index;
  int rows, cols;
//  double *yvals;
  double *data;
  mWORD *words;

  /* retrieve the rows and columns */
  rows = totdoc;
  cols = totwords;
  /* allocate memory for the DOC rows */
  (*docs) = (DOC **)my_malloc(sizeof(DOC *) * rows);

  /* allocate a single buffer in memory for the words (hold n columns) */
  words = (mWORD *)my_malloc(sizeof(mWORD)*(cols+1));

  /* for each row, create a corresponding vector of *DOC and store it */
  for (i = 0; i < rows; i++) {
    SVECTOR *fvec = NULL;
	data = examples[i];

	  for (j = 0; j < cols; j++)
		{
		  index = (rows * j) + i;
		  (words[j]).wnum=(j+1);  /* assign the word number  */
//		  (words[j]).weight=(CFLOAT)data[index];
		  (words[j]).weight=(CFLOAT)data[j];
		}
	  /* assign the last word to be 0 (null) to note the end of the list */
	  (words[j]).wnum = 0;

		/* create the intermediate structure (svector in svm_common.c) */
		fvec = create_svector(words,"",1.0);


//		for (j = 0; j < 2; j++) {		//Why 3 times?
			(*docs)[i] = create_example(i, 0, 0, 1.0, fvec);
//		}

  }
  
  free(words);
}


/*---------------------------------------------------------------------------*/

void set_input_parameters(LEARN_PARM *learn_parm,KERNEL_PARM *kernel_parm)
{
	/* set default */
	strcpy (learn_parm->predfile, "trans_predictions");
	strcpy (learn_parm->alphafile, "");

	learn_parm->biased_hyperplane=1;
	learn_parm->sharedslack=0;
	learn_parm->remove_inconsistent=0;
	learn_parm->skip_final_opt_check=0;
	learn_parm->svm_maxqpsize=10;
	learn_parm->svm_newvarsinqp=0;
	learn_parm->svm_iter_to_shrink=2;//-9999;	//for LINEAR case
	learn_parm->maxiter=100000;
	learn_parm->kernel_cache_size = 500;
	learn_parm->eps=1.0;//0.1;
	learn_parm->transduction_posratio=-1.0;
	learn_parm->svm_costratio = Soft_SVM_C_ratio;// 1.0;
	learn_parm->svm_costratio_unlab=1.0;
	learn_parm->svm_unlabbound=1E-5;
	learn_parm->epsilon_crit=0.001;
	learn_parm->epsilon_a=1E-15;
	learn_parm->compute_loo=0;
	learn_parm->rho=1.0;
	learn_parm->xa_depth=0;
	learn_parm->type=CLASSIFICATION;		//set by arguments in matlab file
	learn_parm->svm_c=Soft_SVM_C;					//set by arguments in matlab file
	kernel_parm->kernel_type=Model_Kernel_Type;				//Linear(0), Kernel (4)
	kernel_parm->coef_lin=1;
	kernel_parm->coef_const=1;
	kernel_parm->poly_degree=3;
	kernel_parm->rbf_gamma=1.0;
	strcpy(kernel_parm->custom,"empty");
	if(kernel_parm->kernel_type==4)
		printf("\n\n\tSVM Type = HIK");
	else
		printf("\n\n\tSVM Type = Linear");
}


//------------------------------------------------------------------------------------------
//						OLD SVM Interface Functions
//------------------------------------------------------------------------------------------



void freeModelMem(MODEL *model)
{
	long i;

	if(model->supvec) {
		for(i=1;i<model->sv_num;i++) {
			if (model->supvec[i])
				free_example(model->supvec[i],1);
		}
	}
}


double svm_dist_orig(double *Data, MODEL *model, DOC **docs, mWORD *words, int feature_size)
{
  int i;
//  double *yvals;
  int rows = 1;   //hard coded, single example to be classified
  int cols = feature_size; //each example contains 216 elements in original dowloaded HOG face detector
  
  /* for each row, create a corresponding vector of *DOC and store it */
  for (i = 0; i < rows; i++) {
    SVECTOR *fvec = NULL;
//    int j;

	/* parse and copy the given mxData into a mWORD array words */
    dat2words(i, Data, words, feature_size);

	/* create the intermediate structure (svector in svm_common.c) */
    fvec = create_svector(words,"",1.0);


	//for (j = 0; j < 2; j++) {
		(docs)[i] = create_example(i, 0, 0, 1.0, fvec);
	//}

  }
  
  return classify_example(model, docs[0]);
}

double svm_dist_simple_1(double *Data, MODEL *model, int feature_size)
{
	double dist=0;
	double sum = 0;
	double k = 0;
	mWORD * mw;

	for(int i=1;i<model->sv_num;i++) {  
		mw = model->supvec[i]->fvec->words;
		k = 0;
		for(int j=0;j<feature_size;j++){
			k = k + mw->weight*Data[j];
			mw++;
		}
			dist+= k*model->alpha[i];
	}
 
  return(dist-model->b);
}



int dat2words(int row, double *data, mWORD *words, int feature_size)
{
  int i;
  int cols,rows;
  int index;
  
  cols = feature_size;
  rows = 1;

  for (i = 0; i < cols; i++)
    {
      //index = computeOffset(rows, cols, row, i);
	  index = i + row;

      (words[i]).wnum=(i+1);  /* assign the word number  */
      (words[i]).weight=(CFLOAT)data[index];

    }

  (words[i]).wnum = 0;
	
  return 0;
}

int computeOffset(int numRows, int numCols, int i, int j)
{
  int index = (numRows * j) + i;
  return index;
}



