#include "Dataset_Trainings.h"
#include "HSG_HIK.h"
#include"hog_mod.h"

//Global Variables
//vector<double> FP_scores;
//vector<double> TP_scores;

/*			Delete '1' out of every 'step' files in the directory 'DirPath'			*/
void Delete_Files_In_A_Dir(string DirPath, int step) {
	char SaveCurrentPath[FILENAME_MAX];
	_getcwd(SaveCurrentPath, sizeof(SaveCurrentPath));	//Store current directory path
	if (_chdir(DirPath.c_str()) == -1) {				//Change to DirPath
		printf("\n\n%s\n..directory does not exist!!!", DirPath.c_str());
		return;
	}
	_finddata_t fd;
	intptr_t hFile = _findfirst("*.*", &fd);
	printf("\n\n\tDeleting every 1 out of %d existing files ...",step);
	do {
		if (fd.name != string(".") && fd.name != string("..")) remove(fd.name);
		for (int c = 0; c < step-1;c++)		//Skip files (==step-1)
			_findnext(hFile, &fd);
	} while (_findnext(hFile, &fd) == 0);
	_findclose(hFile);

	_chdir(SaveCurrentPath);		//Change back to the original working directory
}

void Create_Neg_Training_Images(string Neg_Full_Img_Src_Dir, string Neg_Train_Img_Dst_Dir, Size Img_Size, int examples_cnt) {
	char SaveCurrentPath[FILENAME_MAX];
	_getcwd(SaveCurrentPath, sizeof(SaveCurrentPath));	//Store current directory path
	vector<string> Src_Img_File_Names;
	intptr_t hFile;
	_finddata_t fd;
	if (_chdir(Neg_Train_Img_Dst_Dir.c_str()) == -1) {
		printf("\n\nNegative Training Images Directory not found!!! Creating new\n\n\t %s", Neg_Train_Img_Dst_Dir.c_str());
		mkdir(Neg_Train_Img_Dst_Dir.c_str());
	}
	Delete_Files_In_A_Dir(Neg_Train_Img_Dst_Dir, 1);

	if (_chdir(Neg_Full_Img_Src_Dir.c_str()) == -1) {
		printf("\n\nNegative Training Images Source Directory not found!!! Press Enter to continue.\n\n\t %s", Neg_Full_Img_Src_Dir.c_str());	getchar();	return;
	}

	hFile = _findfirst("*.*", &fd);
	do {
		if (fd.name != string(".") && fd.name != string("..")) {
			Src_Img_File_Names.push_back(fd.name);
		}
	} while (_findnext(hFile, &fd) == 0);
	_findclose(hFile);
	/*printf("\n%d files found", Src_Img_File_Names.size());
	for (int x = 0; x < Src_Img_File_Names.size(); x++) { printf("\n[%d] %s", x, Src_Img_File_Names[x].c_str()); }getchar();*//*Mat img = imread(Src_Img_File_Names[x]);		imshow("img",img);		waitKey(1);	}*/

	printf("\n\n\tCreating Negative Examples...");
	srand(time(NULL));
	while (examples_cnt > 0) {
		int rnd_fname = rand() % Src_Img_File_Names.size();
		Mat img = imread(Src_Img_File_Names[rnd_fname]);
		float rnd_sc = (float)(25 + rand() % 100) / 100;
		resize(img, img, Size(0, 0), rnd_sc, rnd_sc);
		Rect crop;
		Rect rect_mat(0, 0, img.cols, img.rows);
		crop.width = Img_Size.width;				crop.height = Img_Size.height;
		crop.x = rand() % 200 + rand() % 200;		crop.y = rand() % 200 + rand() % 200;
		if ((crop & rect_mat) == crop) {
			string curr_img = Src_Img_File_Names[rnd_fname].substr(0, Src_Img_File_Names[rnd_fname].find_last_of("."));
			char output_img_file_path[FILENAME_MAX];
			sprintf(output_img_file_path, "%s%s_S%04d_%04d_%04d.png", Neg_Train_Img_Dst_Dir.c_str(), curr_img.c_str(), (int)(rnd_sc * 1000), crop.x, crop.y);
			imwrite(output_img_file_path, img(crop));
			examples_cnt--;
		}
	}
	_chdir(SaveCurrentPath);		//Change back to the original working directory
}

void Create_Pos_Training_Images(string Pos_Full_Img_Src_Dir, string Pos_Train_Img_Dst_Dir, float Scale, Rect crop) {
	char SaveCurrentPath[FILENAME_MAX];
	_getcwd(SaveCurrentPath, sizeof(SaveCurrentPath));	//Store current directory path
	intptr_t hFile;
	_finddata_t fd;
	if (_chdir(Pos_Train_Img_Dst_Dir.c_str()) == -1) {
		printf("\n\Positive Training Images Directory not found!!! Creating new\n\n\t %s", Pos_Train_Img_Dst_Dir.c_str());
		mkdir(Pos_Train_Img_Dst_Dir.c_str());
	}
	Delete_Files_In_A_Dir(Pos_Train_Img_Dst_Dir, 1);

	if (_chdir(Pos_Full_Img_Src_Dir.c_str()) == -1) {
		printf("\n\nPositive Training Images Source Directory not found!!! Press Enter to continue.\n\n\t %s", Pos_Full_Img_Src_Dir.c_str());	getchar();	return;
	}
	hFile = _findfirst("*.*", &fd);
	printf("\n\n\tCreating Positive Examples...");
	do {
		if (fd.name != string(".") && fd.name != string("..")){
			Mat img = imread(fd.name);
			resize(img, img, Size(0, 0), Scale, Scale, INTER_CUBIC);
			char output_img_file_path[FILENAME_MAX];
			String curr_img = fd.name;
			curr_img = curr_img.substr(0, curr_img.find_last_of("."));
			sprintf(output_img_file_path, "%s%s_%02d_%02d.png", Pos_Train_Img_Dst_Dir.c_str(), curr_img.c_str(), crop.x, crop.y);
			imwrite(output_img_file_path, img(crop));
		}
	} while (_findnext(hFile, &fd) == 0);
	_findclose(hFile);
	_chdir(SaveCurrentPath);		//Change back to the original working directory
}


string INRIA_Train_Neg = "INRIA\\Train\\neg\\";
string INRIA_Train_Pos = "INRIA\\96X160H96\\Train\\pos\\";

void Training_Master(Musawwir_Obj_Detector &MOD){
	string Pos_Full_Img_Src_Dir = MainDir + INRIA_Train_Pos;
	string Neg_Full_Img_Src_Dir = MainDir + INRIA_Train_Neg;
	string Pos_Train_Img_Dst_Dir = MainDir + "train\\pos\\";
	string Neg_Train_Img_Dst_Dir = MainDir + "train\\neg\\";
	string SVM_Model_FilePath = MainDir + "SVM_Data\\HOG_HIK_Q32.svm";

//	Prepare training examples directories (Positive & Negative)
//	Delete_Files_In_A_Dir(Neg_Train_Img_Dst_Dir, 2);
//	Delete_Files_In_A_Dir(Neg_Train_Img_Dst_Dir, 2);
//	Create_Neg_Training_Images(Neg_Full_Img_Src_Dir, Neg_Train_Img_Dst_Dir, Size(64,128), 500);
//	Rect crop(16, 16, 64, 128);//(8, 8, 32, 64);// (24, 20, 48, 120); //
//	Create_Pos_Training_Images(Pos_Full_Img_Src_Dir, Pos_Train_Img_Dst_Dir, 1, crop);

//	SVM_Training(MOD, Pos_Train_Img_Dst_Dir, Neg_Train_Img_Dst_Dir, SVM_Model_FilePath);
	MOD.Fill_SVM_Wts(SVM_Model_FilePath);
//	Purge_Examples(MOD, Neg_Train_Img_Dst_Dir, 0, 0);
	return;

	float SVM_Score_Th = 2.0;
	float Scale_Oct = 2;
	int Spatial_Stride = 32;
	float init_scale = 2;	
	int hard_ex_cnt;

/*	SVM_Score_Th = 0;
	Scale_Oct = 8;
	Spatial_Stride = 8;
	goto bootstrap_stg_4;*/

bootstrap_stg_1:
	for (; SVM_Score_Th >= -0.001; SVM_Score_Th = SVM_Score_Th - 0.2) {
		printf("\n\n\tSVM Threshold for current stage: %.2f\n", SVM_Score_Th);
		hard_ex_cnt = Hard_Negative_Mining(MOD, init_scale, Scale_Oct, Size(Spatial_Stride, Spatial_Stride), SVM_Score_Th, Neg_Full_Img_Src_Dir, Neg_Train_Img_Dst_Dir);
		printf("\n%d hard examples mined.", hard_ex_cnt);
		if (hard_ex_cnt > 20) {
			SVM_Training(MOD, Pos_Train_Img_Dst_Dir, Neg_Train_Img_Dst_Dir, SVM_Model_FilePath);
			MOD.Fill_SVM_Wts(SVM_Model_FilePath);
			hard_ex_cnt = 0;
		}
	}
	if (hard_ex_cnt > 0) {
		SVM_Training(MOD, Pos_Train_Img_Dst_Dir, Neg_Train_Img_Dst_Dir, SVM_Model_FilePath);
		MOD.Fill_SVM_Wts(SVM_Model_FilePath);
	}

bootstrap_stg_2:
	SVM_Score_Th = 0;
	for (Scale_Oct = 3; Scale_Oct <= 8.001; Scale_Oct = Scale_Oct + 1) {
		printf("\n\n\tScale stride for current stage: %.4f =2^(1/%.2f)\n", pow(2, (1 / Scale_Oct)), Scale_Oct);
		hard_ex_cnt = Hard_Negative_Mining(MOD, init_scale, Scale_Oct, Size(Spatial_Stride, Spatial_Stride), SVM_Score_Th, Neg_Full_Img_Src_Dir, Neg_Train_Img_Dst_Dir);
		printf("\n%d hard examples mined.", hard_ex_cnt);
		if (hard_ex_cnt > 20) {
			SVM_Training(MOD, Pos_Train_Img_Dst_Dir, Neg_Train_Img_Dst_Dir, SVM_Model_FilePath);
			MOD.Fill_SVM_Wts(SVM_Model_FilePath);
			hard_ex_cnt = 0;
		}
	}
	if (hard_ex_cnt > 0) {
		SVM_Training(MOD, Pos_Train_Img_Dst_Dir, Neg_Train_Img_Dst_Dir, SVM_Model_FilePath);
		MOD.Fill_SVM_Wts(SVM_Model_FilePath);
	}

bootstrap_stg_3:
	Scale_Oct = 8;
	for (unsigned char stride = 24; stride >= 8; stride = stride - 8) {
		printf("\n\n\tStride for current stage: %d\n", stride);
		hard_ex_cnt = Hard_Negative_Mining(MOD, init_scale, Scale_Oct, Size(Spatial_Stride, Spatial_Stride), SVM_Score_Th, Neg_Full_Img_Src_Dir, Neg_Train_Img_Dst_Dir);
		printf("\n%d hard examples mined.", hard_ex_cnt);
		if (hard_ex_cnt > 20) {
			SVM_Training(MOD, Pos_Train_Img_Dst_Dir, Neg_Train_Img_Dst_Dir, SVM_Model_FilePath);
			MOD.Fill_SVM_Wts(SVM_Model_FilePath);
			hard_ex_cnt = 0;
		}
	}

	if (hard_ex_cnt > 0) {
		SVM_Training(MOD, Pos_Train_Img_Dst_Dir, Neg_Train_Img_Dst_Dir, SVM_Model_FilePath);
		MOD.Fill_SVM_Wts(SVM_Model_FilePath);
	}
	return;

bootstrap_stg_4:
	int bootstrap_cycle = 3;
	while (bootstrap_cycle>0) {
		hard_ex_cnt = Hard_Negative_Mining(MOD, init_scale, Scale_Oct, Size(Spatial_Stride, Spatial_Stride), SVM_Score_Th, Neg_Full_Img_Src_Dir, Neg_Train_Img_Dst_Dir);
		printf("\n%d hard examples mined.", hard_ex_cnt);
		if (hard_ex_cnt < 10) {
			SVM_Training(MOD, Pos_Train_Img_Dst_Dir, Neg_Train_Img_Dst_Dir, SVM_Model_FilePath);
			break;
		}
		SVM_Training(MOD, Pos_Train_Img_Dst_Dir, Neg_Train_Img_Dst_Dir, SVM_Model_FilePath);
		MOD.Fill_SVM_Wts(SVM_Model_FilePath);
		bootstrap_cycle--;
	}
}

void LUT_CSV(void) {
	char file_path[200];
	sprintf(file_path, "%sSVM_Data\\Kernel_LUT_7.csv", MainDir.c_str());
	ofstream svm_file;
	svm_file.open(file_path);

	Kernel_LUT_Q = 64;
	int feature_size = 3630;
	extern float Kernel_LUT_7[64][4096];

	printf("\nWriting %d x %d LUT...", Kernel_LUT_Q, feature_size);
	for (int k = 0; k<Kernel_LUT_Q; k++) {
		for (int j = 0; j<feature_size; j++) {
			svm_file << Kernel_LUT_7[k][j] << ", ";
		}
		svm_file << "\n";
	}
	svm_file.close();
}

int Hard_Negative_Mining(Musawwir_Obj_Detector &MOD, float Init_Scale, float Scale_Stride, Size Spatial_Stride, float Score_Threshold, string Neg_Full_Img_Src_Dir, string Neg_Train_Img_Dst_Dir) {
	Mat frame_orig, frame_proc;
	vector<Point> Object_Locations;
	vector<double> Object_Scores;
	Size Detector_WinSize;
	switch (MOD.Active_Detector_Obj)
	{
	case Musawwir_Obj_Detector::HOG_OpenCV_Mod:
		Detector_WinSize = (*MOD.HOG_OpenCV_Mod_Obj).winSize;
		break;
	case Musawwir_Obj_Detector::HOG_OpenCV:
		Detector_WinSize = (*MOD.HOG_OpenCV_Obj).winSize;
		break;
	case Musawwir_Obj_Detector::HSG:
		break;
	default:
		break;
	}
	
	int file_cnt = 0;
	int neg_ex_cnt = 0;
	intptr_t hFile;
	_finddata_t fd;
	_chdir(Neg_Full_Img_Src_Dir.c_str());
	hFile = _findfirst("*.*", &fd);
	do {
		if (fd.name != string(".") && fd.name != string("..")) {
			printf("\r%.2f %%", (double)file_cnt / 12.18);
			file_cnt++;
			//sprintf("\nReading file ->%s", fd.name);
			frame_orig = imread(fd.name);
			String curr_img = fd.name;
			curr_img = curr_img.substr(0, curr_img.find_last_of("."));
			
			//		Run Detector on various scales, Cannot run multi-scale detector because we need to crop window that was detected at a particular scale
			float scale = Init_Scale;
			resize(frame_orig, frame_proc, Size(0, 0), scale, scale);			
			while ((frame_proc.rows >= Detector_WinSize.height)&(frame_proc.cols >= Detector_WinSize.width)) {
				switch (MOD.Active_Detector_Obj)
				{
				case Musawwir_Obj_Detector::HOG_OpenCV_Mod:
					(*MOD.HOG_OpenCV_Mod_Obj).detect(frame_proc, Object_Locations, Object_Scores, Score_Threshold, Spatial_Stride, Size(0, 0));
					break;
				case Musawwir_Obj_Detector::HOG_OpenCV:
					(*MOD.HOG_OpenCV_Obj).detect(frame_proc, Object_Locations, Object_Scores, Score_Threshold, Spatial_Stride, Size(0, 0));
					break;
				case Musawwir_Obj_Detector::HSG:
					break;
				default:
					break;
				}
				//		Save detected Hard Negative window 
				for (int i = 0; i < Object_Locations.size(); i++)
				{
					Rect r(Object_Locations[i].x, Object_Locations[i].y, Detector_WinSize.width, Detector_WinSize.height);
					char output_img_file_path[FILENAME_MAX];
					sprintf(output_img_file_path, "%s%s_S%04d_%04d_%04d.png", Neg_Train_Img_Dst_Dir.c_str(), curr_img.c_str(), (int)(scale * 1000), r.x, r.y);
					imwrite(output_img_file_path, frame_proc(r));
				}
				neg_ex_cnt += Object_Locations.size();
				scale = scale / pow(2, 1 / Scale_Stride);
				resize(frame_orig, frame_proc, Size(0, 0), scale, scale);
			}
		}
	} while (_findnext(hFile, &fd) == 0);
	_findclose(hFile);
	return neg_ex_cnt;
}

void Purge_Examples(Musawwir_Obj_Detector &MOD, string Neg_Train_Img_Dst_Dir, int count, bool type) {
	Mat frame;
	vector<double> List_Scores;
	Size Detector_WinSize;
	switch (MOD.Active_Detector_Obj)
	{
	case Musawwir_Obj_Detector::HOG_OpenCV_Mod:
		Detector_WinSize = (*MOD.HOG_OpenCV_Mod_Obj).winSize;
		break;
	case Musawwir_Obj_Detector::HOG_OpenCV:
		Detector_WinSize = (*MOD.HOG_OpenCV_Obj).winSize;
		break;
	case Musawwir_Obj_Detector::HSG:
		break;
	default:
		break;
	}

	int ex_cnt = 0;
	intptr_t hFile;
	_finddata_t fd;
	_chdir(Neg_Train_Img_Dst_Dir.c_str());
	hFile = _findfirst("*.*", &fd);
	do {
		if (fd.name != string(".") && fd.name != string("..")) {
//			printf("\n%s...", fd.name);
			frame = imread(fd.name);
			String curr_img = fd.name;
			curr_img = curr_img.substr(0, curr_img.find_last_of("."));
			vector<double> Object_Scores;
			vector<Point> Object_Locations;
			//		Run Detector on each training example
			switch (MOD.Active_Detector_Obj)
			{
				case Musawwir_Obj_Detector::HOG_OpenCV_Mod:
					(*MOD.HOG_OpenCV_Mod_Obj).detect(frame, Object_Locations, Object_Scores, -20, Size(8, 8), Size(0, 0));
					break;
				case Musawwir_Obj_Detector::HOG_OpenCV:
					(*MOD.HOG_OpenCV_Obj).detect(frame, Object_Locations, Object_Scores, -20, Size(8, 8), Size(0, 0));
					break;
				case Musawwir_Obj_Detector::HSG:
					break;
				default:
					break;
			}
				//		Store scores in the global score list 
				for (int i = 0; i < Object_Locations.size(); i++)
				{
//					printf("\t Found %.2f at (%d,%d)", Object_Scores[i], Object_Locations[i].x, Object_Locations[i].y);
					List_Scores.push_back(Object_Scores[i]);
				}
				ex_cnt++;
		}
	} while (_findnext(hFile, &fd) == 0);
	sort(List_Scores.begin(), List_Scores.end());
	double threshold = List_Scores[count];

	for (int i = 0; i < List_Scores.size(); i++)
		printf("\n%.2f", List_Scores[i]);
	printf("\n %d examples found", ex_cnt);

	hFile = _findfirst("*.*", &fd);
	do {
		if (fd.name != string(".") && fd.name != string("..")) {
			frame = imread(fd.name);
			String curr_img = fd.name;
			curr_img = curr_img.substr(0, curr_img.find_last_of("."));
			vector<double> Object_Scores;
			vector<Point> Object_Locations;
			//		Run Detector on each training example
			switch (MOD.Active_Detector_Obj)
			{
			case Musawwir_Obj_Detector::HOG_OpenCV_Mod:
				(*MOD.HOG_OpenCV_Mod_Obj).detect(frame, Object_Locations, Object_Scores, -20, Size(8, 8), Size(0, 0));
				break;
			case Musawwir_Obj_Detector::HOG_OpenCV:
				(*MOD.HOG_OpenCV_Obj).detect(frame, Object_Locations, Object_Scores, -20, Size(8, 8), Size(0, 0));
				break;
			case Musawwir_Obj_Detector::HSG:
				break;
			default:
				break;
			}
			//		Store scores in the global score list 
			for (int i = 0; i < Object_Locations.size(); i++)
			{
				if(Object_Scores[i]<threshold)
					remove(fd.name);
			}
		}
	} while (_findnext(hFile, &fd) == 0);
	_findclose(hFile);
}

void read_annotations(vector<Rect>& Ann_Rects, char* ann_file_path, int min_height, float min_overlap) {
	ifstream ann_file;
	char read_line[50];
	char label[10];
	int occl;
	Rect r, v;
	ann_file.open(ann_file_path);

	ann_file.getline(read_line, 100);	//Read first line
										//	cout << endl << read_line;

	while (ann_file.getline(read_line, 100)) {//Read annotations
		label[6] = 0;
		sprintf(label, "%s", strtok(read_line, " "));
		//printf("\nLabel = %s", label);			//person, person-fa, person?, people (Ignore all but the first type)
		if (label[6] == '?') continue;
		if (label[2] == 'o') continue;
		if (label[6] == '-') continue;
		r.x = stoi(strtok(NULL, " "), NULL, 10);
		r.y = stoi(strtok(NULL, " "), NULL, 10);
		r.width = stoi(strtok(NULL, " "), NULL, 10);
		r.height = stoi(strtok(NULL, " "), NULL, 10);
		occl = stoi(strtok(NULL, " "), NULL, 10);
		v.x = stoi(strtok(NULL, " "), NULL, 10);
		v.y = stoi(strtok(NULL, " "), NULL, 10);
		v.width = stoi(strtok(NULL, " "), NULL, 10);
		v.height = stoi(strtok(NULL, " "), NULL, 10);
		//printf("\nRect = %d, %d, %d, %d", r.x, r.y, r.width, r.height);
		if (r.height >= min_height) {
			if (occl == 1) {
				Rect ol = r&v;		//overlap between gt and visible ann
				if (ol.area() >= min_overlap*r.area())
					Ann_Rects.push_back(r);
			}
			else
				Ann_Rects.push_back(r);
		}
	}
}