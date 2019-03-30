#include "Defines.h"

void Caltech_PDollar_Format_Results(Mat& img, Musawwir_Obj_Detector* MOD, float pre_scaling, string res_path, string ann_path);

void Musawwir_Obj_Detector::Process_Test_Datasets(string Exp) {
	char buff[500];
	string dataset_name, res_file_dir, sets_dir_path, vids_dir_path, frm_path, res_file_path, ann_file_path;
	ofstream Res_File;

	int sets_cnt = 0, vids_cnt = 0, fr_no = 0, frame_cnt = 0;
	Mat	cur_frame;
	float pre_scaling = 2;

	switch (Dataset) {
	case Ped_INRIA:
		sets_cnt = 1;
		pre_scaling = 1;
		dataset_name = "data-INRIA";
		break;
	case Ped_ETH:
		dataset_name = "data-ETH";
		break;
	case Ped_TUDBrussels:
		dataset_name = "data-TudBrussels";
		break;
	case Ped_USA:
		fr_no = 29;
		dataset_name = "data-USA";
		break;
	case Ped_USA_Train:
		fr_no = 29;
		dataset_name = "data-USA";
		break;
	case Ped_USA_Test:
		sets_cnt = 6;
		fr_no = 29;
		dataset_name = "data-USA";
		break;
	default:
		break;
	}

	printf("\n\n\tProcessing \"%s\" to generate files for Pitor Dollar Matlab Toolbox (ROC plots)\n\n");

	//Create result directory root
	res_file_dir = MainDir + "Caltech\\code\\" + dataset_name + "\\res\\" + Exp;
	if (mkdir(res_file_dir.c_str()) < 0) {
		printf("\n\nCould not create results directory!!");
	}

	while (true) {	//Loop through 'sets'
		
		sprintf(buff, "%sCaltech\\code\\%s\\images\\set%02d", MainDir.c_str(), dataset_name.c_str(), sets_cnt);
		sets_dir_path = buff;

		if (_chdir(sets_dir_path.c_str()) == -1)	break;		//Looped through all available directories in numeric order 
		cout << endl << "Set " << sets_cnt;
		sprintf(buff, "%sCaltech\\code\\%s\\res\\%s\\set%02d", MainDir.c_str(), dataset_name.c_str(), Exp.c_str(), sets_cnt);
		res_file_dir = buff;
		mkdir(res_file_dir.c_str());

		while (true) {	//Loop through 'videos' in the current 'set'
			sprintf(buff, "%s\\V%03d", sets_dir_path.c_str(), vids_cnt);
			vids_dir_path = buff;

			if (_chdir(vids_dir_path.c_str()) == -1)	break;		//Looped through all available videos in numeric order starting from zero
			cout << endl << "\tVid " << vids_cnt << " :";
			sprintf(buff, "%sCaltech\\code\\%s\\res\\%s\\set%02d\\V%03d", MainDir.c_str(), dataset_name.c_str(), Exp.c_str(), sets_cnt, vids_cnt);
			res_file_dir = buff;
			mkdir(res_file_dir.c_str());

			while (true) {	//Loop through 'frames' in the current 'video'

				if ((Dataset == Ped_USA) | (Dataset == Ped_USA_Train) | (Dataset == Ped_USA_Test))
					sprintf(buff, "%s\\I%05d.jpg", vids_dir_path.c_str(), fr_no);
				else
					sprintf(buff, "%s\\I%05d.png", vids_dir_path.c_str(), fr_no);

				frm_path = buff;
				cur_frame = imread(frm_path);
				if (!cur_frame.data) break;
				else {
					sprintf(buff, "%sCaltech\\code\\%s\\res\\%s\\set%02d\\V%03d\\I%05d.txt", MainDir.c_str(), dataset_name.c_str(), Exp.c_str(), sets_cnt, vids_cnt, fr_no);
					res_file_path = buff;
					sprintf(buff, "%sCaltech\\code\\%s\\annotations\\set%02d\\V%03d\\I%05d.txt", MainDir.c_str(), dataset_name.c_str(), sets_cnt, vids_cnt, fr_no);
					ann_file_path = buff;

					resize(cur_frame, cur_frame, Size(0, 0), pre_scaling, pre_scaling);
					//blur(cur_frame, cur_frame, Size(5, 5), Point(-1, -1));
					Caltech_PDollar_Format_Results(cur_frame, this, pre_scaling, res_file_path, ann_file_path);
					imshow("Processed Frame", cur_frame);
					waitKey(1);
					frame_cnt++;
				}
					if ((Dataset == Ped_USA) | (Dataset == Ped_USA_Train) | (Dataset == Ped_USA_Test))
						fr_no = fr_no + 30;
					else
						fr_no = fr_no + 1;
			}
			vids_cnt++;
			if ((Dataset == Ped_USA) | (Dataset == Ped_USA_Train) | (Dataset == Ped_USA_Test))
				fr_no = 29;
			else
				fr_no = 0;
		}
		vids_cnt = 0;
		sets_cnt++;
		if (Dataset == Ped_USA_Train & sets_cnt == 6) break;
	}
	cout << "\n\nTotal frames Processed = " << frame_cnt;	//A quick check to see if all the directories, videos and frames were processed or not
}

void Caltech_PDollar_Format_Results(Mat& img, Musawwir_Obj_Detector* MOD, float pre_scaling, string res_path, string ann_path)
{
	vector<Rect> found;
	vector<double> scores;
	ofstream Res_File;
	Res_File.open(res_path);
	Mat temp;
	img.copyTo(temp);
	ftime(&t_start);
	switch (MOD->Active_Detector_Obj)
	{
	case Musawwir_Obj_Detector::HOG_OpenCV_Mod:
		(*MOD->HOG_OpenCV_Mod_Obj).detectMultiScale(temp, found, scores, MOD->Detection_Threshold, MOD->Spatial_Stride, MOD->Padding, MOD->Scale_Stride, 0, 1);
		break;
	case Musawwir_Obj_Detector::HOG_OpenCV:
		(*MOD->HOG_OpenCV_Obj).detectMultiScale(temp, found, scores, MOD->Detection_Threshold, MOD->Spatial_Stride, MOD->Padding, MOD->Scale_Stride, 0, 1);
		break;
	case Musawwir_Obj_Detector::HSG:
		(*MOD->HSG_Obj).MultiScale_Detector(temp, found, scores);
		break;
	default:
		break;
	}
	ftime(&t_end);
	t_elapsed = (float)((t_end.time - t_start.time) * 1000 + (t_end.millitm - t_start.millitm));
	fps = 0.9*fps + 100 / t_elapsed;
	char str[10];
	sprintf(str, "%.01f", fps);
	cv::putText(img, str, Point(20, 20), CV_FONT_HERSHEY_COMPLEX, 1, Scalar(255, 0, 0), 1, 8);

	for (int x = 0; x < scores.size(); x++) {
		if (scores[x] < 0) continue;
		Rect detected_bb;
		char str[50];
		sprintf(str, "%.02f", scores[x]);
		detected_bb = found[x];
		rectangle(img, detected_bb, Scalar(255, 0, 0), 4);
	}

	size_t i;
	for (i = 0; i < found.size(); i++)
	{
		Rect r = found[i];
		if (scores[i] < MOD->Detection_Threshold) continue;
		found[i].x = (found[i].x / pre_scaling);
		found[i].y = (found[i].y / pre_scaling);
		found[i].width = (found[i].width / pre_scaling);
		found[i].height = (found[i].height / pre_scaling);
		Res_File << found[i].x << "," << found[i].y << "," << found[i].width << "," << found[i].height << "," << scores[i] << endl;
	}
	Res_File.close();
}

//---------------------------------Annotation and ROC (Work in progress)--------------------------------------------
	/*Mat img_temp,win_temp;
	int margin = 128;
	copyMakeBorder(img, img_temp, margin, margin, margin, margin, BORDER_CONSTANT, Scalar(0, 0, 0));
	vector<Rect> gt;
	read_annotations(gt, ann_path,100,100);
	for (int x = 0; x < gt.size(); x++){
	Rect detected_bb;
	detected_bb = gt[x];
	rectangle(img, detected_bb, Scalar(0, 255, 0), 4, 4);
	detected_bb.x += margin;
	detected_bb.y += margin;
	float sc = (float)96 / (float)detected_bb.height;
	detected_bb.y -= (float)16 / sc;
	detected_bb.height += (float)32 / sc;
	float x_c = (float)detected_bb.x + ((float)detected_bb.width / 2);
	detected_bb.x = (int)(x_c - 32 / sc);
	detected_bb.width = 64 / sc;
	resize(img_temp(detected_bb), win_temp, Size(64, 128));
	imshow("Person", win_temp);
	waitKey(1);
	char str[500];
	sprintf(str, "E:\\RnD\\Current_Projects\\Musawwir\\Frameworks\\SW\\Dataset\\Person\\Caltech\\Train\\pos\\128X64(100_inf)\\%05d.bmp", img_ctr++);
	imwrite(str, win_temp);
	flip(win_temp, win_temp, 1);
	sprintf(str, "E:\\RnD\\Current_Projects\\Musawwir\\Frameworks\\SW\\Dataset\\Person\\Caltech\\Train\\pos\\128X64(100_inf)\\%05d.bmp", img_ctr++);
	imwrite(str, win_temp);

	height_hist[detected_bb.height]++;
	total_dt++;
	}*/
	
	//		ROC curve generation (Work in progress)
	/*
	//	if (scores[i]<SVM_Score_Th) continue;
	//sort rectangles according to their scores
	for (int i = 0; i < found.size()-1; i++){
	for (int j = i+1; j < found.size(); j++){
	if (scores[j]>scores[i]){
	double temp_s = scores[i];
	scores[i] = scores[j];
	scores[j] = temp_s;
	Rect temp_r = found[i];
	found[i] = found[j];
	found[j] = temp_r;
	}
	}
	}

	/*
	int* ftp = new int[found.size()];
	for (int i = 0; i < found.size(); i++) ftp[i] = 0;	//By default all detections are false positives

	for (int x = 0; x < gt.size(); x++){
	Rect gt_bb = gt[x];
	for (int y = 0; y < found.size(); y++){
	Rect dt_bb = found[y];
	Rect a = gt_bb & dt_bb;	//intersection
	Rect b = gt_bb | dt_bb;	//union
	if (a.area()>(b.area()*0.5)){		//if intersection area is greater than some 50 % of union area, it's a match!!
	ftp[y] = 1;
	TP_scores.push_back(scores[y]);
	break;
	}
	}
	}
	for (int y = 0; y < found.size(); y++){
	if (ftp[y] == 0){
	if (scores[y]>=SVM_Score_Th)
	FP_scores.push_back(scores[y]);
	}
	}*/

	/*	cout << endl << endl << "FP:\t";
	for (int y = 0; y < FP_scores.size(); y++)
	cout << FP_scores[y] << "\t";
	cout << endl << endl << "TP:\t";
	for (int y = 0; y < TP_scores.size(); y++)
	cout << TP_scores[y] << "\t";
	*/	//getchar();
//}


/*ofstream stats_file;
sprintf(ann_file_path, "%sCaltech\\code\\%s\\annotations\\%s_Stats.csv", Dataset_Path, dataset, dataset);
stats_file.open(ann_file_path);
for (int x = 0; x < 1000; x++){
stats_file << x << ",\t" << height_hist[x]<<endl;
}
stats_file.close();*/

/*	ofstream stats_file;
sprintf(ann_file_path, "%sCaltech\\code\\%s\\%s_Detection_Stats.csv", Dataset_Path, dataset, dataset);
stats_file.open(ann_file_path);
for (int y = 0; y < FP_scores.size(); y++)
stats_file << FP_scores[y] << ", ";
stats_file << "\n\n";
for (int y = 0; y < TP_scores.size(); y++)
stats_file << TP_scores[y] << ", ";
stats_file.close();*/
//	cout << "\n\nTotal true detections = " << total_dt;


