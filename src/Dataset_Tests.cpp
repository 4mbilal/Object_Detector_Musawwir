#include "Defines.h"
//Global Variables


void Caltech_PDollar_Format_Results(Mat& img, Musawwir_Obj_Detector* MOD, float pre_scaling, string res_path, string ann_path);
void read_annotations(vector<Rect>& Ann_Rects, string ann_file_path, int min_height, float min_overlap);
void write_detections_stats(string det_file_path, Musawwir_Obj_Detector* MOD);
float LAMR_Dataset(string FPPI_MR_file_path, Musawwir_Obj_Detector* MOD);


void Musawwir_Obj_Detector::Process_Test_Datasets(string Exp) {
	char buff[500];
	string dataset_name, res_file_dir, sets_dir_path, vids_dir_path, frm_path, res_file_path, ann_file_path;
	ofstream Res_File;

	int sets_cnt = 0, vids_cnt = 0, fr_no = 0;
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

	sprintf(buff, "%sCaltech\\code\\%s\\res\\%s_Detection_Stats.csv", MainDir.c_str(), dataset_name.c_str(), Exp.c_str());
	string det_stats_file_path = buff;
	write_detections_stats(det_stats_file_path, this);
	sprintf(buff, "%sCaltech\\code\\%s\\res\\%s_MR_FPPI.csv", MainDir.c_str(), dataset_name.c_str(), Exp.c_str());
	string FPPI_MR_file_path = buff;
	lamr = LAMR_Dataset(FPPI_MR_file_path, this);
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

	if (MOD->load_show_annotations) {
		vector<Rect> gt;
		Rect detected_bb;
		//1- Read annotations (Ground Truth)
		read_annotations(gt, ann_path, 50, 50);
		//2- Display annotations on frame
		for (int x = 0; x < gt.size(); x++) {
			detected_bb = gt[x];
			rectangle(img, detected_bb, Scalar(0, 255, 0), 4, 4);
		}
		MOD->total_objects += gt.size();

		//3- Sort detections according to their scores
		if (found.size() > 1) {
			for (int i = 0; i < found.size() - 1; i++) {
				for (int j = i + 1; j < found.size(); j++) {
					if (scores[j] > scores[i]) {
						double temp_s = scores[i];
						scores[i] = scores[j];
						scores[j] = temp_s;
						Rect temp_r = found[i];
						found[i] = found[j];
						found[j] = temp_r;
					}
				}
			}
		}

		//4- Initialize datastructure to label all detections as FP(0) or TP(1)
		MOD->ftp = new int[found.size()];
		for (int i = 0; i < found.size(); i++) MOD->ftp[i] = 0;	//By default all detections are false positives (0)

		//5- Search for the first detection (highest score) that overlaps (50%) with each annotation. Label these detections as TP(1)
		for (int x = 0; x < gt.size(); x++) {
			Rect gt_bb = gt[x];
			for (int y = 0; y < found.size(); y++) {
				Rect dt_bb = found[y];
				Rect a = gt_bb & dt_bb;	//intersection between ground truth and detection
				Rect b = gt_bb | dt_bb;	//union
				if (a.area()>(b.area()*0.25)) {		//if intersection area is greater than some 50 % of union area, it's a match!!
//				if (a.area()>((gt_bb.area()+dt_bb.area())*0.5)) {		//if intersection area is greater than some 50 % of union area, it's a match!!
					MOD->ftp[y] = 1;
					MOD->TP_scores.push_back(scores[y]);
					break;		//The top scoring detection has been found. All the others are False Positives. So, break here. 
				}
			}
		}
		//6- All the remaining detections get listed as FP
		for (int y = 0; y < found.size(); y++) {
			if (MOD->ftp[y] == 0) {
				MOD->FP_scores.push_back(scores[y]);
			}
		}
	}
}

float LAMR_Dataset(string FPPI_MR_file_path, Musawwir_Obj_Detector* MOD) {
	MOD->total_detections = MOD->FP_scores.size() + MOD->TP_scores.size();
	MOD->fp = new float[MOD->total_detections];
	MOD->tp = new float[MOD->total_detections];

	for (int i = 0; i < MOD->total_detections; i++) {
		MOD->fp[i] = 0;
		MOD->tp[i] = 0;
	}

	vector<double> FTP;
	for (int i = 0; i < MOD->TP_scores.size(); i++) {
		FTP.push_back(MOD->TP_scores[i]);
		MOD->tp[i] = 1;
	}
	for (int i = 0; i < MOD->FP_scores.size(); i++) {
		FTP.push_back(MOD->FP_scores[i]);
	}

	for (int i = 0; i < FTP.size() - 1; i++) {
		for (int j = i + 1; j < FTP.size(); j++) {
			if (FTP[j] > FTP[i]) {
				double temp_d = FTP[i];
				FTP[i] = FTP[j];
				FTP[j] = temp_d;
				int temp_i = MOD->tp[i];
				MOD->tp[i] = MOD->tp[j];
				MOD->tp[j] = temp_i;
			}
		}
	}

	for (int i = 0; i < MOD->total_detections; i++)
		MOD->fp[i] = 1-MOD->tp[i];

	for (int i = 1; i < MOD->total_detections; i++) {
		MOD->fp[i] = MOD->fp[i] + MOD->fp[i - 1];
		MOD->tp[i] = MOD->tp[i] + MOD->tp[i - 1];
	}

	float LAMR = 0;	//Log Average Miss Rate
	for (int i = 0; i < MOD->total_detections; i++) {
		MOD->tp[i] = ((float)MOD->total_objects - MOD->tp[i]) / (float)MOD->total_objects;
		MOD->fp[i] = MOD->fp[i] / (float)MOD->frame_cnt;
		LAMR += log(MOD->tp[i]);
	}
	LAMR = LAMR / (float)MOD->total_detections;
	//LAMR should probably be calculated over only the 'top' values (i.e. lower FPPI). Otherwise curves that reach bottom on the right can artificially give lower values.

	ofstream stats_file;
	stats_file.open(FPPI_MR_file_path);
	for (int i = 0; i < MOD->total_detections-1; i++)
		stats_file << MOD->tp[i] << ", ";
	stats_file << MOD->tp[MOD->total_detections-1];//last value should not have a comma at the end.

	stats_file << "\n\n";
	for (int i = 0; i < MOD->total_detections-1; i++)
		stats_file << MOD->fp[i] << ", ";
	stats_file << MOD->fp[MOD->total_detections-1];

	stats_file << "\n\n";
	stats_file.close();

	return abs(LAMR);
}


void write_detections_stats(string det_file_path, Musawwir_Obj_Detector* MOD) {
	ofstream stats_file;
	stats_file.open(det_file_path);
	
	for (int y = 0; y < MOD->FP_scores.size(); y++)
		stats_file << MOD->FP_scores[y] << ", ";
	stats_file << "\n\n";

	for (int y = 0; y < MOD->TP_scores.size(); y++)
	stats_file << MOD->TP_scores[y] << ", ";
	stats_file << "\n\n";
	
	stats_file << MOD->total_objects << "\n";
	stats_file.close();
}

void read_annotations(vector<Rect>& Ann_Rects, string ann_file_path, int min_height, float min_overlap) {
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

//---------------------------------Annotation to extract positive examples (Work in progress)--------------------------------------------
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

/*ofstream stats_file;
sprintf(ann_file_path, "%sCaltech\\code\\%s\\annotations\\%s_Stats.csv", Dataset_Path, dataset, dataset);
stats_file.open(ann_file_path);
for (int x = 0; x < 1000; x++){
stats_file << x << ",\t" << height_hist[x]<<endl;
}
stats_file.close();*/

