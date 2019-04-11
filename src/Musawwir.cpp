#include "Defines.h"
using namespace cv;
using namespace dnn;
using namespace std;

void Musawwir_Obj_Detector::Vid_Test(string dir_path) {
	VideoCapture capture;
	if (dir_path.empty()) {
		printf("\nError opening offline video...\nTrying webcam...\n");
		capture.open(0);
		if (!capture.isOpened()) {
			printf("--(!)\n\nError opening webcam\n\n");
			return;
		}
	}
	else {
		capture.open(dir_path);
		if (!capture.isOpened()) {
			printf("\nError opening offline video...\n\n");
			return;
		}
	}

	Mat frame;
	while (capture.read(frame)) {
		resize(frame, frame, Size(640, 480));	//VGA resolution
		Detect(frame);
	}
}

void Musawwir_Obj_Detector::Img_Dir_Test(string dir_path) {
	vector<String> fn;
	vector<cv::Mat> data;
	Mat img;
	cv::glob(dir_path, fn, true);
	for (size_t k = 0; k < fn.size(); ++k)
	{
		img = cv::imread(fn[k]);
		if (img.empty()) continue;
		Detect(img);
	}
}

//Perform the actual detection using the active detector
void Musawwir_Obj_Detector::Detect(Mat &img) {
	BB_Rects.clear();
	BB_Scores.clear();
	ftime(&t_start);
	switch (Active_Detector_Obj)
	{
	case Musawwir_Obj_Detector::HOG_OpenCV_Mod:
		(*HOG_OpenCV_Mod_Obj).detectMultiScale(img, BB_Rects, BB_Scores, Detection_Threshold, Spatial_Stride, Padding, Scale_Stride, 0, 1);
		break;
	case Musawwir_Obj_Detector::HOG_OpenCV:
		(*HOG_OpenCV_Obj).detectMultiScale(img, BB_Rects, BB_Scores, Detection_Threshold, Spatial_Stride, Padding, Scale_Stride, 0, 1);
		break;
	case Musawwir_Obj_Detector::HSG:
		(*HSG_Obj).MultiScale_Detector(img, BB_Rects, BB_Scores);
		break;
	case Musawwir_Obj_Detector::CNN_YOLO:
		CNN_YOLO_Detector(img, BB_Rects, BB_Scores);
	default:
		break;
	}
	ftime(&t_end);
	t_elapsed = (float)((t_end.time - t_start.time) * 1000 + (t_end.millitm - t_start.millitm));
	fps = 0.9*fps + 100 / t_elapsed;
	char str[10];
	sprintf(str, "%.01f", fps);
	cv::putText(img, str, Point(20, 20), CV_FONT_HERSHEY_COMPLEX, 1, Scalar(255, 0, 0), 1, 8);

	if (monitor_detections) {
		for (int x = 0; x < BB_Scores.size(); x++) {
			if (BB_Scores[x] < Detection_Threshold) continue;
			Rect detected_bb;
			char str[50];
			sprintf(str, "%.02f", BB_Scores[x]);
			detected_bb = BB_Rects[x];
			rectangle(img, detected_bb, Scalar(255, 0, 0), 4);
		}
		imshow("Processed Frame", img);
		waitKey(1);
	}
}