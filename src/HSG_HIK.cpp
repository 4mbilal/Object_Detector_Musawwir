/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

/*							** HSG-HIK **
	Open Source implementation of Histogram of Significant Gradients (HSG) features and LUT-based Histogram-Intersection-Kernel (HIK) SVM 
	for pedestrian detection as described in the following papers.

			1- M Bilal, A Khan, MUK Khan, CM Kyung, "A Low Complexity Pedestrian Detection Framework for Smart Video Surveillance Systems", 
			IEEE Transactions on Circuits and Systems for Video Technology, vol. 27, no. 10, pp. 2260-2273, Oct. 2017.
			2- M. Bilal, "Algorithmic optimisation of histogram intersection kernel support vector machine-based pedestrian detection using low complexity features,"
			IET Computer Vision, vol. 11, no. 5, pp. 350-357, 8 2017.
			3- M. Bilal, M. S. Hanif, "High Performance Real-Time Pedestrian Detection using Light Weight Features and Fast Cascaded Kernel SVM Classification",
			Springer Journal of Signal Processing Systems, 2018.

			Briefly, HSG features were proposed in #1 since they provide a very low cost alternative to HOG despite being more effective for pedestrian detection.
			#2 describes some improvement in the training process and adds color information in addition to orientation binning.
			#3 proposes a cascaded SVM implementation plus use of multiple detectors of different window sizes to speed up significantly.
			
	This implementation reuses OpenCV code from hog.cpp (some portions of computeGradient and Parallel Loopbobdy functions). Hence, the above copyright notice has been reproduced.
	This code has been provided on as-is basis and while queries are welcome, the code is not actively maintained as of January, 2018.

	Muhammad Bilal (4mbilal@gmail.com)

*/
#include "HSG_HIK.h"
#include "NMS.h"
#include "HSG004_HIK_LUT.h"		//{ 54, 18 }  9 bin
#include "HSG005_HIK_LUT.h"		//{ 78, 30 } 9 bin
#include "HSG006_HIK_LUT.h"		//{ 90, 36 }  9 bin	
#include "HSG007_HIK_LUT.h"		//{ 102, 36 } 9 bin

int HSG_HIK_Test(void)
{
	printf("\n\t\t** Pedestrian Detection Test **\n\tHSG Vs. OpenCV Default HOG");

	HOGDescriptor OCV_Default;	//Initialize pedestrian detectors for each model
	OCV_Default.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());	//Set default OpenCV Model
	HSGDetector HSG;

	Mat img, out_default, out_hsg, out_combined;
	vector<Rect> found_default, found_hsg;
	vector<double> scores_default, scores_hsg;
	char str[150];

	String path("E:\\test_images\\");
	vector<String> fn;
	vector<cv::Mat> data;
	cv::glob(path, fn, true); 
	for (size_t k = 0; k<fn.size(); ++k)
	{
		img = cv::imread(fn[k]);
		if (img.empty()) continue; 

		out_default = img.clone();
		out_hsg = img.clone();

		//Run both detectors on same image
		OCV_Default.detectMultiScale(out_default, found_default, scores_default, 0, Size(8, 8), Size(24, 24), 1.05, 0, 1);
		HSG.MultiScale_Detector(out_hsg, found_hsg, scores_hsg);

		//Draw output boundary boxes
		HSG.DrawRectangles(out_default, found_default, scores_default, 0);
		HSG.DrawRectangles(out_hsg, found_hsg, scores_hsg, 0);

		//Prepare for displaying the results
		copyMakeBorder(out_default, out_default, 8, 8, 8, 8, BORDER_CONSTANT, 0);
		copyMakeBorder(out_hsg, out_hsg, 8, 8, 8, 8, BORDER_CONSTANT, 0);
		cv::putText(out_default, "OpenCV Default HOG", Point(32, 32), CV_FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 255), 2, 16);
		cv::putText(out_hsg, "HSG-HIK", Point(32, 32), CV_FONT_HERSHEY_COMPLEX, 1, Scalar(0, 255, 0), 2, 16);
		hconcat(out_default, out_hsg, out_combined);
		namedWindow("Output Comparison" , WINDOW_NORMAL);
		imshow("Output Comparison", out_combined);

		//Write the output images
		sprintf(str, "E:\\Output\\%s", fn[k].substr(fn[k].find_last_of("\\"), fn[k].find_last_of(".")).c_str());
		cout << endl << str;
		imwrite(str, out_combined);
		waitKey(1);
	}

	if (1) {
		//Live WebCam Speed Test
		VideoCapture capture;
		capture.open(0);
		if (!capture.isOpened()) { printf("--(!)Error opening video capture\n"); return -1; }
		Mat frame;
		float fps_1 = 1, fps_2 = 1, msec;

		while (capture.read(frame)) {
			resize(frame, frame, Size(640, 480));	//VGA resolution
			frame.copyTo(out_default);
			frame.copyTo(out_hsg);

			ftime(&HSG.t_start);
			HSG.MultiScale_Detector(out_hsg, found_hsg, scores_hsg);
			ftime(&HSG.t_end);
			msec = int((HSG.t_end.time - HSG.t_start.time) * 1000 + (HSG.t_end.millitm - HSG.t_start.millitm));
			fps_1 = 0.9*fps_1 + 100 / msec;

			ftime(&HSG.t_start);
			OCV_Default.detectMultiScale(out_default, found_default, scores_default, 0, Size(8, 8), Size(24, 24), 1.05, 0, 1);
			ftime(&HSG.t_end);
			msec = int((HSG.t_end.time - HSG.t_start.time) * 1000 + (HSG.t_end.millitm - HSG.t_start.millitm));
			fps_2 = 0.9*fps_2 + 100 / msec;

			HSG.DrawRectangles(out_default, found_default, scores_default, 0);
			HSG.DrawRectangles(out_hsg, found_hsg, scores_hsg, 0);
			//Prepare for displaying the results
			copyMakeBorder(out_default, out_default, 8, 8, 8, 8, BORDER_CONSTANT, 0);
			copyMakeBorder(out_hsg, out_hsg, 8, 8, 8, 8, BORDER_CONSTANT, 0);
			char str[50];
			sprintf(str, "OpenCV Default HOG - %.1f fps", fps_2);
			cv::putText(out_default, str, Point(32, 32), CV_FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 255), 2, 16);
			sprintf(str, "HSG-HIK - %.1f fps", fps_1);
			cv::putText(out_hsg, str, Point(32, 32), CV_FONT_HERSHEY_COMPLEX, 1, Scalar(0, 255, 0), 2, 16);
			hconcat(out_default, out_hsg, out_combined);
			namedWindow("Live Video", WINDOW_NORMAL);
			imshow("Live Video", out_combined);
			waitKey(1);
		}
	}
}


template <size_t xx, size_t yy>
void SVM_Window_Scan(HSGDetector* hsg, Mat& frame, unsigned char* frame_Hists, int rows, int cols, vector<Rect>& BB_Rects, vector<double>& BB_Scores, double scale, vector<double>& BB_Scales, int* WinSize, float(&HIK_LUT)[xx][yy], float hik_bias, float padding_scale, int* cascade_indices, float* cascade_stage_th, int cascade_stage_cnt) {
	//	ftime(&t_start);
	int no_horiz_hists = ((cols - 2) / hsg->CellStride[1]);
	int no_vert_hists = ((rows - 2) / hsg->CellStride[0]);
	int hi_max = (WinSize[0] / hsg->CellStride[0]) - 1;
	int hj_max = (WinSize[1] / hsg->CellStride[1]) - 1;
	int hij_max = (hj_max)*(hi_max);
	int tot_bins = hsg->OrBins + hsg->ColorBins;
	int hjk_max = hj_max*tot_bins;
	int horiz_elements = no_horiz_hists*tot_bins;//2180 for 640x480 frame

	int mem_siz = (tot_bins)*(no_vert_hists)*(no_horiz_hists);
	for (int x = 0; x < mem_siz; x++) {
		int Q_Val = frame_Hists[x];
		if (Q_Val >(hsg->Kernel_LUT_Q - 1))
			frame_Hists[x] = hsg->Kernel_LUT_Q - 1;
	}

	Mutex mtx;
	int wind_cnt = 0, oper_cnt = 0;
	int nwindows = (no_vert_hists - hi_max)*(no_horiz_hists - hj_max);
	//	printf("\nNo. of windows = %d", nwindows); getchar();

	int nwindowsX = no_horiz_hists - hj_max;
#ifdef Use_P4
	parallel_for(size_t(0), size_t(no_vert_hists - hi_max), size_t(hsg->Detect_Win_Stride[0]), [&](size_t v_adr_i) {
		unsigned char* frame_Hists_row_ptr = frame_Hists + v_adr_i*horiz_elements;
		parallel_for(size_t(0), size_t(no_horiz_hists - hj_max), size_t(hsg->Detect_Win_Stride[1]), [&](size_t h_adr_i) {
#else
	for (int v_adr_i = 0; v_adr_i < no_vert_hists - hi_max; v_adr_i += hsg->Detect_Win_Stride[0]) {
		unsigned char* frame_Hists_row_ptr = frame_Hists + v_adr_i*horiz_elements;
		for (int h_adr_i = 0; h_adr_i < no_horiz_hists - hj_max; h_adr_i += hsg->Detect_Win_Stride[1]) {
#endif
			
			float score = 0;
			int feature_indx = 0;
			//wind_cnt++;
			unsigned char* frame_Hists_win_ptr = frame_Hists_row_ptr + h_adr_i * tot_bins;

#ifdef Use_Cascade
			int c = 0;
			for (int cascade_stage = 0; cascade_stage < cascade_stage_cnt; cascade_stage++) {//Need to add more stages. Collect stats to get further insight 
				for (int ic=0; ic < 15; ic++) {
					int hi = cascade_indices[c * 2];
					int hj = cascade_indices[c * 2 + 1];
					int a1 = hi*horiz_elements;
					int f1 = hi*hjk_max;
					int a2 = hj * tot_bins;
					int f2 = f1 + a2;
					for (int hk = 0; hk < tot_bins; hk++) {
						score += HIK_LUT[frame_Hists_win_ptr[a1 + a2 + hk]][f2 + hk];
					}
					c++;
				}
				if (score + cascade_stage_th[cascade_stage] < hik_bias)
					goto end_cascade;// break;
			}
			for (; c < hi_max*hj_max; c++) {
				int hi = cascade_indices[c * 2];
				int hj = cascade_indices[c * 2 + 1];
				int a1 = hi*horiz_elements;
				int f1 = hi*hjk_max;
				int a2 = hj * tot_bins;
				int f2 = f1 + a2;
				for (int hk = 0; hk < tot_bins; hk++) {
					score += HIK_LUT[frame_Hists_win_ptr[a1 + a2 + hk]][f2 + hk];
				}
			}
#else
			for (int hi=0; hi < hi_max; hi++) {
				int a1 = hi*horiz_elements;
				int f1 = hi*hjk_max;
				for (int hj = 0; hj < hj_max; hj++) {
					int a2 = hj * tot_bins;
					int f2 = f1 + a2;
					for (int hk = 0; hk < tot_bins; hk++) {
						score += HIK_LUT[frame_Hists_win_ptr[a1 + a2 + hk]][f2 + hk];
					}
				}
			}

#endif			

end_cascade:
			score = score - hik_bias;
			//printf("\n%.2f", score); getchar();

			if (score > hsg->SVM_Score_Th) {
				int i = h_adr_i*hsg->CellStride[1] + 1;
				int j = v_adr_i*hsg->CellStride[0] + 1;

				Rect detected_bb;
				bool exist = false;
				//		Original Detected BB in padded and scaled frame
				detected_bb.x = (float)i;
				detected_bb.y = (float)j;
				detected_bb.width = (float)WinSize[1];
				detected_bb.height = (float)WinSize[0];
				//		Deal with objects towards edges of the frame (detected due to padding)
				detected_bb.x = detected_bb.x - hsg->Frame_Padding[2] * padding_scale;
				detected_bb.y = detected_bb.y - hsg->Frame_Padding[0] * padding_scale;
				if (detected_bb.x < 0) {
					detected_bb.width = detected_bb.width + detected_bb.x;
					detected_bb.x = 0;
				}
				if (detected_bb.y < 0) {
					detected_bb.height = detected_bb.height + detected_bb.y;
					detected_bb.y = 0;
				}

				//		Add scaling effect
				detected_bb.x = (detected_bb.x / scale);
				detected_bb.y = (detected_bb.y / scale);
				detected_bb.width = (detected_bb.width / scale);
				detected_bb.height = (detected_bb.height / scale);

				mtx.lock();
				BB_Rects.push_back(detected_bb);
				BB_Scores.push_back((double)score);
				BB_Scales.push_back(scale * (float)hsg->WinSize.height / (float)WinSize[0]);
				//cout << "\nDetected = " << BB_Scores.size() << endl;
				mtx.unlock();
			}
#ifdef Use_P4
		});
	});
#else	
		}
	}
#endif
	//	ftime(&t_end);
	//	printf("\nWindwos = %d, Operations = %d", wind_cnt, oper_cnt);	getchar();
}

//-------------------------------------------------------------------------------------------------------------
void HSGDetector::DrawRectangles(Mat& Frame, vector<Rect>& BB_Rects, vector<double>& BB_Scores, float SVM_Score_Th) {
	for (int x = 0; x < BB_Scores.size(); x++) {
		if (BB_Scores[x] < SVM_Score_Th) continue;
		Rect detected_bb;
		char str[50];
		sprintf(str, "%.02f", BB_Scores[x]);
		detected_bb = BB_Rects[x];
		cv::putText(Frame, str, Point(detected_bb.x, detected_bb.y), CV_FONT_HERSHEY_COMPLEX, 1, Scalar(255, 0, 0), 1, 8);
		rectangle(Frame, detected_bb, Scalar(255, 0, 0), 4);
	}
}

void HSGDetector::computeGradient(const Mat& img, float* Grad_Mag, unsigned char* Grad_Or) {
	Size gradsize(img.cols, img.rows);
	Size wholeSize;
	Point roiofs;
	img.locateROI(wholeSize, roiofs);
	const float pi = 3.141593;

	int i, y;
	bool gammaCorrection = 0;	//not using the gamma correction part of this function
	float fGamma = 0.5;
	int width = gradsize.width;
	float angleScale = (float)(OrBins / CV_PI);		//No Orientation [0 180], Use 2*Pi for [0 360] range

	Mat_<float> _lut(1, 256);
	const float* const lut = &_lut(0, 0);
#ifdef Use_SSE
	const int indeces[] = { 0, 1, 2, 3 };
	__m128i idx = _mm_loadu_si128((const __m128i*)indeces);		//This instruction not translated in SSE2NEON.h
	__m128i ifour = _mm_set1_epi32(4);

	float* const _data = &_lut(0, 0);
	if (gammaCorrection)
		for (i = 0; i < 256; i += 4)
		{
			_mm_storeu_ps(_data + i, _mm_sqrt_ps(_mm_cvtepi32_ps(idx)));
			idx = _mm_add_epi32(idx, ifour);
		}
	else
		for (i = 0; i < 256; i += 4)
		{
			_mm_storeu_ps(_data + i, _mm_cvtepi32_ps(idx));
			idx = _mm_add_epi32(idx, ifour);
		}
#else
	if (gammaCorrection)
		for (i = 0; i < 256; i++)
			_lut(0, i) = saturate_cast<uchar>(pow((float)(i / 255.0), fGamma) * 255.0f); //std::sqrt((float)i);
	else
		for (i = 0; i < 256; i++)
			_lut(0, i) = (float)i;
#endif

	// x- & y- derivatives for the whole row
#ifdef Use_P4
	parallel_for(size_t(1), size_t(gradsize.height - 1), size_t(1), [&](size_t y) {
#endif
		AutoBuffer<float> _dbuf(width * 4);			//this will contain Dx, Dy, Mag and Angle for the whole row, hence width*4 size.
		float* const dbuf = _dbuf;
		Mat Dx(1, width, CV_32F, dbuf);
		Mat Dy(1, width, CV_32F, dbuf + width);
		Mat Mag(1, width, CV_32F, dbuf + width * 2);
		Mat Angle(1, width, CV_32F, dbuf + width * 3);
#ifndef Use_P4
		for (y = 1; y < gradsize.height - 1; y++) {		//iterate over rows	
#endif
			const uchar* imgPtr = img.ptr(y);					//address of the first pixel in this row (y)
			const uchar* prevPtr = img.data + img.step*(y + 1);	//these have been interchanged because of the way HSG was trained.
			const uchar* nextPtr = img.data + img.step*(y - 1);

			float* gradPtr = Grad_Mag + y*width;
			unsigned char* qanglePtr = Grad_Or + y*width;

			int x = 1;
#ifdef Use_SSE		//With and without SSE give slightly different results, Investigate!
			for (; x <= width - 4; x += 4)
			{
				int x0 = x * 3, x1 = (x + 1) * 3, x2 = (x + 2) * 3, x3 = (x + 3) * 3;
				typedef const uchar* const T;
				T p02 = imgPtr + (x + 1) * 3, p00 = imgPtr + (x - 1) * 3;
				T p12 = imgPtr + (x + 2) * 3, p10 = imgPtr + x * 3;
				T p22 = imgPtr + (x + 3) * 3, p20 = p02;
				T p32 = imgPtr + (x + 4) * 3, p30 = p12;

				//To-Do 1- _mm_loadu_si128 2- _mm_sub_epi8, _mm_sub_epi16

				//The instructions below are too slow!! Because they use _mm_set_ps to move scalars (interleaved rgb values) into sse registers
				//Take a look at the assembly and you will see a lot of instructions being generted against these. 
				//So planar rgb is best for SSE as done by ACF.
				__m128 _dx0 = _mm_sub_ps(_mm_set_ps(lut[p32[0]], lut[p22[0]], lut[p12[0]], lut[p02[0]]),
					_mm_set_ps(lut[p30[0]], lut[p20[0]], lut[p10[0]], lut[p00[0]]));
				__m128 _dx1 = _mm_sub_ps(_mm_set_ps(lut[p32[1]], lut[p22[1]], lut[p12[1]], lut[p02[1]]),
					_mm_set_ps(lut[p30[1]], lut[p20[1]], lut[p10[1]], lut[p00[1]]));
				__m128 _dx2 = _mm_sub_ps(_mm_set_ps(lut[p32[2]], lut[p22[2]], lut[p12[2]], lut[p02[2]]),
					_mm_set_ps(lut[p30[2]], lut[p20[2]], lut[p10[2]], lut[p00[2]]));

				__m128 _dy0 = _mm_sub_ps(_mm_set_ps(lut[nextPtr[x3]], lut[nextPtr[x2]], lut[nextPtr[x1]], lut[nextPtr[x0]]),
					_mm_set_ps(lut[prevPtr[x3]], lut[prevPtr[x2]], lut[prevPtr[x1]], lut[prevPtr[x0]]));
				__m128 _dy1 = _mm_sub_ps(_mm_set_ps(lut[nextPtr[x3 + 1]], lut[nextPtr[x2 + 1]], lut[nextPtr[x1 + 1]], lut[nextPtr[x0 + 1]]),
					_mm_set_ps(lut[prevPtr[x3 + 1]], lut[prevPtr[x2 + 1]], lut[prevPtr[x1 + 1]], lut[prevPtr[x0 + 1]]));
				__m128 _dy2 = _mm_sub_ps(_mm_set_ps(lut[nextPtr[x3 + 2]], lut[nextPtr[x2 + 2]], lut[nextPtr[x1 + 2]], lut[nextPtr[x0 + 2]]),
					_mm_set_ps(lut[prevPtr[x3 + 2]], lut[prevPtr[x2 + 2]], lut[prevPtr[x1 + 2]], lut[prevPtr[x0 + 2]]));

				__m128 _mag0 = _mm_add_ps(_mm_mul_ps(_dx0, _dx0), _mm_mul_ps(_dy0, _dy0));
				__m128 _mag1 = _mm_add_ps(_mm_mul_ps(_dx1, _dx1), _mm_mul_ps(_dy1, _dy1));
				__m128 _mag2 = _mm_add_ps(_mm_mul_ps(_dx2, _dx2), _mm_mul_ps(_dy2, _dy2));

				__m128 mask = _mm_cmpgt_ps(_mag2, _mag1);
				_dx2 = _mm_or_ps(_mm_and_ps(_dx2, mask), _mm_andnot_ps(mask, _dx1));
				_dy2 = _mm_or_ps(_mm_and_ps(_dy2, mask), _mm_andnot_ps(mask, _dy1));

				mask = _mm_cmpgt_ps(_mm_max_ps(_mag2, _mag1), _mag0);
				_dx2 = _mm_or_ps(_mm_and_ps(_dx2, mask), _mm_andnot_ps(mask, _dx0));
				_dy2 = _mm_or_ps(_mm_and_ps(_dy2, mask), _mm_andnot_ps(mask, _dy0));

				_mm_storeu_ps(dbuf + x, _dx2);
				_mm_storeu_ps(dbuf + x + width, _dy2);
			}
#endif
			for (; x < width - 1; x++)
			{
				int x1 = x * 3;
				float dx0, dy0, dx, dy, mag0, mag;
				const uchar* p2 = imgPtr + (x + 1) * 3;
				const uchar* p0 = imgPtr + (x - 1) * 3;

				dx0 = lut[p2[2]] - lut[p0[2]];
				dy0 = lut[nextPtr[x1 + 2]] - lut[prevPtr[x1 + 2]];
				mag0 = dx0*dx0 + dy0*dy0;		//Magnitude for Red

				dx = lut[p2[1]] - lut[p0[1]];
				dy = lut[nextPtr[x1 + 1]] - lut[prevPtr[x1 + 1]];
				mag = dx*dx + dy*dy;			//Magnitude for Green
				if (mag0 < mag) {
					dx0 = dx;
					dy0 = dy;
					mag0 = mag;
				}

				dx = lut[p2[0]] - lut[p0[0]];
				dy = lut[nextPtr[x1]] - lut[prevPtr[x1]];
				mag = dx*dx + dy*dy;		//Magnitude for Blue
				if (mag0 < mag) {
					dx0 = dx;
					dy0 = dy;
					mag0 = mag;
				}

				dbuf[x] = dx0;				//dx for max mag
				dbuf[x + width] = dy0;		//dy for max mag
			}
			// computing angles and magnidutes
			cartToPolar(Dx, Dy, Mag, Angle, false);

			// filling the result matrix
			x = 0;
#ifdef Use_SSE
			__m128 fzero = _mm_setzero_ps();
			__m128 _angleScale = _mm_set1_ps(angleScale), fone = _mm_set1_ps(1.0f);
			__m128i _nbins = _mm_set1_epi32(OrBins), izero = _mm_setzero_si128();
			__m128 Q = _mm_set1_ps(0.0625f);
			__m128i store[3];
			int ans[4];

			int n = 1;
			for (; x <= width - n * 4; x += n * 4)
			{
				for (int p = 0; p < 2; p++) {
					int a = p * 4;
					__m128 mag = _mm_loadu_ps(dbuf + x + a + (width << 1));				//float mag = dbuf[x+width*2]
					__m128 mag_q = _mm_mul_ps(mag, Q);								//mag_q = mag*0.0625 (mag/16), use "_mm_cvtps_epi32" or "_mm_cvttps_epi32" for rounding/truncating to int
					_mm_storeu_ps(gradPtr + x + a, mag_q);								//store the quantized mag (mag_q)
					__m128 _angle = _mm_loadu_ps(dbuf + x + a + width * 3);				//float angle = dbuf[x+width*3]
					_angle = _mm_mul_ps(_angleScale, _angle);	//angle*=angleScale;
					__m128 sign = _mm_and_ps(fone, _mm_cmplt_ps(_angle, fzero));
					__m128i _hidx = _mm_cvttps_epi32(_angle);						//int hidx = cvFloor(angle);
					_hidx = _mm_sub_epi32(_hidx, _mm_cvtps_epi32(sign));			//?
					_angle = _mm_sub_ps(_angle, _mm_cvtepi32_ps(_hidx));			// angle -= hidx;
					__m128i mask0 = _mm_sub_epi32(izero, _mm_srli_epi32(_hidx, 31));
					__m128i it0 = _mm_and_si128(mask0, _nbins);
					mask0 = _mm_cmplt_epi32(_hidx, _nbins);
					__m128i it1 = _mm_andnot_si128(mask0, _nbins);
					_hidx = _mm_add_epi32(_hidx, _mm_sub_epi32(it0, it1));
					store[p] = _hidx;
				}

				//__m128i or_16 = _mm_packs_epi16(_mm_packs_epi32(store[0], store[1]), _mm_packs_epi32(store[2], store[3]));
				//_mm_storel_epi64((__m128i*)(qanglePtr + x), or_16);

				//_mm_storeu_si128((__m128i*)(qanglePtr + x), or_16);

				_mm_store_si128((__m128i*)ans, store[0]);
				qanglePtr[x] = ans[0];
				qanglePtr[x + 1] = ans[1];
				qanglePtr[x + 2] = ans[2];
				qanglePtr[x + 3] = ans[3];
			}
#endif
			for (; x < width; x++)
			{
				gradPtr[x] = dbuf[x + width * 2] / GradMagQ;
				/*float angle = dbuf[x + width * 3];
				if (angle > CV_PI) angle -= CV_PI;
				if (angle < 0) angle += CV_PI;
				angle *= angleScale;
				angle = cvFloor(angle);
				qanglePtr[x] = (uchar)angle;*/
				//------
				float angle = dbuf[x + width * 3] * angleScale;
				int hidx = cvFloor(angle);
				angle -= hidx;
				if (hidx < 0)
					hidx += OrBins;
				else if (hidx >= OrBins)
					hidx -= OrBins;
				qanglePtr[x] = (uchar)hidx;
			}
#ifdef Use_P4
		});
#else
	}
#endif
}

void HSGDetector::HSG_Feature(Mat frame, unsigned char* frame_features) {
	//	ftime(&t_start);
	int no_horiz_hists = ((frame.cols - 2) / CellStride[1]);
	int no_vert_hists = ((frame.rows - 2) / CellStride[0]);
	int tot_bins = OrBins + ColorBins;
	int horiz_elements = no_horiz_hists*tot_bins;
	unsigned char *frame_data_ptr = (unsigned char*)(frame.data);

	float* Grad_Mag;
	unsigned char* Grad_Or;
	int rc = frame.rows*frame.cols;
	Grad_Mag = new float[rc];
	Grad_Or = new unsigned char[rc];

	//	ftime(&t_start);

	//Color Space Conversion
	Mat luv;
	cvtColor(frame, luv, CV_BGR2Luv);	//Uses Parallel Loopbody, 
	unsigned char *luv_ptr = (unsigned char*)(luv.data);

	//Gradient Calculation (Magnitude and Orientation)
	computeGradient(frame, Grad_Mag, Grad_Or);	//This function directly returns the indices of orientation bins in Grad_Or
	//	ftime(&t_end);

	

	//	ftime(&t_start);
	//Leave one-pixel boundary around the frame where gradient data is not available
	unsigned char r0 = 1, c0 = 1;			//Set boundary for rows and columns (Start processing from these values instead of 0,0)

#ifdef Use_P4
	parallel_for(size_t(r0), size_t(frame.rows - CellSize.height), size_t(CellStride[0]), [&](size_t x) {
#else
	for (unsigned int x = r0; x < frame.rows - CellSize.height; x += CellStride[0]) {
#endif
		int a1 = int((x - 1) / CellStride[0])*horiz_elements;
		unsigned char* row_ptr = frame_features + a1;
		unsigned char* luv_row = luv_ptr + x*frame.cols * 3;
		float* Grad_Mag_row = Grad_Mag + x*frame.cols;
		unsigned char* Grad_Or_row = Grad_Or + x*frame.cols;
		for (unsigned int y = c0; y < frame.cols - CellSize.height; y += CellStride[1]) {
			int a2 = int((y - 1) / CellStride[1]);
			unsigned char* hist_ptr = row_ptr + a2*tot_bins;
			unsigned char* luv_block = luv_row + y * 3;
			float* Grad_Mag_block = Grad_Mag_row + y;
			unsigned char* Grad_Or_block = Grad_Or_row + y;

			int luv_U_avg = 0;
			//int luv_V_avg = 0;
			for (int i = 0; i < CellSize.height; i++) {
				unsigned char* luv_block_row = luv_block + i*frame.cols * 3;
				float* Grad_Mag_block_row = Grad_Mag_block + i*frame.cols;
				unsigned char* Grad_Or_block_row = Grad_Or_block + i*frame.cols;
				for (int j = 0; j < CellSize.width; j++) {
					float grad_mag = Grad_Mag_block_row[j];
					luv_U_avg += luv_block_row[j * 3 + 1];
					if ((int)grad_mag > 0) {
						//---------- 3 bin votes 1-2-1 -------------------
						const unsigned char side_bin_lut[2][9] = { { 8,0,1,2,3,4,5,6,7 },{ 1,2,3,4,5,6,7,8,0 } };	//LUT to get indices for side bins
						unsigned char b = Grad_Or_block_row[j];
						hist_ptr[b] += 2;						//Main bin gets two votes
						hist_ptr[side_bin_lut[0][b]] += 1;		//Side bins get one vote each
						hist_ptr[side_bin_lut[1][b]] += 1;
					}
				}
			}
			luv_U_avg = luv_U_avg / (CellSize.height * CellSize.width);
			hist_ptr[OrBins] = (int)(luv_U_avg / 4);		//Color information is only the average value of U part as of now, stored in the first bin allocated for color information
		}
#ifdef Use_P4
	});
#else
	}
#endif
	//	getchar();

	//	ftime(&t_end);
	//	Release allocated memory
	delete Grad_Mag;
	delete Grad_Or;
	//	ftime(&t_end);
}

void HSGDetector::SingleScale_Detector(Mat& frame, vector<Rect>& BB_Rects, vector<double>& BB_Scores, double scale, vector<double>& BB_Scales) {
	//	ftime(&t_start);
	//---------------------------------------------------------------------------------------------------------
	//				Insert border around the frame to detect partially occluded objects on the sides
	//---------------------------------------------------------------------------------------------------------
	copyMakeBorder(frame, frame, Frame_Padding[0], Frame_Padding[1], Frame_Padding[2], Frame_Padding[3], BORDER_CONSTANT, Scalar(255, 255, 255));
	int frows = frame.rows;
	int fcols = frame.cols;
	int no_horiz_hists = ((fcols - 2) / CellStride[1]);
	int no_vert_hists = ((frows - 2) / CellStride[0]);
	/*	cout << "\nRows = " << rows;	cout << ", Cols = " << cols;	cout << ", horiz hists = " << no_horiz_hists;	cout << ", vert hists = " << no_vert_hists;	cout << "\nscale = " << scale;	getchar();*/
	//---------------------------------------------------------------------------------------------------------
	//									Memory Allocation & Initialization
	//---------------------------------------------------------------------------------------------------------
	unsigned char* frame_features;	//A single data structure to hold features calculated for the whole frame
	int mem_siz = (no_vert_hists)* (no_horiz_hists)*(OrBins + ColorBins);
	frame_features = new unsigned char[mem_siz];			
	for (int i = 0; i < mem_siz; i++)
		frame_features[i] = 0;		//initialize hists to zero*/
	//	ftime(&t_end);
	//---------------------------------------------------------------------------------------------------------
	//									Calculate Features (Histograms + Color info) for the whole image
	//---------------------------------------------------------------------------------------------------------
	//	ftime(&t_start);
	HSG_Feature(frame, frame_features);
	//	ftime(&t_end);
	//---------------------------------------------------------------------------------------------------------
	//											SVM Scanning Window
	//---------------------------------------------------------------------------------------------------------
#ifdef Training_Mode
	HSG_Window_Scan(frame, frame_Hists, rows, cols, BB_Rects, BB_Scores, scale, BB_Scales, training_mode, HSG_Descriptor_WinSize, Kernel_LUT, b, 1);
#else
	//ftime(&t_start);
	if (scale>0.6)	//Small window detector does not work too well on downsampled images. So limit their use for higher resolutions only.
		SVM_Window_Scan(this, frame, frame_features, frows, fcols, BB_Rects, BB_Scores, scale, BB_Scales, WinSize_4, Kernel_LUT_4, b_4, 1, cascade_indices_4, cascade_stage_th_4, 5);
	SVM_Window_Scan(this, frame, frame_features, frows, fcols, BB_Rects, BB_Scores, scale, BB_Scales, WinSize_5, Kernel_LUT_5, b_5, 1, cascade_indices_5, cascade_stage_th_5, 6);
	SVM_Window_Scan(this, frame, frame_features, frows, fcols, BB_Rects, BB_Scores, scale, BB_Scales, WinSize_6, Kernel_LUT_6, b_6, 1, cascade_indices_6, cascade_stage_th_6, 6);
	SVM_Window_Scan(this, frame, frame_features, frows, fcols, BB_Rects, BB_Scores, scale, BB_Scales, WinSize_7, Kernel_LUT_7, b_7, 1, cascade_indices_7, cascade_stage_th_7, 6);
	//ftime(&t_end);
#endif
	//---------------------------------------------------------------------------------------------------------
	//											Release Memory
	//---------------------------------------------------------------------------------------------------------
	//	ftime(&t_start);
	delete frame_features;
	//	ftime(&t_end);
}

class MultiScale_Detector_Parallel_Process : public ParallelLoopBody
{
private:
	HSGDetector* hsg;
	Mat frame;
	vector<Rect>* BB_Rects;
	vector<double>* BB_Scores;
	vector<double>* BB_Scales;
	vector<double>* Scales;
	Mutex* mtx;

public:
	MultiScale_Detector_Parallel_Process(HSGDetector* _hsg, const Mat _frame, vector<Rect>* _BB_Rects, vector<double>* _BB_Scores, vector<double>* _BB_Scales, vector<double>* _Scales, Mutex* _mtx)
	{
		hsg = _hsg;
		frame = _frame;
		BB_Rects = _BB_Rects;
		BB_Scores = _BB_Scores;
		BB_Scales = _BB_Scales;
		Scales = _Scales;
		mtx = _mtx;
	}

	void operator()(const Range& range) const
	{
		int i, i1 = range.start, i2 = range.end;

		for (i = i1; i < i2; i++)
		{
			Mat frame_resized, frame_blurred;
			Size sz(cvRound(frame.cols*Scales->at(i)), cvRound(frame.rows*Scales->at(i)));
			resize(frame, frame_resized, sz);// , CV_INTER_CUBIC);

			vector<Rect> BB_Rects_i;		//Detected BBs for the current scale indexed by i
			vector<double> BB_Scores_i;		//Corresponding scores for the current scale indexed by i
			vector<double> BB_Scales_i;		//Corresponding scales for the current scale indexed by i
			hsg->SingleScale_Detector(frame_resized, BB_Rects_i, BB_Scores_i, Scales->at(i), BB_Scales_i);
			
			mtx->lock();
			for (int ii = 0; ii < BB_Scores_i.size(); ii++) {
				BB_Scores->push_back(BB_Scores_i[ii]);
				BB_Rects->push_back(BB_Rects_i[ii]);
				BB_Scales->push_back(1 / BB_Scales_i[ii]);
			}
			BB_Rects_i.clear();
			BB_Scores_i.clear();
			mtx->unlock();
		}
	}

};

void HSGDetector::GammaCorrection(Mat& src, Mat& dst, float fGamma) {
	unsigned char lut[256];
	for (int i = 0; i < 256; i++)
		lut[i] = saturate_cast<uchar>(pow((float)(i / 255.0), fGamma) * 255.0f);

	dst = src.clone();
	MatIterator_<Vec3b> it, end;
	for (it = dst.begin<Vec3b>(), end = dst.end<Vec3b>(); it != end; it++)
	{
		(*it)[0] = lut[((*it)[0])];
		(*it)[1] = lut[((*it)[1])];
		(*it)[2] = lut[((*it)[2])];
	}
	/*Mat correctGamma(Mat& img, double gamma) {
	double inverse_gamma = 1.0 / gamma;

	Mat lut_matrix(1, 256, CV_8UC1);
	uchar * ptr = lut_matrix.ptr();
	for (int i = 0; i < 256; i++)
	ptr[i] = (int)(pow((double)i / 255.0, inverse_gamma) * 255.0);

	Mat result;
	LUT(img, lut_matrix, result);

	return result;
	}*/
}

void HSGDetector::MultiScale_Detector(Mat& Frame_, vector<Rect>& BB_Rects, vector<double>& BB_Scores) {
	//Pre-processing
	Mat Frame;
	blur(Frame_, Frame, pre_filter, Point(-1, -1));
	GammaCorrection(Frame, Frame, Gamma);

	//Clear Data structures if they are not already
	BB_Rects.clear();
	BB_Scores.clear();
	vector<double> BB_Scales;		//Detected Scales, used in sync with BB_Rects and BB_Scores

	int c1 = 0;
	//1- Setup detection Scales
	vector<double> Scales;
	float scale = (float)WinSize.height / min_object_height;
	while ((floor(Frame.rows*scale) > (WinSize.height + 2)) && floor(Frame.cols*scale) > (WinSize.width + 2)) {
		c1++;
		//c1 = 1;
		switch (c1) {
		case 1:
		case 2:
		case 7:
		case 8:
		case 12:
		case 13:
		case 18:
		case 19:
		{
			Scales.push_back(scale);
			//cout << endl << scale;
			break;
		}
		}

#ifdef Scale_Linear
		scale = scale - Scale_Step;
#else
		scale = scale / ScaleStride;
#endif
	}

	//2- Spawn parallel processing of all scales
	//cout << "\n\tNumber of scales = " << Scales.size();
	//getchar();
	Mutex mtx;
	parallel_for_(cv::Range(0, Scales.size()), MultiScale_Detector_Parallel_Process(this, Frame, &BB_Rects, &BB_Scores, &BB_Scales, &Scales, &mtx));

	//3- Non-Maxima Suppression
	switch (nms) {
	case GroupRectangles_OCV: {
		vector<int> levels(BB_Rects.size(), 0);
		groupRectangles(BB_Rects, levels, BB_Scores, 1, 0.2);
		break;
	}
	case MeanShift_OCV:
		NMS_DetectedBB_MeanShift(BB_Rects, BB_Scores, BB_Scales, SVM_Score_Th, WinSize);
		break;
	case Custom_NMS:
		NMS_Custom(BB_Rects, BB_Scores, BB_Scales, SVM_Score_Th, Size(23, 56));// 20, 48));// 39, 96));
	}
}
