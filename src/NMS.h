//#include "precomp.hpp"
#include <cstdio>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/ml/ml.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void NMS_DetectedBB_MeanShift(std::vector<Rect>& rectList, std::vector<double>& foundWeights,std::vector<double>& foundScales, double detectThreshold, Size winDetSize);
void NMS_Custom(std::vector<Rect>& rectList, std::vector<double>& foundWeights, std::vector<double>& foundScales, double detectThreshold, Size winDetSize);
void sort_BB(std::vector<Rect>& rectList, std::vector<double>& foundWeights, std::vector<double>& foundScales);
void NMS_test(void);