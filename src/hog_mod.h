#ifndef _hog_mod_
#define _hog_mod_
//		OpenCV
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//		System
#include <fstream>
#include <sys/timeb.h>
#include <time.h>
#include <iostream>
#include <stdio.h>
#include <direct.h>
#include <io.h>
#include <cmath>
#include <algorithm>

//		Exclusive to Windows environment
#ifndef __linux__
#define NOMINMAX
#include <Windows.h>
#include <ppl.h>
#include <xmmintrin.h>
#include <emmintrin.h>
using namespace concurrency;
#endif

using namespace std;
using namespace cv;

extern timeb t_start, t_end;
void hog_mod_test(void);

#define UseSSE

struct HOGDescriptor_Mod
{
public:
	enum {
		L2Hys = 0
	};
	enum {
		DEFAULT_NLEVELS = 64
	};

	HOGDescriptor_Mod() : winSize(64, 128), blockSize(16, 16), blockStride(8, 8),
		cellSize(8, 8), nbins(9), derivAperture(1), winSigma(-1),
		histogramNormType(HOGDescriptor_Mod::L2Hys), L2HysThreshold(0.2), gammaCorrection(true),
		free_coef(-1.f), nlevels(HOGDescriptor_Mod::DEFAULT_NLEVELS), signedGradient(false)
	{}

	HOGDescriptor_Mod(Size _winSize, Size _blockSize, Size _blockStride,
		Size _cellSize, int _nbins, int _derivAperture = 1, double _winSigma = -1,
		int _histogramNormType = HOGDescriptor_Mod::L2Hys,
		double _L2HysThreshold = 0.2, bool _gammaCorrection = false,
		int _nlevels = HOGDescriptor_Mod::DEFAULT_NLEVELS, bool _signedGradient = false)
		: winSize(_winSize), blockSize(_blockSize), blockStride(_blockStride), cellSize(_cellSize),
		nbins(_nbins), derivAperture(_derivAperture), winSigma(_winSigma),
		histogramNormType(_histogramNormType), L2HysThreshold(_L2HysThreshold),
		gammaCorrection(_gammaCorrection), free_coef(-1.f), nlevels(_nlevels), signedGradient(_signedGradient)
	{}

	HOGDescriptor_Mod(const String& filename)
	{
		load(filename);
	}

	HOGDescriptor_Mod(const HOGDescriptor_Mod& d)
	{
		d.copyTo(*this);
	}

	virtual ~HOGDescriptor_Mod() {}

	size_t getDescriptorSize() const;
	bool checkDetectorSize() const;
	double getWinSigma() const;

	virtual void setSVMDetector(InputArray _svmdetector);

	virtual bool read(FileNode& fn);
	virtual void write(FileStorage& fs, const String& objname) const;

	virtual bool load(const String& filename, const String& objname = String());
	virtual void save(const String& filename, const String& objname = String()) const;
	virtual void copyTo(HOGDescriptor_Mod& c) const;

	virtual void compute(InputArray img,
		std::vector<float>& descriptors,
		Size winStride = Size(), Size padding = Size(),
		const std::vector<Point>& locations = std::vector<Point>()) const;

	//! with found weights output
	virtual void detect(const Mat& img, std::vector<Point>& foundLocations,
		std::vector<double>& weights,
		double hitThreshold = 0, Size winStride = Size(),
		Size padding = Size(),
		const std::vector<Point>& searchLocations = std::vector<Point>()) const;
	//! without found weights output
	virtual void detect(const Mat& img, std::vector<Point>& foundLocations,
		double hitThreshold = 0, Size winStride = Size(),
		Size padding = Size(),
		const std::vector<Point>& searchLocations = std::vector<Point>()) const;

	//! with result weights output
	virtual void detectMultiScale(InputArray img, std::vector<Rect>& foundLocations,
		std::vector<double>& foundWeights, double hitThreshold = 0,
		Size winStride = Size(), Size padding = Size(), double scale = 1.05,
		double finalThreshold = 2.0, bool useMeanshiftGrouping = false) const;
	//! without found weights output
	virtual void detectMultiScale(InputArray img, std::vector<Rect>& foundLocations,
		double hitThreshold = 0, Size winStride = Size(),
		Size padding = Size(), double scale = 1.05,
		double finalThreshold = 2.0, bool useMeanshiftGrouping = false) const;

	virtual void computeGradient(const Mat& img, Mat& grad, Mat& angleOfs,
		Size paddingTL = Size(), Size paddingBR = Size()) const;

	static std::vector<float> HOG_Optimal_64_128();
	static std::vector<float> HOG_Optimal_48_120();
	static std::vector<float> HOG_Optimal_48_96();
	static std::vector<float> HOG_Optimal_32_64();
	static std::vector<float> HOG_Caltech_64_128();

	Size winSize;
	Size blockSize;
	Size blockStride;
	Size cellSize;
	int nbins;
	int derivAperture;
	double winSigma;
	int histogramNormType;
	double L2HysThreshold;
	bool gammaCorrection;
	UMat oclSvmDetector;
	float free_coef;
	int nlevels;
	bool signedGradient;
	enum { SVM_Dot_Product, SVM_LUT };//Dot product always for linear, LUT could be for both linear and kernel
	int SVM_Eval_Method;
	std::vector<float> svmDetector;
	float** svmDetectorLUT;
	int svmDetectorLUT_Q;
	float svmDetectorBias;

	//! evaluate specified ROI and return confidence value for each location
	virtual void detectROI(const cv::Mat& img, const std::vector<cv::Point> &locations,
		std::vector<cv::Point>& foundLocations, std::vector<double>& confidences,
		double hitThreshold = 0, cv::Size winStride = Size(),
		cv::Size padding = Size()) const;

	//! evaluate specified ROI and return confidence value for each location in multiple scales
	virtual void detectMultiScaleROI(const cv::Mat& img,
		std::vector<cv::Rect>& foundLocations,
		std::vector<DetectionROI>& locations,
		double hitThreshold = 0,
		int groupThreshold = 0) const;

	//! read/parse Dalal's alt model file
	void readALTModel(String modelfile);
	void groupRectangles(std::vector<cv::Rect>& rectList, std::vector<double>& weights, int groupThreshold, double eps) const;
};

void clipObjects_mod(Size sz, std::vector<Rect>& objects,
	std::vector<int>* a, std::vector<double>* b);



#endif