#include "NMS.h"
/*
	Non-Maximum Suppression for detected Bounding Boxes, code taken from OpenCV cascadedetect.cpp
*/
class MSGrouping
{
public:
	MSGrouping(const Point3d& densKer, const std::vector<Point3d>& posV,
		const std::vector<double>& wV, double eps, int maxIter = 20)
	{
		densityKernel = densKer;
		weightsV = wV;
		positionsV = posV;
		positionsCount = (int)posV.size();
		meanshiftV.resize(positionsCount);
		distanceV.resize(positionsCount);
		iterMax = maxIter;
		modeEps = eps;

		for (unsigned i = 0; i<positionsV.size(); i++)
		{
			meanshiftV[i] = getNewValue(positionsV[i]);
			distanceV[i] = moveToMode(meanshiftV[i]);
			meanshiftV[i] -= positionsV[i];
		}
	}

	void getModes(std::vector<Point3d>& modesV, std::vector<double>& resWeightsV, const double eps)
	{
		for (size_t i = 0; i <distanceV.size(); i++)
		{
			bool is_found = false;
			for (size_t j = 0; j<modesV.size(); j++)
			{
				if (getDistance(distanceV[i], modesV[j]) < eps)
				{
					is_found = true;
					break;
				}
			}
			if (!is_found)
			{
				modesV.push_back(distanceV[i]);
			}
		}

		resWeightsV.resize(modesV.size());

		for (size_t i = 0; i<modesV.size(); i++)
		{
			resWeightsV[i] = getResultWeight(modesV[i]);
		}
	}

protected:
	std::vector<Point3d> positionsV;
	std::vector<double> weightsV;

	Point3d densityKernel;
	int positionsCount;

	std::vector<Point3d> meanshiftV;
	std::vector<Point3d> distanceV;
	int iterMax;
	double modeEps;

	Point3d getNewValue(const Point3d& inPt) const
	{
		Point3d resPoint(.0);
		Point3d ratPoint(.0);
		for (size_t i = 0; i<positionsV.size(); i++)
		{
			Point3d aPt = positionsV[i];
			Point3d bPt = inPt;
			Point3d sPt = densityKernel;

			sPt.x *= std::exp(aPt.z);
			sPt.y *= std::exp(aPt.z);

			aPt.x /= sPt.x;
			aPt.y /= sPt.y;
			aPt.z /= sPt.z;

			bPt.x /= sPt.x;
			bPt.y /= sPt.y;
			bPt.z /= sPt.z;

			double w = (weightsV[i])*std::exp(-((aPt - bPt).dot(aPt - bPt)) / 2) / std::sqrt(sPt.dot(Point3d(1, 1, 1)));

			resPoint += w*aPt;

			ratPoint.x += w / sPt.x;
			ratPoint.y += w / sPt.y;
			ratPoint.z += w / sPt.z;
		}
		resPoint.x /= ratPoint.x;
		resPoint.y /= ratPoint.y;
		resPoint.z /= ratPoint.z;
		return resPoint;
	}

	double getResultWeight(const Point3d& inPt) const
	{
		double sumW = 0;
		double max = -100;
		int cnt = 0;
		for (size_t i = 0; i<positionsV.size(); i++)
		{
			Point3d aPt = positionsV[i];
			Point3d sPt = densityKernel;

			sPt.x *= std::exp(aPt.z);
			sPt.y *= std::exp(aPt.z);

			aPt -= inPt;

			aPt.x /= sPt.x;
			aPt.y /= sPt.y;
			aPt.z /= sPt.z;

			sumW += (weightsV[i])*std::exp(-(aPt.dot(aPt)) / 2) / std::sqrt(sPt.dot(Point3d(1, 1, 1)));
			//float d = std::exp(-(aPt.dot(aPt)) / 2) / std::sqrt(sPt.dot(Point3d(1, 1, 1)));
			//printf("\nD = %.2f", d);
			//if (d > 1e-5){
				//cnt++;
				//sumW += weightsV[i];
			//	if (weightsV[i]>max)
				//	max = weightsV[i];
			//}
				//max = weightsV[i];
		}
		//sumW = max;
		//printf("\n%.2f",sumW);
		//cnt = 5;
		//printf("\nCnt = %d", cnt);
		//if (cnt < 3)
			//sumW = -10;
		//sumW = sumW - 0.1;

		return sumW;
	}

	Point3d moveToMode(Point3d aPt) const
	{
		Point3d bPt;
		for (int i = 0; i<iterMax; i++)
		{
			bPt = aPt;
			aPt = getNewValue(bPt);
			if (getDistance(aPt, bPt) <= modeEps)
			{
				break;
			}
		}
		return aPt;
	}

	double getDistance(Point3d p1, Point3d p2) const
	{
		Point3d ns = densityKernel;
		//ns.x *= std::exp(p2.z);
		//ns.y *= std::exp(p2.z);
		p2 -= p1;
		p2.x /= ns.x;
		p2.y /= ns.y;
		p2.z /= ns.z;
		return p2.dot(p2);
	}
};

static void nms_meanshift(std::vector<Rect>& rectList, double detectThreshold, std::vector<double>* foundWeights,
	std::vector<double>& scales, Size winDetSize)
{
	int detectionCount = (int)rectList.size();
	std::vector<Point3d> hits(detectionCount), resultHits;
	std::vector<double> hitWeights(detectionCount), resultWeights;
	Point2d hitCenter;

	for (int i = 0; i < detectionCount; i++)
	{
		hitWeights[i] = (*foundWeights)[i];
		hitCenter = (rectList[i].tl() + rectList[i].br())*(0.5); //center of rectangles
		hits[i] = Point3d(hitCenter.x, hitCenter.y, std::log(scales[i]));
	}

	rectList.clear();
	if (foundWeights)
		foundWeights->clear();

//	double logZ = std::log(1.3);
//	Point3d smothing(8, 16, logZ);

	//Best configuration so far
//	double logZ = std::log(2);
//	Point3d smothing(4, 8, logZ);

	double logZ = std::log(2);
	Point3d smothing(4, 8, logZ);

	MSGrouping msGrouping(smothing, hits, hitWeights, 1e-5, 100);// 1e-5, 100);

	msGrouping.getModes(resultHits, resultWeights, 1);

	for (unsigned i = 0; i < resultHits.size(); ++i)
	{

		double scale = std::exp(resultHits[i].z);
		hitCenter.x = resultHits[i].x;
		hitCenter.y = resultHits[i].y;
		Size s(int(winDetSize.width * scale), int(winDetSize.height * scale));
		Rect resultRect(int(hitCenter.x - s.width / 2), int(hitCenter.y - s.height / 2),
			int(s.width), int(s.height));

		if (resultWeights[i] > detectThreshold)
		{
			rectList.push_back(resultRect);
			foundWeights->push_back(resultWeights[i]);
		}
	}
}

void NMS_DetectedBB_MeanShift(std::vector<Rect>& rectList, std::vector<double>& foundWeights,
	std::vector<double>& foundScales, double detectThreshold, Size winDetSize)
{
	nms_meanshift(rectList, detectThreshold, &foundWeights, foundScales, winDetSize);
}

void NMS_Custom(std::vector<Rect>& rectList, std::vector<double>& foundWeights,	std::vector<double>& foundScales, double detectThreshold, Size winDetSize){
	int tot_rects = rectList.size();
	Rect a, b;
	int cnt = 0;

	sort_BB(rectList, foundWeights, foundScales);

	for (int i = 0; i < tot_rects; i++){
		cnt = 0;
			for (int j = i + 1; j < tot_rects; j++){
				if (foundWeights[i]>-10){
				a = rectList[i] & rectList[j];	//intersection
				//b = rectList[i] | rectList[j];	//union
				if (a.area()>(min(rectList[i].area(), rectList[j].area())*0.5)){		//if intersection area is greater than some % of union area
					cnt++;
					if (foundWeights[i] > foundWeights[j]){
//						foundWeights[i] += foundWeights[j];
						//foundWeights[i] += 1;// foundWeights[i];
						foundWeights[j] = -10;
						//break;
					}
					else{
//						foundWeights[j] += foundWeights[i];
						//foundWeights[j] += 1;// foundWeights[j];
						foundWeights[i] = -10;
					}
				}
			}
		}
			if (foundWeights[i] > -10)
				foundWeights[i] = foundWeights[i]*pow((double)(cnt + 1), 2);
//			foundWeights[i] = (foundWeights[i] * 3) + (double)(cnt + 1);// *(double)(cnt + 1);
			//if (cnt<5)	foundWeights[i] = -10;
			//if (foundWeights[i]<2)	foundWeights[i] = -10;
	}

/*	for (int i = 0; i < tot_rects; i++){
		if (foundWeights[i] < -5){
			rectList.erase(rectList.begin() + i);
			foundWeights.erase(foundWeights.begin() + i);
			foundScales.erase(foundScales.begin() + i);
		}
	}*/
}

void sort_BB(std::vector<Rect>& rectList, std::vector<double>& foundWeights, std::vector<double>& foundScales){
	Rect swap_rect;
	double swap_weight;
	double swap_scale;

	for (int i = 0; i < rectList.size(); i++){
		for (int j = i+1; j < rectList.size(); j++){
			if (foundWeights[j] > foundWeights[i]){
				swap_rect = rectList[i];
				swap_weight = foundWeights[i];
				swap_scale = foundScales[i];
				rectList[i] = rectList[j];
				foundWeights[i] = foundWeights[j];
				foundScales[i] = foundScales[j];
				rectList[j] = swap_rect;
				foundWeights[j] = swap_weight;
				foundScales[j] = swap_scale;
			}
		}
	}
}

void NMS_Custom2(std::vector<Rect>& rectList, int groupThreshold, double eps, std::vector<int>* weights, std::vector<double>* levelWeights){
	if (groupThreshold <= 0 || rectList.empty())
	{
		if (weights)
		{
			size_t i, sz = rectList.size();
			weights->resize(sz);
			for (i = 0; i < sz; i++)
				(*weights)[i] = 1;
		}
		return;
	}

	std::vector<int> labels;
	int nclasses = partition(rectList, labels, SimilarRects(eps));

	std::vector<Rect> rrects(nclasses);
	std::vector<int> rweights(nclasses, 0);
	std::vector<int> rejectLevels(nclasses, 0);
	std::vector<double> rejectWeights(nclasses, DBL_MIN);
	int i, j, nlabels = (int)labels.size();
	for (i = 0; i < nlabels; i++)
	{
		int cls = labels[i];
		rrects[cls].x += rectList[i].x;
		rrects[cls].y += rectList[i].y;
		rrects[cls].width += rectList[i].width;
		rrects[cls].height += rectList[i].height;
		rweights[cls]++;
	}

	bool useDefaultWeights = false;

	if (levelWeights && weights && !weights->empty() && !levelWeights->empty())
	{
		for (i = 0; i < nlabels; i++)
		{
			int cls = labels[i];
			if ((*weights)[i] > rejectLevels[cls])
			{
				rejectLevels[cls] = (*weights)[i];
				rejectWeights[cls] = (*levelWeights)[i];
			}
			else if (((*weights)[i] == rejectLevels[cls]) && ((*levelWeights)[i] > rejectWeights[cls]))
				rejectWeights[cls] = (*levelWeights)[i];
		}
	}
	else
		useDefaultWeights = true;

	for (i = 0; i < nclasses; i++)
	{
		Rect r = rrects[i];
		float s = 1.f / rweights[i];
		rrects[i] = Rect(saturate_cast<int>(r.x*s),
			saturate_cast<int>(r.y*s),
			saturate_cast<int>(r.width*s),
			saturate_cast<int>(r.height*s));
	}

	rectList.clear();
	if (weights)
		weights->clear();
	if (levelWeights)
		levelWeights->clear();

	for (i = 0; i < nclasses; i++)
	{
		Rect r1 = rrects[i];
		int n1 = rweights[i];
		double w1 = rejectWeights[i];
		int l1 = rejectLevels[i];

		// filter out rectangles which don't have enough similar rectangles
		if (n1 <= groupThreshold)
			continue;
		// filter out small face rectangles inside large rectangles
		for (j = 0; j < nclasses; j++)
		{
			int n2 = rweights[j];

			if (j == i || n2 <= groupThreshold)
				continue;
			Rect r2 = rrects[j];

			int dx = saturate_cast<int>(r2.width * eps);
			int dy = saturate_cast<int>(r2.height * eps);

			if (i != j &&
				r1.x >= r2.x - dx &&
				r1.y >= r2.y - dy &&
				r1.x + r1.width <= r2.x + r2.width + dx &&
				r1.y + r1.height <= r2.y + r2.height + dy &&
				(n2 > std::max(3, n1) || n1 < 3))
				break;
		}

		if (j == nclasses)
		{
			rectList.push_back(r1);
			if (weights)
				weights->push_back(useDefaultWeights ? n1 : l1);
			if (levelWeights)
				levelWeights->push_back(w1);
		}
	}
}


void NMS_test(void){
	Mat img = imread("D:\\RnD\\Current_Projects\\Musawwir\\Object_Detection\\Frameworks\\SW\\Dataset\\Person\\Caltech\\code\\data-INRIA\\images\\set01\\V000\\I00003.png");
	float r = 0.5;
	resize(img, img, Size(), r, r);
	namedWindow("Test NMS");// , WINDOW_NORMAL);
	std::vector<Rect> BB_Rects;
	std::vector<double> BB_Scores;
	std::vector<Rect> BB_Rects2;
	std::vector<Rect> BB_Rects_Orig;
	std::vector<double> BB_Scores_Orig;
	std::vector<double> BB_Scores2;
	std::vector<double> BB_Scales;
	double threshold = 0;
	Point2d p(r*600, r*475);
	int w = 64, h = 128;
	float sc_max = 0.4;
	float sc_min = 0.35;
	float sc_step = 1.05;
	float sc_detect = 0.375;
	float score_max = sc_max - sc_min;

	for (float sc = sc_max; sc > sc_min; sc = sc / sc_step){
		Rect bb;
		bb.x = p.x - (float)w / sc*0.5;// +rand() % 35 - 17.5;
		bb.y = p.y - (float)h / sc*0.5;// +rand() % 35 - 17.5;
		bb.width = (float)w / sc;
		bb.height = (float)h / sc;
		BB_Rects.push_back(bb);
		double score = (score_max - abs(sc_detect - sc)) / score_max;
		std::cout << "\n" << score;
		BB_Scores.push_back(score);
		BB_Scales.push_back(1/sc);
		rectangle(img, bb, Scalar(255, 0, 0), 2);
	}

	cout << "\n Number of input BBs = " << BB_Rects.size();
	std::vector<int> levels(BB_Rects.size(), 0);
	BB_Rects_Orig = BB_Rects;
	BB_Scores_Orig = BB_Scores;
	BB_Rects2 = BB_Rects;
	BB_Scores2 = BB_Scores;
	//groupRectangles(BB_Rects2, levels, BB_Scores2, 2, 0.2);
	NMS_DetectedBB_MeanShift(BB_Rects, BB_Scores, BB_Scales, threshold, Size(39, 96));
	std::cout << "\n Number of output BBs = " << BB_Rects.size();

	float final_score;
	for (int x = 0; x < BB_Rects.size(); x++){
		final_score = 0;
		for (int y = 0; y < BB_Rects_Orig.size(); y++){
			Rect a = BB_Rects[x] & BB_Rects_Orig[y];	//intersection
			Rect b = BB_Rects[x] | BB_Rects_Orig[y];	//union
			/*printf("\nR1 = (%d,%d) - (%d,%d)", BB_Rects[x].x, BB_Rects[x].y, BB_Rects[x].width, BB_Rects[x].height);
			printf("\nR2 = (%d,%d) - (%d,%d)", BB_Rects_Orig[x].x, BB_Rects_Orig[x].y, BB_Rects_Orig[x].width, BB_Rects_Orig[x].height);
			printf("\na = (%d,%d) - (%d,%d)", a.x, a.y, a.width, a.height);
			printf("\nb = (%d,%d) - (%d,%d)", b.x, b.y, b.width, b.height);
			printf("\nIntersection Area = %d", a.area());
			printf("\tUnion Area = %d", b.area());*/
			if (a.area()>b.area()*0.2){
				final_score += BB_Scores_Orig[y];
				//printf("\nMatched.. BB_Scores_Orig[%d]=%.2f", y, BB_Scores_Orig[y]);
			}
		}
		BB_Scores[x] = final_score;
		printf("\nBB_Scores[%d]=%.2f", x, final_score);
	}


	for (int x = 0; x < BB_Scores.size(); x++){
		Rect detected_bb;
		char str[50];
		sprintf(str, "%.02f", BB_Scores[x]);
		detected_bb = BB_Rects[x];
		cv::putText(img, str, Point(detected_bb.x, detected_bb.y), CV_FONT_HERSHEY_COMPLEX, 1, Scalar(255, 0, 0), 1, 8);
		rectangle(img, detected_bb, Scalar(0, 0, 255), 2);
		detected_bb = BB_Rects2[x];
		//rectangle(img, detected_bb, Scalar(0, 255, 0), 2);
		std::cout << "\n" << BB_Scores[x];
		//std::cout << "\n Output Scale = " << (float)w / (float)detected_bb.width;
		//std::cout << "\n Output Width = " << (float)detected_bb.width;
	}

	imshow("Test NMS", img);
	waitKey(0);
	//getchar();
}