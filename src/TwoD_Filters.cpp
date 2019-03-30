#include "TwoD_Filters.h"



#ifdef Code_Junk
/*
void cartToPolar_Mod_OCV(InputArray src1, InputArray src2, OutputArray dst1, OutputArray dst2, bool angleInDegrees){
	CV_INSTRUMENT_REGION()

		CV_OCL_RUN(dst1.isUMat() && dst2.isUMat(),
			ocl_cartToPolar(src1, src2, dst1, dst2, angleInDegrees))

		Mat X = src1.getMat(), Y = src2.getMat();
	int type = X.type(), depth = X.depth(), cn = X.channels();
	CV_Assert(X.size == Y.size && type == Y.type() && (depth == CV_32F || depth == CV_64F));
	dst1.create(X.dims, X.size, type);
	dst2.create(X.dims, X.size, type);
	Mat Mag = dst1.getMat(), Angle = dst2.getMat();

	const Mat* arrays[] = { &X, &Y, &Mag, &Angle, 0 };
	uchar* ptrs[4];
	NAryMatIterator it(arrays, ptrs);
	int j, total = (int)(it.size*cn), blockSize = std::min(total, ((BLOCK_SIZE + cn - 1) / cn)*cn);
	size_t esz1 = X.elemSize1();

	for (size_t i = 0; i < it.nplanes; i++, ++it)
	{
		for (j = 0; j < total; j += blockSize)
		{
			int len = std::min(total - j, blockSize);
			if (depth == CV_32F)
			{
				const float *x = (const float*)ptrs[0], *y = (const float*)ptrs[1];
				float *mag = (float*)ptrs[2], *angle = (float*)ptrs[3];
				hal::magnitude32f(x, y, mag, len);
				hal::fastAtan32f(y, x, angle, len, angleInDegrees);
			}
			else
			{
				const double *x = (const double*)ptrs[0], *y = (const double*)ptrs[1];
				double *angle = (double*)ptrs[3];
				hal::magnitude64f(x, y, (double*)ptrs[2], len);
				hal::fastAtan64f(y, x, angle, len, angleInDegrees);
			}
			ptrs[0] += len*esz1;
			ptrs[1] += len*esz1;
			ptrs[2] += len*esz1;
			ptrs[3] += len*esz1;
		}
	}
}
*/


Mat Filters_Speed_Test(Mat src, int option) {
	Mat dst, src_gray, temp;

	switch (option) {
	case 0:	//Canny Edge Detector		4.7 msec HD Video
		cvtColor(src, temp, CV_BGR2GRAY);
		Canny(temp, temp, 50, 150, 3);
		cvtColor(temp, dst, CV_GRAY2BGR);
		break;
	case 1:	//Simple Blur				6.5 msec HD Video
			//cvtColor(src, src_gray, CV_BGR2GRAY);
		blur(src, dst, Size(3, 3));
		//cvtColor(temp, dst, CV_GRAY2BGR);
		break;
	case 2:	//Gaussian Blur				4.6 msec HD Video
			//cvtColor(src, src_gray, CV_BGR2GRAY);
		GaussianBlur(src, dst, Size(3, 3), 0, 0, BORDER_DEFAULT);
		//cvtColor(temp, dst, CV_GRAY2BGR);
		break;
	case 3:	//Laplacian (OpenCV Gradient)	4.1 msec HD Video
		cvtColor(src, src_gray, CV_BGR2GRAY);
		Laplacian(src_gray, temp, CV_16S, 3, 1, 0, BORDER_DEFAULT);
		convertScaleAbs(temp, temp);
		cvtColor(temp, dst, CV_GRAY2BGR);
		break;
	case 4: {//Gradient (Old func. used by HSG, based on parallel for loops)	54.5/10.7 msec without/with parallelization on HD Video
		float* Grad_Mag;
		unsigned char* Grad_Or;
		Grad_Mag = new float[src.rows*src.cols];
		Grad_Or = new unsigned char[src.rows*src.cols];
		ftime(&t_start);
		Calc_Grad_Img(src, Grad_Mag, Grad_Or);
		ftime(&t_end);
		Mat t(src.rows, src.cols, CV_32F, Grad_Mag);
		t.convertTo(t, CV_8U);
		t.copyTo(dst);
		t.release();
		delete Grad_Mag;
		delete Grad_Or;
		break; }
	case 5: {//Gradient (Adopted from OpenCV HOG function)	30.4/15.8 msec without/with SSE on HD Video
		ftime(&t_start);
		computeGradient(src, 0, 0);// , Size(0, 0), Size(0, 0));
		ftime(&t_end);
		src.copyTo(dst);
		break; }
	case 6:	//Gradient (Adopted from ACF)	7.9 msec (on planar RGB data) 15.9 msec (including interleaved to planar rgb conversion) on HD Video
		ftime(&t_start);
#ifdef Use_SSE
		dst = Calc_Grad_Img_SSE(src, 0, 0);
#endif
		ftime(&t_end);
		break;
	}


	return dst;
}

Mat Calc_Grad_Img(Mat frame, float* Grad_Mag, unsigned char* Grad_Or) {
	const float pi = 3.141593;	//This value of pi is not sufficient. Use CV_PI
	int rows = frame.rows;
	int cols = frame.cols;
	unsigned char *frame_data_ptr = (unsigned char*)(frame.data);
	int nbins = 9;
	float angleScale = (float)(nbins / CV_PI);

	const float HSG_bin_size = pi / nbins;

#ifdef Use_P4
	parallel_for(size_t(1), size_t(rows - 1), size_t(1), [&](size_t j) {
		parallel_for(size_t(1), size_t(cols - 1), size_t(1), [&](size_t i) {
#else
	for (int j = 1; j < rows - 1; j++) {			//leave boundary pixels
		for (int i = 1; i < cols - 1; i++) {		//leave boundary pixels
#endif
			int dx[3], dy[3];
			int k, temp_mag1, temp_mag2;
			//			float temp_grad;
			int jcols3 = j*cols * 3;
			int cols3 = cols * 3;
			int i3 = i * 3;
			int a1 = jcols3 + i3;
			dx[0] = frame_data_ptr[a1 + 3] - frame_data_ptr[a1 - 3];
			dx[1] = frame_data_ptr[a1 + 4] - frame_data_ptr[a1 - 2];
			dx[2] = frame_data_ptr[a1 + 5] - frame_data_ptr[a1 - 1];
			dy[0] = -frame_data_ptr[a1 + cols3] + frame_data_ptr[a1 - cols3];
			dy[1] = -frame_data_ptr[a1 + cols3 + 1] + frame_data_ptr[a1 - cols3 + 1];
			dy[2] = -frame_data_ptr[a1 + cols3 + 2] + frame_data_ptr[a1 - cols3 + 2];

			temp_mag1 = (dx[0] * dx[0] + dy[0] * dy[0]);// = abs(dx[0]) + abs(dy[0]);
			k = 0;//greatest mag. value index
			temp_mag2 = (dx[1] * dx[1] + dy[1] * dy[1]);// = abs(dx[1]) + abs(dy[1]);
			if (temp_mag2 > temp_mag1) {
				temp_mag1 = temp_mag2;
				k = 1;
			}
			temp_mag2 = (dx[2] * dx[2] + dy[2] * dy[2]);// = abs(dx[2]) + abs(dy[2]);
			if (temp_mag2 > temp_mag1) {
				temp_mag1 = temp_mag2;
				k = 2;
			}
			Grad_Mag[j*cols + i] = int(sqrtf(float(temp_mag1)));	// = temp_mag1; // = (int)temp_mag1/2;
			float Grad_Or_temp = atan2f(float(dy[k]), float(dx[k]));
			if (Grad_Or_temp < 0) Grad_Or_temp += pi;
			if (Grad_Or_temp > pi) Grad_Or_temp -= pi;

			Grad_Or[j*cols + i] = (int)floor(Grad_Or_temp / HSG_bin_size);
			//Grad_Or_temp *= angleScale;
			//Grad_Or_temp = cvFloor(Grad_Or_temp);
			//Grad_Or[j*cols + i] = (unsigned char)Grad_Or_temp;
#ifdef Use_P4
		});
	});
#else
		}
	}
#endif
	return frame;
}




// Constants for rgb2luv conversion and lookup table for y-> l conversion
template<class oT> oT* rgb2luv_setup(oT z, oT *mr, oT *mg, oT *mb,
	oT &minu, oT &minv, oT &un, oT &vn)
{
	// set constants for conversion
	const oT y0 = (oT)((6.0 / 29)*(6.0 / 29)*(6.0 / 29));
	const oT a = (oT)((29.0 / 3)*(29.0 / 3)*(29.0 / 3));
	un = (oT) 0.197833; vn = (oT) 0.468331;

	//http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
	//PAL/SECAM RGB
//	mr[0] = (oT) 0.430574*z; mr[1] = (oT) 0.222015*z; mr[2] = (oT) 0.020183*z;
//	mg[0] = (oT) 0.341550*z; mg[1] = (oT) 0.706655*z; mg[2] = (oT) 0.129553*z;
//	mb[0] = (oT) 0.178325*z; mb[1] = (oT) 0.071330*z; mb[2] = (oT) 0.939180*z;

	//sRGB
	mr[0] = (oT) 0.412453*z; mg[0] = (oT) 0.357580*z; mb[0] = (oT) 0.180423*z;
	mr[1] = (oT) 0.212671*z; mg[1] = (oT) 0.715160*z; mb[1] = (oT) 0.072169*z;
	mr[2] = (oT) 0.019334*z; mg[2] = (oT) 0.119193*z; mb[2] = (oT) 0.950227*z;

	oT maxi = (oT) 1.0 / 270; minu = -88 * maxi; minv = -134 * maxi;
	// build (padded) lookup table for y->l conversion assuming y in [0,1]
	static oT lTable[1064]; static bool lInit = false;
	if (lInit) return lTable; oT y, l;
	for (int i = 0; i<1025; i++) {
		y = (oT)(i / 1024.0);
		l = y>y0 ? 116 * (oT)pow((double)y, 1.0 / 3.0) - 16 : y*a;
		lTable[i] = l*maxi;
	}
	for (int i = 1025; i<1064; i++) lTable[i] = lTable[i - 1];
	lInit = true; return lTable;
}

// Convert from rgb to luv
template<class iT, class oT> void rgb2luv(iT *I, unsigned char *J, int n, oT nrm) {
	oT minu, minv, un, vn, mr[3], mg[3], mb[3];
	oT *lTable = rgb2luv_setup(nrm, mr, mg, mb, minu, minv, un, vn);
	const oT y0 = 0.008856;// (oT)((6.0 / 29)*(6.0 / 29)*(6.0 / 29));
	const oT a = 903.3;// (oT)((29.0 / 3)*(29.0 / 3)*(29.0 / 3));
	un = (oT) 0.19793943; vn = (oT) 0.46831096;
	float z = 1;// (float)1 / (float)1024;

//	oT *L = J, *U = L + n, *V = U + n;
//	iT *R = I, *G = R + n, *B = G + n;
	unsigned char *L = J, *U = L + 1, *V = U + 1;
	unsigned char *R = I, *G = R + 1, *B = G + 1;
	for (int i = 0; i<n; i++) {
		float r, g, b, x, y, z, l, u, v;
		r = (float)*R; g = (float)*G; b = (float)*B;
		r = r / 255; g = g / 255; b = b / 255;
		//r = r <= 0.04045f ? r*(1.f / 12.92f) : (float)std::pow((double)(r + 0.055)*(1. / 1.055), 2.4);
		//g = g <= 0.04045f ? g*(1.f / 12.92f) : (float)std::pow((double)(g + 0.055)*(1. / 1.055), 2.4);
		//b = b <= 0.04045f ? b*(1.f / 12.92f) : (float)std::pow((double)(b + 0.055)*(1. / 1.055), 2.4);

		x = mr[0] * r + mg[0] * g + mb[0] * b;
		y = mr[1] * r + mg[1] * g + mb[1] * b;
		z = mr[2] * r + mg[2] * g + mb[2] * b;

		l = y>y0 ? 116 * (float)pow((double)y, 1.0 / 3.0) - 16 : y*a;
		z = 1 / (x + 15 * y + 3 * z + (oT)1e-35);
		u = 13*l * (4*x*z - un);
		v = 13*l * (9*y*z - vn);
		l = l *(float)255.0 / (float)100.0;
		u = (u + 134)*(float)255.0 / (float)354.0;
		v = (v + 140)*(float)255.0 / (float)262.0;
		*(L) = unsigned char(l);
		*(V) = unsigned char(255-u);
		*(U) = unsigned char(255-v);

		R += 3; G += 3; B += 3;
		L += 3; U += 3; V += 3;
	}
}




// Convert from rgb to luv using sse
template<class iT> void rgb2luv_sse(iT *I, float *J, int n, float nrm) {
	printf("\nIn here SSE");
	const int k = 256; float R[k], G[k], B[k];
	if ((size_t(R) & 15 || size_t(G) & 15 || size_t(B) & 15 || size_t(I) & 15 || size_t(J) & 15)
		|| n % 4>0) {
		rgb2luv(I, J, n, nrm); return;
	}
	int i = 0, i1, n1; float minu, minv, un, vn, mr[3], mg[3], mb[3];
	float *lTable = rgb2luv_setup(nrm, mr, mg, mb, minu, minv, un, vn);
	while (i<n) {
		n1 = i + k; if (n1>n) n1 = n; float *J1 = J + i; float *R1, *G1, *B1;
		// convert to floats (and load input into cache)
		if (typeid(iT) != typeid(float)) {
			R1 = R; G1 = G; B1 = B; iT *Ri = I + i, *Gi = Ri + n, *Bi = Gi + n;
			for (i1 = 0; i1<(n1 - i); i1++) {
				R1[i1] = (float)*Ri++; G1[i1] = (float)*Gi++; B1[i1] = (float)*Bi++;
			}
		}
		else { R1 = ((float*)I) + i; G1 = R1 + n; B1 = G1 + n; }
		// compute RGB -> XYZ
		for (int j = 0; j<3; j++) {
			__m128 _mr, _mg, _mb, *_J = (__m128*) (J1 + j*n);
			__m128 *_R = (__m128*) R1, *_G = (__m128*) G1, *_B = (__m128*) B1;
			_mr = SET_SSE(mr[j]); _mg = SET_SSE(mg[j]); _mb = SET_SSE(mb[j]);
			for (i1 = i; i1<n1; i1 += 4) *(_J++) = ADD_SSE(ADD_SSE(MUL_SSE(*(_R++), _mr),
				MUL_SSE(*(_G++), _mg)), MUL_SSE(*(_B++), _mb));
		}
		{ // compute XZY -> LUV (without doing L lookup/normalization)
			__m128 _c15, _c3, _cEps, _c52, _c117, _c1024, _cun, _cvn;
			_c15 = SET_SSE(15.0f); _c3 = SET_SSE(3.0f); _cEps = SET_SSE(1e-35f);
			_c52 = SET_SSE(52.0f); _c117 = SET_SSE(117.0f), _c1024 = SET_SSE(1024.0f);
			_cun = SET_SSE(13 * un); _cvn = SET_SSE(13 * vn);
			__m128 *_X, *_Y, *_Z, _x, _y, _z;
			_X = (__m128*) J1; _Y = (__m128*) (J1 + n); _Z = (__m128*) (J1 + 2 * n);
			for (i1 = i; i1<n1; i1 += 4) {
				_x = *_X; _y = *_Y; _z = *_Z;
				_z = RCP_SSE(ADD_SSE(_x, ADD_SSE(_cEps, ADD_SSE(MUL_SSE(_c15, _y), MUL_SSE(_c3, _z)))));
				*(_X++) = MUL_SSE(_c1024, _y);
				*(_Y++) = SUB_SSE(MUL_SSE(MUL_SSE(_c52, _x), _z), _cun);
				*(_Z++) = SUB_SSE(MUL_SSE(MUL_SSE(_c117, _y), _z), _cvn);
			}
		}
		{ // perform lookup for L and finalize computation of U and V
			for (i1 = i; i1<n1; i1++) J[i1] = lTable[(int)J[i1]];
			__m128 *_L, *_U, *_V, _l, _cminu, _cminv;
			_L = (__m128*) J1; _U = (__m128*) (J1 + n); _V = (__m128*) (J1 + 2 * n);
			_cminu = SET_SSE(minu); _cminv = SET_SSE(minv);
			for (i1 = i; i1<n1; i1 += 4) {
				_l = *(_L++);
				*_U = SUB_SSE(MUL_SSE(_l, *_U), _cminu); _U++;
				*_V = SUB_SSE(MUL_SSE(_l, *_V), _cminv); _V++;
			}
		}
		i = n1;
	}
}
template<class iT, class oT> void normalize(iT *I, oT *J, int n, oT nrm) {
	for (int i = 0; i<n; i++) *(J++) = (oT)*(I++)*nrm;
}

// Convert rgb to various colorspaces
template<class iT, class oT>
oT* rgbConvert(iT *I, int n, int d, int flag, oT nrm) {
	oT *J = (oT*)wrMalloc(n*(flag == 0 ? (d == 1 ? 1 : d / 3) : d)*sizeof(oT));
	int i, n1 = d*(n<1000 ? n / 10 : 100); oT thr = oT(1.001);
	if (flag>1 && nrm == 1) for (i = 0; i<n1; i++) if (I[i]>thr)
		wrError("For floats all values in I must be smaller than 1.");
	bool useSse = n % 4 == 0 && typeid(oT) == typeid(float);
	if (flag == 2 && useSse)
		for (i = 0; i<d / 3; i++) rgb2luv_sse(I + i*n * 3, (float*)(J + i*n * 3), n, (float)nrm);
	else if ((flag == 0 && d == 1) || flag == 1) normalize(I, J, n*d, nrm);
	else if (flag == 0) for (i = 0; i<d / 3; i++) rgb2gray(I + i*n * 3, J + i*n * 1, n, nrm);
	else if (flag == 2) for (i = 0; i<d / 3; i++) rgb2luv(I + i*n * 3, J + i*n * 3, n, nrm);
	else if (flag == 3) for (i = 0; i<d / 3; i++) rgb2hsv(I + i*n * 3, J + i*n * 3, n, nrm);
	else wrError("Unknown flag.");
	return J;
}

void RGB2LUV_ACF(unsigned char* I, unsigned char* J, int n, int d, int flag) {
	float nrm = 1;
	rgb2luv(I, (unsigned char*)J, n, (float)nrm);
}

void test(void) {
	VideoCapture capture;
	Mat frame;
	float proc_t = 0, fps;
	char str[20];
	capture.open(0);
	if (!capture.isOpened()) { printf("--(!)Error opening video capture\n"); return; }
	capture.read(frame);
	Mat luv, out;
	//out.create(480, 640, CV_8UC3);
	//

	cvtColor(frame, luv, CV_BGR2Luv);	//Uses Parallel Loopbody, 
	unsigned char *luv_ptr = (unsigned char*)(luv.data);
	unsigned char *I = (unsigned char *)(frame.data);
	//setNumThreads(0);
	int n = frame.rows * frame.cols;
	namedWindow("OCV Output");
	namedWindow("ACF Output");
	while (capture.read(frame)) {
		cvtColor(frame, luv, CV_BGR2Luv);	//Uses Parallel Loopbody, 3 msec on VGA frame, 15 msec without parallelization
											/*vector<Mat> luvChannels(3);
											split(luv, luvChannels);
											Mat z = Mat::zeros(Size(frame.cols, frame.rows), CV_8UC1);
											vector<Mat> channels;
											channels.push_back(luvChannels[0]);
											channels.push_back(luvChannels[2]);
											channels.push_back(luvChannels[1]);
											merge(channels, luv);
											*/
		imshow("OCV Output", luv);
		ftime(&t_start);
		RGB2LUV_ACF(I, luv_ptr, n, 3, 2);
		ftime(&t_end);
		float msec = int((t_end.time - t_start.time) * 1000 + (t_end.millitm - t_start.millitm));
		proc_t = 0.99*proc_t + 0.01*msec;
		printf("\nfps = %.1f\ttime = %.1f", 1000 / msec, proc_t);
		fps = 1000 / proc_t;
		sprintf(str, "%.1f", fps);
		cv::putText(frame, str, Point(20, 20), CV_FONT_HERSHEY_COMPLEX, 1, Scalar(255, 0, 0), 1, 8);
		imshow("ACF Output", luv);
		waitKey(1);
	}
}

#ifdef Use_SSE
void grad1(float *I, float *Gx, float *Gy, int h, int w, int x) {
	int y, y1; float *Ip, *In, r; __m128 *_Ip, *_In, *_G, _r;
	// compute column of Gx
	Ip = I - h; In = I + h; r = .5f;
	if (x == 0) { r = 1; Ip += h; }
	else if (x == w - 1) { r = 1; In -= h; }
	if (h<4 || h % 4>0 || (size_t(I) & 15) || (size_t(Gx) & 15)) {
		for (y = 0; y<h; y++) *Gx++ = (*In++ - *Ip++)*r;
	}
	else {
		_G = (__m128*) Gx; _Ip = (__m128*) Ip; _In = (__m128*) In; _r = SET_SSE(r);
		for (y = 0; y<h; y += 4) *_G++ = MUL_SSE(SUB_SSE(*_In++, *_Ip++), _r);
	}
	// compute column of Gy
#define GRADY(r) *Gy++=(*In++-*Ip++)*r;
	Ip = I; In = Ip + 1;
	// GRADY(1); Ip--; for(y=1; y<h-1; y++) GRADY(.5f); In--; GRADY(1);
	y1 = ((~((size_t)Gy) + 1) & 15) / 4; if (y1 == 0) y1 = 4; if (y1>h - 1) y1 = h - 1;
	GRADY(1); Ip--; for (y = 1; y<y1; y++) GRADY(.5f);
	_r = SET_SSE(.5f); _G = (__m128*) Gy;
	for (; y + 4<h - 1; y += 4, Ip += 4, In += 4, Gy += 4)
		*_G++ = MUL_SSE(SUB_SSE(LDu_SSE(*In), LDu_SSE(*Ip)), _r);
	for (; y<h - 1; y++) GRADY(.5f); In--; GRADY(1);
#undef GRADY
}

// compute x and y gradients at each location (uses sse)
/*void grad2(float *I, float *Gx, float *Gy, int h, int w, int d) {
	int o, x, c, a = w*h; for (c = 0; c<d; c++) for (x = 0; x<w; x++) {
		o = c*a + x*h; grad1(I + o, Gx + o, Gy + o, h, w, x);
	}
}*/

// build lookup table a[] s.t. a[x*n]~=acos(x) for x in [-1,1]
float* acosTable() {
	const int n = 10000, b = 10; int i;
	static float a[n * 2 + b * 2]; static bool init = false;
	float *a1 = a + n + b; if (init) return a1;
	for (i = -n - b; i<-n; i++)   a1[i] = PI;
	for (i = -n; i<n; i++)      a1[i] = float(acos(i / float(n)));
	for (i = n; i<n + b; i++)     a1[i] = 0;
	for (i = -n - b; i<n / 10; i++) if (a1[i] > PI - 1e-6f) a1[i] = PI - 1e-6f;
	init = true; return a1;
}



// compute gradient magnitude and orientation at each location (uses sse)
void gradMag(float *I, float *M, float *O, int h, int w, int d, bool full) {
	int x, y, y1, c, h4, s; float *Gx, *Gy, *M2; __m128 *_Gx, *_Gy, *_M2, _m;
	float *acost = acosTable(), acMult = 10000.0f;
	// allocate memory for storing one column of output (padded so h4%4==0)
	h4 = (h % 4 == 0) ? h : h - (h % 4) + 4; s = d*h4*sizeof(float);
	M2 = (float*)alMalloc(s, 16); _M2 = (__m128*) M2;
	Gx = (float*)alMalloc(s, 16); _Gx = (__m128*) Gx;
	Gy = (float*)alMalloc(s, 16); _Gy = (__m128*) Gy;
	// compute gradient magnitude and orientation for each column
	for (x = 0; x<w; x++) {
		// compute gradients (Gx, Gy) with maximum squared magnitude (M2)
		for (c = 0; c<d; c++) {
			grad1(I + x*h + c*w*h, Gx + c*h4, Gy + c*h4, h, w, x);
			for (y = 0; y<h4 / 4; y++) {
				y1 = h4 / 4 * c + y;
				_M2[y1] = ADD_SSE(MUL_SSE(_Gx[y1], _Gx[y1]), MUL_SSE(_Gy[y1], _Gy[y1]));
				if (c == 0) continue; _m = CMPGT_SSE(_M2[y1], _M2[y]);
				//Following 3 instructions implement the MUX, i.e. if(_M2[y1] > _M2[y]) _m = _M2[y1], else _m = _M2[y]
				_M2[y] = OR_SSE(AND_SSE(_m, _M2[y1]), ANDNOT_SSE(_m, _M2[y]));
				_Gx[y] = OR_SSE(AND_SSE(_m, _Gx[y1]), ANDNOT_SSE(_m, _Gx[y]));
				_Gy[y] = OR_SSE(AND_SSE(_m, _Gy[y1]), ANDNOT_SSE(_m, _Gy[y]));
			}
		}
		// compute gradient mangitude (M) and normalize Gx
		for (y = 0; y<h4 / 4; y++) {
			_m = MIN_SSE(RCPSQRT_SSE(_M2[y]), SET_SSE(1e10f));
			_M2[y] = RCP_SSE(_m);
			_M2[y] = MUL_SSE(_M2[y], 32);
			if (O) _Gx[y] = MUL_SSE(MUL_SSE(_Gx[y], _m), SET_SSE(acMult));
			if (O) _Gx[y] = XOR_SSE(_Gx[y], AND_SSE(_Gy[y], SET_SSE(-0.f)));
		};
		memcpy(M + x*h, M2, h*sizeof(float));
		// compute and store gradient orientation (O) via table lookup
		if (O != 0) for (y = 0; y<h; y++) O[x*h + y] = acost[(int)Gx[y]];
		if (O != 0 && full) {
			y1 = ((~size_t(O + x*h) + 1) & 15) / 4; y = 0;
			for (; y<y1; y++) O[y + x*h] += (Gy[y]<0)*PI;
			for (; y<h - 4; y += 4) STRu_SSE(O[y + x*h],
				ADD_SSE(LDu_SSE(O[y + x*h]), AND_SSE(CMPLT_SSE(LDu_SSE(Gy[y]), SET_SSE(0.f)), SET_SSE(PI))));
			for (; y<h; y++) O[y + x*h] += (Gy[y]<0)*PI;
		}
	}
	alFree(Gx); alFree(Gy); alFree(M2);
}

void* alMalloc(size_t size, int alignment) {
	const size_t pSize = sizeof(void*), a = alignment - 1;
	void *raw = wrMalloc(size + a + pSize);
	void *aligned = (void*)(((size_t)raw + pSize + a) & ~a);
	*(void**)((size_t)aligned - pSize) = raw;
	return aligned;
}

// platform independent alignned memory de-allocation (see also alMalloc)
void alFree(void* aligned) {
	void* raw = *(void**)((char*)aligned - sizeof(void*));
	wrFree(raw);
}

Mat Calc_Grad_Img_SSE(Mat frame, float* Grad_Mag, float* Grad_Or){
	int rows = frame.rows;
	int cols = frame.cols;

	Mat Mag_Img(rows, cols, CV_32FC1, double(0));
	Mat Or_Img(rows, cols, CV_32FC1, double(0));
	float *M = (float*)(Mag_Img.data);
	float *O = (float*)(Or_Img.data);
	unsigned char* frame_data_ptr = (unsigned char*)frame.data;

	bool full = 0;

	Mat frame_f(rows, cols, CV_32FC3);
	float *frame_f_data_ptr = (float*)(frame_f.data);

	//Mat frame_f(rows, cols, CV_8UC3);
	//unsigned char *frame_f_data_ptr = (unsigned char*)(frame_f.data);
	//Interleaved2Planar(frame,frame_f);
	//return frame_f;

#ifdef Use_P4	
	parallel_for(size_t(0), size_t(frame.rows), size_t(1), [&](size_t y) {
#else
	for (size_t y = 0; y < frame.rows; y++) {
#endif
		for (size_t x = 0; x < frame.cols; x++) {
			int i = y * frame.cols + x; // opencv is col first
			frame_f_data_ptr[i] = frame_data_ptr[3 * i + 0];
			frame_f_data_ptr[frame.cols*frame.rows + i] = frame_data_ptr[3 * i + 1];
			frame_f_data_ptr[2 * frame.cols*frame.rows + i] = frame_data_ptr[3 * i + 2];
		}
#ifdef Use_P4
	});
#else
	}
#endif
	
	//ftime(&t_start);
	gradMag(frame_f_data_ptr, M, O, cols, rows, 3, full);
	//ftime(&t_end);

	/*Mat frame_f(rows, cols, CV_32FC1);
	float *frame_f_data_ptr = (float*)(frame_f.data); 
	Mat temp;
	cvtColor(frame, temp, CV_BGR2GRAY);
	temp.convertTo(frame_f, CV_32FC1); // or CV_32F works (too)
	gradMag(frame_f_data_ptr, M, O, cols, rows, 1, full);*/

	Mat out;
	Mag_Img.convertTo(Mag_Img, CV_8UC1);
	//return Mag_Img;
	cvtColor(Mag_Img, out, CV_GRAY2BGR);
	return out;
}

void Interleaved2Planar(Mat& src, Mat& dst) {
	unsigned char* source_pixels = (unsigned char*)src.data;
	unsigned char* dest_pixels = (unsigned char*)dst.data;
	int length = src.rows*src.cols;

	int x = 0;
	for (; x < length*3; x = x + 96) {		//Code by Marat Dukhan https://docs.google.com/presentation/d/1I0-SiHid1hTsv7tjLST2dYW5YF5AJVfs9l4Rg9rvz48/edit#slide=id.p
		__m128i layer0_chunk0 = _mm_loadu_si128((__m128i*)source_pixels);
		__m128i layer0_chunk1 = _mm_loadu_si128((__m128i*)(source_pixels + 16));
		__m128i layer0_chunk2 = _mm_loadu_si128((__m128i*)(source_pixels + 32));
		__m128i layer0_chunk3 = _mm_loadu_si128((__m128i*)(source_pixels + 48));
		__m128i layer0_chunk4 = _mm_loadu_si128((__m128i*)(source_pixels + 64));
		__m128i layer0_chunk5 = _mm_loadu_si128((__m128i*)(source_pixels + 80));

		__m128i layer1_chunk0 = _mm_unpacklo_epi8(layer0_chunk0, layer0_chunk3);
		__m128i layer1_chunk1 = _mm_unpackhi_epi8(layer0_chunk0, layer0_chunk3);
		__m128i layer1_chunk2 = _mm_unpacklo_epi8(layer0_chunk1, layer0_chunk4);
		__m128i layer1_chunk3 = _mm_unpackhi_epi8(layer0_chunk1, layer0_chunk4);
		__m128i layer1_chunk4 = _mm_unpacklo_epi8(layer0_chunk2, layer0_chunk5);
		__m128i layer1_chunk5 = _mm_unpackhi_epi8(layer0_chunk2, layer0_chunk5);

		__m128i layer2_chunk0 = _mm_unpacklo_epi8(layer1_chunk0, layer1_chunk3);
		__m128i layer2_chunk1 = _mm_unpackhi_epi8(layer1_chunk0, layer1_chunk3);
		__m128i layer2_chunk2 = _mm_unpacklo_epi8(layer1_chunk1, layer1_chunk4);
		__m128i layer2_chunk3 = _mm_unpackhi_epi8(layer1_chunk1, layer1_chunk4);
		__m128i layer2_chunk4 = _mm_unpacklo_epi8(layer1_chunk2, layer1_chunk5);
		__m128i layer2_chunk5 = _mm_unpackhi_epi8(layer1_chunk2, layer1_chunk5);

		__m128i layer3_chunk0 = _mm_unpacklo_epi8(layer2_chunk0, layer2_chunk3);
		__m128i layer3_chunk1 = _mm_unpackhi_epi8(layer2_chunk0, layer2_chunk3);
		__m128i layer3_chunk2 = _mm_unpacklo_epi8(layer2_chunk1, layer2_chunk4);
		__m128i layer3_chunk3 = _mm_unpackhi_epi8(layer2_chunk1, layer2_chunk4);
		__m128i layer3_chunk4 = _mm_unpacklo_epi8(layer2_chunk2, layer2_chunk5);
		__m128i layer3_chunk5 = _mm_unpackhi_epi8(layer2_chunk2, layer2_chunk5);

		__m128i layer4_chunk0 = _mm_unpacklo_epi8(layer3_chunk0, layer3_chunk3);
		__m128i layer4_chunk1 = _mm_unpackhi_epi8(layer3_chunk0, layer3_chunk3);
		__m128i layer4_chunk2 = _mm_unpacklo_epi8(layer3_chunk1, layer3_chunk4);
		__m128i layer4_chunk3 = _mm_unpackhi_epi8(layer3_chunk1, layer3_chunk4);
		__m128i layer4_chunk4 = _mm_unpacklo_epi8(layer3_chunk2, layer3_chunk5);
		__m128i layer4_chunk5 = _mm_unpackhi_epi8(layer3_chunk2, layer3_chunk5);

		__m128i red_chunk0 = _mm_unpacklo_epi8(layer4_chunk0, layer4_chunk3);
		__m128i red_chunk1 = _mm_unpackhi_epi8(layer4_chunk0, layer4_chunk3);
		__m128i green_chunk0 = _mm_unpacklo_epi8(layer4_chunk1, layer4_chunk4);
		__m128i green_chunk1 = _mm_unpackhi_epi8(layer4_chunk1, layer4_chunk4);
		__m128i blue_chunk0 = _mm_unpacklo_epi8(layer4_chunk2, layer4_chunk5);
		__m128i blue_chunk1 = _mm_unpackhi_epi8(layer4_chunk2, layer4_chunk5);

		_mm_store_si128((__m128i*)dest_pixels, red_chunk0);
		_mm_store_si128((__m128i*)(dest_pixels + 16), red_chunk1);
		_mm_store_si128((__m128i*)(dest_pixels + length), green_chunk0);
		_mm_store_si128((__m128i*)(dest_pixels + length + 16), green_chunk1);
		_mm_store_si128((__m128i*)(dest_pixels + length*2), blue_chunk0);
		_mm_store_si128((__m128i*)(dest_pixels + length*2 + 16), blue_chunk1);

		dest_pixels += 32;		//32 pixels rgb data sorted per iteration
		source_pixels += 96;	//32 pixels rgb data sorted per iteration
	}
}
#endif

#endif