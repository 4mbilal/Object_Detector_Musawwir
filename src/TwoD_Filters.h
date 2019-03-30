#ifndef _TwoD_Filters_H_
#define _TwoD_Filters_H_
#include "Defines.h"



#ifdef Code_Junk
Mat Filters_Speed_Test(Mat src, int option);
Mat Calc_Grad_Img(Mat frame, float* Grad_Mag, unsigned char* Grad_Or);
#ifdef Use_SSE
Mat Calc_Grad_Img_SSE(Mat frame, float* Grad_Mag, float* Grad_Or);
void Interleaved2Planar(Mat& src, Mat& dst);
void RGB2LUV_ACF(unsigned char* I, unsigned char* J, int n, int d, int flag);

//----------------------    SSE Defines  --------------------------------------------
// Taken from Piotr's Toolbox. _SSE was added because MIN function is already defined
// in OpenCV.

/*******************************************************************************
* Piotr's Computer Vision Matlab Toolbox      Version 3.23
* Copyright 2014 Piotr Dollar.  [pdollar-at-gmail.com]
* Licensed under the Simplified BSD License [see external/bsd.txt]
*******************************************************************************/
#define PI 3.1415926535897932384f

#define RETf inline __m128
#define RETi inline __m128i

// set, load and store values
RETf SET_SSE(const float &x) { return _mm_set1_ps(x); }
RETf SET_SSE(float x, float y, float z, float w) { return _mm_set_ps(x, y, z, w); }
RETi SET_SSE(const int &x) { return _mm_set1_epi32(x); }
RETf LD_SSE(const float &x) { return _mm_load_ps(&x); }
RETf LDu_SSE(const float &x) { return _mm_loadu_ps(&x); }
RETf STR_SSE(float &x, const __m128 y) { _mm_store_ps(&x, y); return y; }
RETf STR1_SSE(float &x, const __m128 y) { _mm_store_ss(&x, y); return y; }
RETf STRu_SSE(float &x, const __m128 y) { _mm_storeu_ps(&x, y); return y; }
RETf STR_SSE(float &x, const float y) { return STR_SSE(x, SET_SSE(y)); }

// arithmetic operators
RETi ADD_SSE(const __m128i x, const __m128i y) { return _mm_add_epi32(x, y); }
RETf ADD_SSE(const __m128 x, const __m128 y) { return _mm_add_ps(x, y); }
RETf ADD_SSE(const __m128 x, const __m128 y, const __m128 z) {
	return ADD_SSE(ADD_SSE(x, y), z);
}
RETf ADD_SSE(const __m128 a, const __m128 b, const __m128 c, const __m128 &d) {
	return ADD_SSE(ADD_SSE(ADD_SSE(a, b), c), d);
}
RETf SUB_SSE(const __m128 x, const __m128 y) { return _mm_sub_ps(x, y); }
RETf MUL_SSE(const __m128 x, const __m128 y) { return _mm_mul_ps(x, y); }
RETf MUL_SSE(const __m128 x, const float y) { return MUL_SSE(x, SET_SSE(y)); }
RETf MUL_SSE(const float x, const __m128 y) { return MUL_SSE(SET_SSE(x), y); }
RETf INC_SSE(__m128 &x, const __m128 y) { return x = ADD_SSE(x, y); }
RETf INC_SSE(float &x, const __m128 y) { __m128 t = ADD_SSE(LD_SSE(x), y); return STR_SSE(x, t); }
RETf DEC_SSE(__m128 &x, const __m128 y) { return x = SUB_SSE(x, y); }
RETf DEC_SSE(float &x, const __m128 y) { __m128 t = SUB_SSE(LD_SSE(x), y); return STR_SSE(x, t); }
RETf MIN_SSE(const __m128 x, const __m128 y) { return _mm_min_ps(x, y); }
RETf RCP_SSE(const __m128 x) { return _mm_rcp_ps(x); }
RETf RCPSQRT_SSE(const __m128 x) { return _mm_rsqrt_ps(x); }

// logical operators
RETf AND_SSE(const __m128 x, const __m128 y) { return _mm_and_ps(x, y); }
RETi AND_SSE(const __m128i x, const __m128i y) { return _mm_and_si128(x, y); }
RETf ANDNOT_SSE(const __m128 x, const __m128 y) { return _mm_andnot_ps(x, y); }
RETf OR_SSE(const __m128 x, const __m128 y) { return _mm_or_ps(x, y); }
RETf XOR_SSE(const __m128 x, const __m128 y) { return _mm_xor_ps(x, y); }

// comparison operators
RETf CMPGT_SSE(const __m128 x, const __m128 y) { return _mm_cmpgt_ps(x, y); }
RETf CMPLT_SSE(const __m128 x, const __m128 y) { return _mm_cmplt_ps(x, y); }
RETi CMPGT_SSE(const __m128i x, const __m128i y) { return _mm_cmpgt_epi32(x, y); }
RETi CMPLT_SSE(const __m128i x, const __m128i y) { return _mm_cmplt_epi32(x, y); }

// conversion operators
RETf CVT_SSE(const __m128i x) { return _mm_cvtepi32_ps(x); }
RETi CVT_SSE(const __m128 x) { return _mm_cvttps_epi32(x); }



// wrapper functions if compiling from C/C++
inline void wrError(const char *errormsg) { throw errormsg; }
inline void* wrCalloc(size_t num, size_t size) { return calloc(num, size); }
inline void* wrMalloc(size_t size) { return malloc(size); }
inline void wrFree(void * ptr) { free(ptr); }
void* alMalloc(size_t size, int alignment);
void alFree(void* aligned);


#undef RETf
#undef RETi
#endif
#endif
#endif