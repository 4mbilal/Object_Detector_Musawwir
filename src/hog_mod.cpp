#include"hog_mod.h"
//#include "Linear_LUT_HOG_Optimal_48_96.h"
#include <cstdio>
#include <iterator>
#include <limits>

#define NTHREADS 256

enum {DESCR_FORMAT_COL_BY_COL, DESCR_FORMAT_ROW_BY_ROW};

static int numPartsWithin(int size, int part_size, int stride)
{
    return (size - part_size + stride) / stride;
}

static Size numPartsWithin(cv::Size size, cv::Size part_size,
                                                cv::Size stride)
{
    return Size(numPartsWithin(size.width, part_size.width, stride.width),
        numPartsWithin(size.height, part_size.height, stride.height));
}

static size_t getBlockHistogramSize(Size block_size, Size cell_size, int nbins)
{
    Size cells_per_block = Size(block_size.width / cell_size.width,
        block_size.height / cell_size.height);
    return (size_t)(nbins * cells_per_block.area());
}

size_t HOGDescriptor_Mod::getDescriptorSize() const
{
    CV_Assert(blockSize.width % cellSize.width == 0 &&
        blockSize.height % cellSize.height == 0);
    CV_Assert((winSize.width - blockSize.width) % blockStride.width == 0 &&
        (winSize.height - blockSize.height) % blockStride.height == 0 );

    return (size_t)nbins*
        (blockSize.width/cellSize.width)*
        (blockSize.height/cellSize.height)*
        ((winSize.width - blockSize.width)/blockStride.width + 1)*
        ((winSize.height - blockSize.height)/blockStride.height + 1);
}

double HOGDescriptor_Mod::getWinSigma() const
{
    return winSigma >= 0 ? winSigma : (blockSize.width + blockSize.height)/8.;
}

bool HOGDescriptor_Mod::checkDetectorSize() const
{
    size_t detectorSize = svmDetector.size(), descriptorSize = getDescriptorSize();
    return detectorSize == 0 ||
        detectorSize == descriptorSize ||
        detectorSize == descriptorSize + 1;
}

void HOGDescriptor_Mod::setSVMDetector(InputArray _svmDetector)
{
    _svmDetector.getMat().convertTo(svmDetector, CV_32F);
    CV_Assert(checkDetectorSize());

    Mat detector_reordered(1, (int)svmDetector.size(), CV_32FC1);

    size_t block_hist_size = getBlockHistogramSize(blockSize, cellSize, nbins);
    cv::Size blocks_per_img = numPartsWithin(winSize, blockSize, blockStride);

    for (int i = 0; i < blocks_per_img.height; ++i)
        for (int j = 0; j < blocks_per_img.width; ++j)
        {
            const float *src = &svmDetector[0] + (j * blocks_per_img.height + i) * block_hist_size;
            float *dst = detector_reordered.ptr<float>() + (i * blocks_per_img.width + j) * block_hist_size;
            for (size_t k = 0; k < block_hist_size; ++k)
                dst[k] = src[k];
        }
    size_t descriptor_size = getDescriptorSize();
    free_coef = svmDetector.size() > descriptor_size ? svmDetector[descriptor_size] : 0;
    detector_reordered.copyTo(oclSvmDetector);
}

#define CV_TYPE_NAME_HOG_DESCRIPTOR "opencv-object-detector-hog"

bool HOGDescriptor_Mod::read(FileNode& obj)
{
    if( !obj.isMap() )
        return false;
    FileNodeIterator it = obj["winSize"].begin();
    it >> winSize.width >> winSize.height;
    it = obj["blockSize"].begin();
    it >> blockSize.width >> blockSize.height;
    it = obj["blockStride"].begin();
    it >> blockStride.width >> blockStride.height;
    it = obj["cellSize"].begin();
    it >> cellSize.width >> cellSize.height;
    obj["nbins"] >> nbins;
    obj["derivAperture"] >> derivAperture;
    obj["winSigma"] >> winSigma;
    obj["histogramNormType"] >> histogramNormType;
    obj["L2HysThreshold"] >> L2HysThreshold;
    obj["gammaCorrection"] >> gammaCorrection;
    obj["nlevels"] >> nlevels;
    if (obj["signedGradient"].empty())
        signedGradient = false;
    else
        obj["signedGradient"] >> signedGradient;

    FileNode vecNode = obj["SVMDetector"];
    if( vecNode.isSeq() )
    {
        vecNode >> svmDetector;
        CV_Assert(checkDetectorSize());
    }
    return true;
}

void HOGDescriptor_Mod::write(FileStorage& fs, const String& objName) const
{
    if( !objName.empty() )
        fs << objName;

    fs << "{" CV_TYPE_NAME_HOG_DESCRIPTOR
       << "winSize" << winSize
       << "blockSize" << blockSize
       << "blockStride" << blockStride
       << "cellSize" << cellSize
       << "nbins" << nbins
       << "derivAperture" << derivAperture
       << "winSigma" << getWinSigma()
       << "histogramNormType" << histogramNormType
       << "L2HysThreshold" << L2HysThreshold
       << "gammaCorrection" << gammaCorrection
       << "nlevels" << nlevels
       << "signedGradient" << signedGradient;
    if( !svmDetector.empty() )
        fs << "SVMDetector" << svmDetector;
    fs << "}";
}

bool HOGDescriptor_Mod::load(const String& filename, const String& objname)
{
    FileStorage fs(filename, FileStorage::READ);
    FileNode obj = !objname.empty() ? fs[objname] : fs.getFirstTopLevelNode();
    return read(obj);
}

void HOGDescriptor_Mod::save(const String& filename, const String& objName) const
{
    FileStorage fs(filename, FileStorage::WRITE);
    write(fs, !objName.empty() ? objName : FileStorage::getDefaultObjectName(filename));
}

void HOGDescriptor_Mod::copyTo(HOGDescriptor_Mod & c) const
{
    c.winSize = winSize;
    c.blockSize = blockSize;
    c.blockStride = blockStride;
    c.cellSize = cellSize;
    c.nbins = nbins;
    c.derivAperture = derivAperture;
    c.winSigma = winSigma;
    c.histogramNormType = histogramNormType;
    c.L2HysThreshold = L2HysThreshold;
    c.gammaCorrection = gammaCorrection;
    c.svmDetector = svmDetector;
    c.nlevels = nlevels;
    c.signedGradient = signedGradient;
}

void HOGDescriptor_Mod::computeGradient(const Mat& img, Mat& grad, Mat& qangle,
    Size paddingTL, Size paddingBR) const
{
    CV_Assert( img.type() == CV_8U || img.type() == CV_8UC3 );

    Size gradsize(img.cols + paddingTL.width + paddingBR.width,
        img.rows + paddingTL.height + paddingBR.height);
    grad.create(gradsize, CV_32FC2);  // <magnitude*(1-alpha), magnitude*alpha>
    qangle.create(gradsize, CV_8UC2); // [0..nbins-1] - quantized gradient orientation

    Size wholeSize;
    Point roiofs;
    img.locateROI(wholeSize, roiofs);

    int i, x, y;
    int cn = img.channels();

    Mat_<float> _lut(1, 256);
    const float* const lut = &_lut(0,0);
#ifdef UseSSE
    const int indeces[] = { 0, 1, 2, 3 };
    __m128i idx = _mm_loadu_si128((const __m128i*)indeces);
    __m128i ifour = _mm_set1_epi32(4);

    float* const _data = &_lut(0, 0);
    if( gammaCorrection )
        for( i = 0; i < 256; i += 4 )
        {
            _mm_storeu_ps(_data + i, _mm_sqrt_ps(_mm_cvtepi32_ps(idx)));
            idx = _mm_add_epi32(idx, ifour);
        }
    else
        for( i = 0; i < 256; i += 4 )
        {
            _mm_storeu_ps(_data + i, _mm_cvtepi32_ps(idx));
            idx = _mm_add_epi32(idx, ifour);
        }
#else
    if( gammaCorrection )
        for( i = 0; i < 256; i++ )
            _lut(0,i) = std::sqrt((float)i);
    else
        for( i = 0; i < 256; i++ )
            _lut(0,i) = (float)i;
#endif

    AutoBuffer<int> mapbuf(gradsize.width + gradsize.height + 4);
    int* xmap = (int*)mapbuf + 1;
    int* ymap = xmap + gradsize.width + 2;

    const int borderType = (int)BORDER_REFLECT_101;

    for( x = -1; x < gradsize.width + 1; x++ )
        xmap[x] = borderInterpolate(x - paddingTL.width + roiofs.x,
        wholeSize.width, borderType) - roiofs.x;
    for( y = -1; y < gradsize.height + 1; y++ )
        ymap[y] = borderInterpolate(y - paddingTL.height + roiofs.y,
        wholeSize.height, borderType) - roiofs.y;

    // x- & y- derivatives for the whole row
    int width = gradsize.width;
    AutoBuffer<float> _dbuf(width*4);
    float* const dbuf = _dbuf;
    Mat Dx(1, width, CV_32F, dbuf);
    Mat Dy(1, width, CV_32F, dbuf + width);
    Mat Mag(1, width, CV_32F, dbuf + width*2);
    Mat Angle(1, width, CV_32F, dbuf + width*3);

    if (cn == 3)
    {
        int end = gradsize.width + 2;
        xmap -= 1, x = 0;
#ifdef UseSSE
        __m128i ithree = _mm_set1_epi32(3);
        for ( ; x <= end - 4; x += 4)
            _mm_storeu_si128((__m128i*)(xmap + x), _mm_mullo_epi16(ithree,
                _mm_loadu_si128((const __m128i*)(xmap + x))));
#endif
        for ( ; x < end; ++x)
            xmap[x] *= 3;
        xmap += 1;
    }

    float angleScale = signedGradient ? (float)(nbins/(2.0*CV_PI)) : (float)(nbins/CV_PI);
    for( y = 0; y < gradsize.height; y++ )
    {
        const uchar* imgPtr  = img.ptr(ymap[y]);
        //In case subimage is used ptr() generates an assert for next and prev rows
        //(see http://code.opencv.org/issues/4149)
        const uchar* prevPtr = img.data + img.step*ymap[y-1];
        const uchar* nextPtr = img.data + img.step*ymap[y+1];

        float* gradPtr = grad.ptr<float>(y);
        uchar* qanglePtr = qangle.ptr(y);

        if( cn == 1 )
        {
            for( x = 0; x < width; x++ )
            {
                int x1 = xmap[x];
                dbuf[x] = (float)(lut[imgPtr[xmap[x+1]]] - lut[imgPtr[xmap[x-1]]]);
                dbuf[width + x] = (float)(lut[nextPtr[x1]] - lut[prevPtr[x1]]);
            }
        }
        else
        {
            x = 0;
#ifdef UseSSE
            for( ; x <= width - 4; x += 4 )
            {
                int x0 = xmap[x], x1 = xmap[x+1], x2 = xmap[x+2], x3 = xmap[x+3];
                typedef const uchar* const T;
                T p02 = imgPtr + xmap[x+1], p00 = imgPtr + xmap[x-1];
                T p12 = imgPtr + xmap[x+2], p10 = imgPtr + xmap[x];
                T p22 = imgPtr + xmap[x+3], p20 = p02;
                T p32 = imgPtr + xmap[x+4], p30 = p12;

                __m128 _dx0 = _mm_sub_ps(_mm_set_ps(lut[p32[0]], lut[p22[0]], lut[p12[0]], lut[p02[0]]),
                                         _mm_set_ps(lut[p30[0]], lut[p20[0]], lut[p10[0]], lut[p00[0]]));
                __m128 _dx1 = _mm_sub_ps(_mm_set_ps(lut[p32[1]], lut[p22[1]], lut[p12[1]], lut[p02[1]]),
                                         _mm_set_ps(lut[p30[1]], lut[p20[1]], lut[p10[1]], lut[p00[1]]));
                __m128 _dx2 = _mm_sub_ps(_mm_set_ps(lut[p32[2]], lut[p22[2]], lut[p12[2]], lut[p02[2]]),
                                         _mm_set_ps(lut[p30[2]], lut[p20[2]], lut[p10[2]], lut[p00[2]]));

                __m128 _dy0 = _mm_sub_ps(_mm_set_ps(lut[nextPtr[x3]], lut[nextPtr[x2]], lut[nextPtr[x1]], lut[nextPtr[x0]]),
                                         _mm_set_ps(lut[prevPtr[x3]], lut[prevPtr[x2]], lut[prevPtr[x1]], lut[prevPtr[x0]]));
                __m128 _dy1 = _mm_sub_ps(_mm_set_ps(lut[nextPtr[x3+1]], lut[nextPtr[x2+1]], lut[nextPtr[x1+1]], lut[nextPtr[x0+1]]),
                                         _mm_set_ps(lut[prevPtr[x3+1]], lut[prevPtr[x2+1]], lut[prevPtr[x1+1]], lut[prevPtr[x0+1]]));
                __m128 _dy2 = _mm_sub_ps(_mm_set_ps(lut[nextPtr[x3+2]], lut[nextPtr[x2+2]], lut[nextPtr[x1+2]], lut[nextPtr[x0+2]]),
                                         _mm_set_ps(lut[prevPtr[x3+2]], lut[prevPtr[x2+2]], lut[prevPtr[x1+2]], lut[prevPtr[x0+2]]));

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
            for( ; x < width; x++ )
            {
                int x1 = xmap[x];
                float dx0, dy0, dx, dy, mag0, mag;
                const uchar* p2 = imgPtr + xmap[x+1];
                const uchar* p0 = imgPtr + xmap[x-1];

                dx0 = lut[p2[2]] - lut[p0[2]];
                dy0 = lut[nextPtr[x1+2]] - lut[prevPtr[x1+2]];
                mag0 = dx0*dx0 + dy0*dy0;

                dx = lut[p2[1]] - lut[p0[1]];
                dy = lut[nextPtr[x1+1]] - lut[prevPtr[x1+1]];
                mag = dx*dx + dy*dy;
                if( mag0 < mag )
                {
                    dx0 = dx;
                    dy0 = dy;
                    mag0 = mag;
                }

                dx = lut[p2[0]] - lut[p0[0]];
                dy = lut[nextPtr[x1]] - lut[prevPtr[x1]];
                mag = dx*dx + dy*dy;
                if( mag0 < mag )
                {
                    dx0 = dx;
                    dy0 = dy;
                    mag0 = mag;
                }

                dbuf[x] = dx0;
                dbuf[x+width] = dy0;
            }
        }

        // computing angles and magnidutes
        cartToPolar( Dx, Dy, Mag, Angle, false );

        // filling the result matrix
        x = 0;
#ifdef UseSSE
        __m128 fhalf = _mm_set1_ps(0.5f), fzero = _mm_setzero_ps();
        __m128 _angleScale = _mm_set1_ps(angleScale), fone = _mm_set1_ps(1.0f);
        __m128i ione = _mm_set1_epi32(1), _nbins = _mm_set1_epi32(nbins), izero = _mm_setzero_si128();

        for ( ; x <= width - 4; x += 4)
        {
            int x2 = x << 1;															//Multiply by two because two values per magnitude are being stored
            __m128 _mag = _mm_loadu_ps(dbuf + x + (width << 1));						//float mag = dbuf[x+width*2]
            __m128 _angle = _mm_loadu_ps(dbuf + x + width * 3);							//float angle = dbuf[x+width*3]
            _angle = _mm_sub_ps(_mm_mul_ps(_angleScale, _angle), fhalf);				//angle*=angleScale - 0.5f;

            __m128 sign = _mm_and_ps(fone, _mm_cmplt_ps(_angle, fzero));
            __m128i _hidx = _mm_cvttps_epi32(_angle);									//int hidx = cvFloor(angle);
            _hidx = _mm_sub_epi32(_hidx, _mm_cvtps_epi32(sign));
            _angle = _mm_sub_ps(_angle, _mm_cvtepi32_ps(_hidx));						// angle -= hidx;

            __m128 ft0 = _mm_mul_ps(_mag, _mm_sub_ps(fone, _angle));					//float ft0 = mag*(1.f - angle)
            __m128 ft1 = _mm_mul_ps(_mag, _angle);										//float ft1 = mag*angle
            __m128 ft2 = _mm_unpacklo_ps(ft0, ft1);										//interleave ft0 and ft1 
            __m128 ft3 = _mm_unpackhi_ps(ft0, ft1);										//interleave continued

            _mm_storeu_ps(gradPtr + x2, ft2);											//store the interleaved values
            _mm_storeu_ps(gradPtr + x2 + 4, ft3);

            __m128i mask0 = _mm_sub_epi32(izero, _mm_srli_epi32(_hidx, 31));
            __m128i it0 = _mm_and_si128(mask0, _nbins);
            mask0 = _mm_cmplt_epi32(_hidx, _nbins);
            __m128i it1 = _mm_andnot_si128(mask0, _nbins);
            _hidx = _mm_add_epi32(_hidx, _mm_sub_epi32(it0, it1));

            it0 = _mm_packus_epi16(_mm_packs_epi32(_hidx, izero), izero);				//4 bin values as 4 8-bit unsigned values, rest are all zeros
            _hidx = _mm_add_epi32(ione, _hidx);											//hidx++;
            _hidx = _mm_and_si128(_hidx, _mm_cmplt_epi32(_hidx, _nbins));				//hidx &= hidx < nbins ? -1 : 0;
            it1 = _mm_packus_epi16(_mm_packs_epi32(_hidx, izero), izero);				//4 bin values as 4 8-bit unsigned values, rest are all zeros
            it0 = _mm_unpacklo_epi8(it0, it1);											//interleave to make 8 8-bit unsigned values, rest are all zeros

            _mm_storel_epi64((__m128i*)(qanglePtr + x2), it0);
        }
#endif
        for( ; x < width; x++ )
        {
            float mag = dbuf[x+width*2], angle = dbuf[x+width*3]*angleScale - 0.5f;
            int hidx = cvFloor(angle);
            angle -= hidx;
            gradPtr[x*2] = mag*(1.f - angle);
            gradPtr[x*2+1] = mag*angle;

            if( hidx < 0 )
                hidx += nbins;
            else if( hidx >= nbins )
                hidx -= nbins;

            CV_Assert( (unsigned)hidx < (unsigned)nbins );

            qanglePtr[x*2] = (uchar)hidx;
            hidx++;
            hidx &= hidx < nbins ? -1 : 0;
            qanglePtr[x*2+1] = (uchar)hidx;
        }
    }
}

struct HOGCache
{
    struct BlockData
    {
        BlockData() :
            histOfs(0), imgOffset()
        { }

        int histOfs;
        Point imgOffset;
    };

    struct PixData
    {
        size_t gradOfs, qangleOfs;
        int histOfs[4];
        float histWeights[4];
        float gradWeight;
    };

    HOGCache();
    HOGCache(const HOGDescriptor_Mod* descriptor,
        const Mat& img, const Size& paddingTL, const Size& paddingBR,
        bool useCache, const Size& cacheStride);
    virtual ~HOGCache() { }
    virtual void init(const HOGDescriptor_Mod* descriptor,
        const Mat& img, const Size& paddingTL, const Size& paddingBR,
        bool useCache, const Size& cacheStride);

    Size windowsInImage(const Size& imageSize, const Size& winStride) const;
    Rect getWindow(const Size& imageSize, const Size& winStride, int idx) const;

    const float* getBlock(Point pt, float* buf);
    virtual void normalizeBlockHistogram(float* histogram) const;

    std::vector<PixData> pixData;
    std::vector<BlockData> blockData;

    bool useCache;
    std::vector<int> ymaxCached;
    Size winSize;
    Size cacheStride;
    Size nblocks, ncells;
    int blockHistogramSize;
    int count1, count2, count4;
    Point imgoffset;
    Mat_<float> blockCache;
    Mat_<uchar> blockCacheFlags;

    Mat grad, qangle;
    const HOGDescriptor_Mod* descriptor;
};

HOGCache::HOGCache() :
    blockHistogramSize(), count1(), count2(), count4()
{
    useCache = false;
    descriptor = 0;
}

HOGCache::HOGCache(const HOGDescriptor_Mod* _descriptor,
    const Mat& _img, const Size& _paddingTL, const Size& _paddingBR,
    bool _useCache, const Size& _cacheStride)
{
    init(_descriptor, _img, _paddingTL, _paddingBR, _useCache, _cacheStride);
}

void HOGCache::init(const HOGDescriptor_Mod* _descriptor,
    const Mat& _img, const Size& _paddingTL, const Size& _paddingBR,
    bool _useCache, const Size& _cacheStride)
{
    descriptor = _descriptor;
    cacheStride = _cacheStride;
    useCache = _useCache;
	//ftime(&t_start);
    descriptor->computeGradient(_img, grad, qangle, _paddingTL, _paddingBR);
	//ftime(&t_end);
    imgoffset = _paddingTL;

    winSize = descriptor->winSize;
    Size blockSize = descriptor->blockSize;
    Size blockStride = descriptor->blockStride;
    Size cellSize = descriptor->cellSize;
    int i, j, nbins = descriptor->nbins;
    int rawBlockSize = blockSize.width*blockSize.height;

    nblocks = Size((winSize.width - blockSize.width)/blockStride.width + 1,
        (winSize.height - blockSize.height)/blockStride.height + 1);
    ncells = Size(blockSize.width/cellSize.width, blockSize.height/cellSize.height);
    blockHistogramSize = ncells.width*ncells.height*nbins;

    if( useCache )
    {
        Size cacheSize((grad.cols - blockSize.width)/cacheStride.width+1,
            (winSize.height/cacheStride.height)+1);

        blockCache.create(cacheSize.height, cacheSize.width*blockHistogramSize);
        blockCacheFlags.create(cacheSize);

        size_t cacheRows = blockCache.rows;
        ymaxCached.resize(cacheRows);
        for(size_t ii = 0; ii < cacheRows; ii++ )
            ymaxCached[ii] = -1;
    }

    Mat_<float> weights(blockSize);
    float sigma = (float)descriptor->getWinSigma();
    float scale = 1.f/(sigma*sigma*2);

    {
        AutoBuffer<float> di(blockSize.height), dj(blockSize.width);
        float* _di = (float*)di, *_dj = (float*)dj;
        float bh = blockSize.height * 0.5f, bw = blockSize.width * 0.5f;

        i = 0;
    #ifdef UseSSE
        const int a[] = { 0, 1, 2, 3 };
        __m128i idx = _mm_loadu_si128((__m128i*)a);
        __m128 _bw = _mm_set1_ps(bw), _bh = _mm_set1_ps(bh);
        __m128i ifour = _mm_set1_epi32(4);

        for (; i <= blockSize.height - 4; i += 4)
        {
            __m128 t = _mm_sub_ps(_mm_cvtepi32_ps(idx), _bh);
            t = _mm_mul_ps(t, t);
            idx = _mm_add_epi32(idx, ifour);
            _mm_storeu_ps(_di + i, t);
        }
    #endif
        for ( ; i < blockSize.height; ++i)
        {
            _di[i] = i - bh;
            _di[i] *= _di[i];
        }

        j = 0;
    #ifdef UseSSE
        idx = _mm_loadu_si128((__m128i*)a);
        for (; j <= blockSize.width - 4; j += 4)
        {
            __m128 t = _mm_sub_ps(_mm_cvtepi32_ps(idx), _bw);
            t = _mm_mul_ps(t, t);
            idx = _mm_add_epi32(idx, ifour);
            _mm_storeu_ps(_dj + j, t);
        }
    #endif
        for ( ; j < blockSize.width; ++j)
        {
            _dj[j] = j - bw;
            _dj[j] *= _dj[j];
        }

        for(i = 0; i < blockSize.height; i++)
            for(j = 0; j < blockSize.width; j++)
                weights(i,j) = std::exp(-(_di[i] + _dj[j])*scale);
    }

    blockData.resize(nblocks.width*nblocks.height);
    pixData.resize(rawBlockSize*3);

    // Initialize 2 lookup tables, pixData & blockData.
    // Here is why:
    //
    // The detection algorithm runs in 4 nested loops (at each pyramid layer):
    //  loop over the windows within the input image
    //    loop over the blocks within each window
    //      loop over the cells within each block
    //        loop over the pixels in each cell
    //
    // As each of the loops runs over a 2-dimensional array,
    // we could get 8(!) nested loops in total, which is very-very slow.
    //
    // To speed the things up, we do the following:
    //   1. loop over windows is unrolled in the HOGDescriptor_Mod::{compute|detect} methods;
    //         inside we compute the current search window using getWindow() method.
    //         Yes, it involves some overhead (function call + couple of divisions),
    //         but it's tiny in fact.
    //   2. loop over the blocks is also unrolled. Inside we use pre-computed blockData[j]
    //         to set up gradient and histogram pointers.
    //   3. loops over cells and pixels in each cell are merged
    //       (since there is no overlap between cells, each pixel in the block is processed once)
    //      and also unrolled. Inside we use PixData[k] to access the gradient values and
    //      update the histogram
    //

    count1 = count2 = count4 = 0;
    for( j = 0; j < blockSize.width; j++ )
        for( i = 0; i < blockSize.height; i++ )
        {
            PixData* data = 0;
            float cellX = (j+0.5f)/cellSize.width - 0.5f;
            float cellY = (i+0.5f)/cellSize.height - 0.5f;
            int icellX0 = cvFloor(cellX);
            int icellY0 = cvFloor(cellY);
            int icellX1 = icellX0 + 1, icellY1 = icellY0 + 1;
            cellX -= icellX0;
            cellY -= icellY0;

            if( (unsigned)icellX0 < (unsigned)ncells.width &&
               (unsigned)icellX1 < (unsigned)ncells.width )
            {
                if( (unsigned)icellY0 < (unsigned)ncells.height &&
                   (unsigned)icellY1 < (unsigned)ncells.height )
                {
                    data = &pixData[rawBlockSize*2 + (count4++)];
                    data->histOfs[0] = (icellX0*ncells.height + icellY0)*nbins;
                    data->histWeights[0] = (1.f - cellX)*(1.f - cellY);
                    data->histOfs[1] = (icellX1*ncells.height + icellY0)*nbins;
                    data->histWeights[1] = cellX*(1.f - cellY);
                    data->histOfs[2] = (icellX0*ncells.height + icellY1)*nbins;
                    data->histWeights[2] = (1.f - cellX)*cellY;
                    data->histOfs[3] = (icellX1*ncells.height + icellY1)*nbins;
                    data->histWeights[3] = cellX*cellY;
                }
                else
                {
                    data = &pixData[rawBlockSize + (count2++)];
                    if( (unsigned)icellY0 < (unsigned)ncells.height )
                    {
                        icellY1 = icellY0;
                        cellY = 1.f - cellY;
                    }
                    data->histOfs[0] = (icellX0*ncells.height + icellY1)*nbins;
                    data->histWeights[0] = (1.f - cellX)*cellY;
                    data->histOfs[1] = (icellX1*ncells.height + icellY1)*nbins;
                    data->histWeights[1] = cellX*cellY;
                    data->histOfs[2] = data->histOfs[3] = 0;
                    data->histWeights[2] = data->histWeights[3] = 0;
                }
            }
            else
            {
                if( (unsigned)icellX0 < (unsigned)ncells.width )
                {
                    icellX1 = icellX0;
                    cellX = 1.f - cellX;
                }

                if( (unsigned)icellY0 < (unsigned)ncells.height &&
                   (unsigned)icellY1 < (unsigned)ncells.height )
                {
                    data = &pixData[rawBlockSize + (count2++)];
                    data->histOfs[0] = (icellX1*ncells.height + icellY0)*nbins;
                    data->histWeights[0] = cellX*(1.f - cellY);
                    data->histOfs[1] = (icellX1*ncells.height + icellY1)*nbins;
                    data->histWeights[1] = cellX*cellY;
                    data->histOfs[2] = data->histOfs[3] = 0;
                    data->histWeights[2] = data->histWeights[3] = 0;
                }
                else
                {
                    data = &pixData[count1++];
                    if( (unsigned)icellY0 < (unsigned)ncells.height )
                    {
                        icellY1 = icellY0;
                        cellY = 1.f - cellY;
                    }
                    data->histOfs[0] = (icellX1*ncells.height + icellY1)*nbins;
                    data->histWeights[0] = cellX*cellY;
                    data->histOfs[1] = data->histOfs[2] = data->histOfs[3] = 0;
                    data->histWeights[1] = data->histWeights[2] = data->histWeights[3] = 0;
                }
            }
            data->gradOfs = (grad.cols*i + j)*2;
            data->qangleOfs = (qangle.cols*i + j)*2;
            data->gradWeight = weights(i,j);
        }

    assert( count1 + count2 + count4 == rawBlockSize );
    // defragment pixData
    for( j = 0; j < count2; j++ )
        pixData[j + count1] = pixData[j + rawBlockSize];
    for( j = 0; j < count4; j++ )
        pixData[j + count1 + count2] = pixData[j + rawBlockSize*2];
    count2 += count1;
    count4 += count2;

    // initialize blockData
    for( j = 0; j < nblocks.width; j++ )
        for( i = 0; i < nblocks.height; i++ )
        {
            BlockData& data = blockData[j*nblocks.height + i];
            data.histOfs = (j*nblocks.height + i)*blockHistogramSize;
            data.imgOffset = Point(j*blockStride.width,i*blockStride.height);
        }
}

const float* HOGCache::getBlock(Point pt, float* buf)
{
    float* blockHist = buf;
    assert(descriptor != 0);

//    Size blockSize = descriptor->blockSize;
    pt += imgoffset;

//    CV_Assert( (unsigned)pt.x <= (unsigned)(grad.cols - blockSize.width) &&
//        (unsigned)pt.y <= (unsigned)(grad.rows - blockSize.height) );

    if( useCache )
    {
        CV_Assert( pt.x % cacheStride.width == 0 &&
                   pt.y % cacheStride.height == 0 );
        Point cacheIdx(pt.x/cacheStride.width,
                       (pt.y/cacheStride.height) % blockCache.rows);
        if( pt.y != ymaxCached[cacheIdx.y] )
        {
            Mat_<uchar> cacheRow = blockCacheFlags.row(cacheIdx.y);
            cacheRow = (uchar)0;
            ymaxCached[cacheIdx.y] = pt.y;
        }

        blockHist = &blockCache[cacheIdx.y][cacheIdx.x*blockHistogramSize];
        uchar& computedFlag = blockCacheFlags(cacheIdx.y, cacheIdx.x);
        if( computedFlag != 0 )
            return blockHist;
        computedFlag = (uchar)1; // set it at once, before actual computing
    }

    int k, C1 = count1, C2 = count2, C4 = count4;
    const float* gradPtr = grad.ptr<float>(pt.y) + pt.x*2;
    const uchar* qanglePtr = qangle.ptr(pt.y) + pt.x*2;

//    CV_Assert( blockHist != 0 );
    memset(blockHist, 0, sizeof(float) * blockHistogramSize);

    const PixData* _pixData = &pixData[0];

    for( k = 0; k < C1; k++ )
    {
        const PixData& pk = _pixData[k];
        const float* const a = gradPtr + pk.gradOfs;
        float w = pk.gradWeight*pk.histWeights[0];
        const uchar* h = qanglePtr + pk.qangleOfs;
        int h0 = h[0], h1 = h[1];

        float* hist = blockHist + pk.histOfs[0];
        float t0 = hist[h0] + a[0]*w;
        float t1 = hist[h1] + a[1]*w;
        hist[h0] = t0; hist[h1] = t1;
    }

#ifdef UseSSE
    float hist0[4], hist1[4];
    for( ; k < C2; k++ )
    {
        const PixData& pk = _pixData[k];
        const float* const a = gradPtr + pk.gradOfs;
        const uchar* const h = qanglePtr + pk.qangleOfs;
        int h0 = h[0], h1 = h[1];

        __m128 _a0 = _mm_set1_ps(a[0]), _a1 = _mm_set1_ps(a[1]);
        __m128 _w = _mm_mul_ps(_mm_set1_ps(pk.gradWeight), _mm_loadu_ps(pk.histWeights));
        __m128 _t0 = _mm_mul_ps(_a0, _w), _t1 = _mm_mul_ps(_a1, _w);

        _mm_storeu_ps(hist0, _t0);
        _mm_storeu_ps(hist1, _t1);

        float* hist = blockHist + pk.histOfs[0];
        float t0 = hist[h0] + hist0[0];
        float t1 = hist[h1] + hist1[0];
        hist[h0] = t0; hist[h1] = t1;

        hist = blockHist + pk.histOfs[1];
        t0 = hist[h0] + hist0[1];
        t1 = hist[h1] + hist1[1];
        hist[h0] = t0; hist[h1] = t1;
    }
#else
    for( ; k < C2; k++ )
    {
        const PixData& pk = _pixData[k];
        const float* const a = gradPtr + pk.gradOfs;
        float w, t0, t1, a0 = a[0], a1 = a[1];
        const uchar* const h = qanglePtr + pk.qangleOfs;
        int h0 = h[0], h1 = h[1];

        float* hist = blockHist + pk.histOfs[0];
        w = pk.gradWeight*pk.histWeights[0];
        t0 = hist[h0] + a0*w;
        t1 = hist[h1] + a1*w;
        hist[h0] = t0; hist[h1] = t1;

        hist = blockHist + pk.histOfs[1];
        w = pk.gradWeight*pk.histWeights[1];
        t0 = hist[h0] + a0*w;
        t1 = hist[h1] + a1*w;
        hist[h0] = t0; hist[h1] = t1;
    }
#endif

#ifdef UseSSE
    for( ; k < C4; k++ )
    {
        const PixData& pk = _pixData[k];
        const float* const a = gradPtr + pk.gradOfs;
        const uchar* const h = qanglePtr + pk.qangleOfs;
        int h0 = h[0], h1 = h[1];

        __m128 _a0 = _mm_set1_ps(a[0]), _a1 = _mm_set1_ps(a[1]);
        __m128 _w = _mm_mul_ps(_mm_set1_ps(pk.gradWeight), _mm_loadu_ps(pk.histWeights));
        __m128 _t0 = _mm_mul_ps(_a0, _w), _t1 = _mm_mul_ps(_a1, _w);

        _mm_storeu_ps(hist0, _t0);
        _mm_storeu_ps(hist1, _t1);

        float* hist = blockHist + pk.histOfs[0];
        float t0 = hist[h0] + hist0[0];
        float t1 = hist[h1] + hist1[0];
        hist[h0] = t0; hist[h1] = t1;

        hist = blockHist + pk.histOfs[1];
        t0 = hist[h0] + hist0[1];
        t1 = hist[h1] + hist1[1];
        hist[h0] = t0; hist[h1] = t1;

        hist = blockHist + pk.histOfs[2];
        t0 = hist[h0] + hist0[2];
        t1 = hist[h1] + hist1[2];
        hist[h0] = t0; hist[h1] = t1;

        hist = blockHist + pk.histOfs[3];
        t0 = hist[h0] + hist0[3];
        t1 = hist[h1] + hist1[3];
        hist[h0] = t0; hist[h1] = t1;

//        __m128 _hist0 = _mm_set_ps((blockHist + pk.histOfs[3])[h0], (blockHist + pk.histOfs[2])[h0],
//            (blockHist + pk.histOfs[1])[h0], (blockHist + pk.histOfs[0])[h0]);
//        __m128 _hist1 = _mm_set_ps((blockHist + pk.histOfs[3])[h1], (blockHist + pk.histOfs[2])[h1],
//            (blockHist + pk.histOfs[1])[h1], (blockHist + pk.histOfs[0])[h1]);
//
//        _hist0 = _mm_add_ps(_t0, _hist0);
//        _hist1 = _mm_add_ps(_t1, _hist1);
//
//        _mm_storeu_ps(hist0, _hist0);
//        _mm_storeu_ps(hist1, _hist1);
//
//        (pk.histOfs[0] + blockHist)[h0] = hist0[0];
//        (pk.histOfs[1] + blockHist)[h0] = hist0[1];
//        (pk.histOfs[2] + blockHist)[h0] = hist0[2];
//        (pk.histOfs[3] + blockHist)[h0] = hist0[3];
//
//        (pk.histOfs[0] + blockHist)[h1] = hist1[0];
//        (pk.histOfs[1] + blockHist)[h1] = hist1[1];
//        (pk.histOfs[2] + blockHist)[h1] = hist1[2];
//        (pk.histOfs[3] + blockHist)[h1] = hist1[3];
    }
#else
    for( ; k < C4; k++ )
    {
        const PixData& pk = _pixData[k];
        const float* a = gradPtr + pk.gradOfs;
        float w, t0, t1, a0 = a[0], a1 = a[1];
        const uchar* h = qanglePtr + pk.qangleOfs;
        int h0 = h[0], h1 = h[1];

        float* hist = blockHist + pk.histOfs[0];
        w = pk.gradWeight*pk.histWeights[0];
        t0 = hist[h0] + a0*w;
        t1 = hist[h1] + a1*w;
        hist[h0] = t0; hist[h1] = t1;

        hist = blockHist + pk.histOfs[1];
        w = pk.gradWeight*pk.histWeights[1];
        t0 = hist[h0] + a0*w;
        t1 = hist[h1] + a1*w;
        hist[h0] = t0; hist[h1] = t1;

        hist = blockHist + pk.histOfs[2];
        w = pk.gradWeight*pk.histWeights[2];
        t0 = hist[h0] + a0*w;
        t1 = hist[h1] + a1*w;
        hist[h0] = t0; hist[h1] = t1;

        hist = blockHist + pk.histOfs[3];
        w = pk.gradWeight*pk.histWeights[3];
        t0 = hist[h0] + a0*w;
        t1 = hist[h1] + a1*w;
        hist[h0] = t0; hist[h1] = t1;
    }
#endif

    normalizeBlockHistogram(blockHist);

    return blockHist;
}

void HOGCache::normalizeBlockHistogram(float* _hist) const
{
    float* hist = &_hist[0], sum = 0.0f, partSum[4];
    size_t i = 0, sz = blockHistogramSize;

#ifdef UseSSE
    __m128 p0 = _mm_loadu_ps(hist);
    __m128 s = _mm_mul_ps(p0, p0);

    for (i = 4; i <= sz - 4; i += 4)
    {
        p0 = _mm_loadu_ps(hist + i);
        s = _mm_add_ps(s, _mm_mul_ps(p0, p0));
    }
    _mm_storeu_ps(partSum, s);
#else
    partSum[0] = 0.0f;
    partSum[1] = 0.0f;
    partSum[2] = 0.0f;
    partSum[3] = 0.0f;
    for ( ; i <= sz - 4; i += 4)
    {
        partSum[0] += hist[i] * hist[i];
        partSum[1] += hist[i+1] * hist[i+1];
        partSum[2] += hist[i+2] * hist[i+2];
        partSum[3] += hist[i+3] * hist[i+3];
    }
#endif
    float t0 = partSum[0] + partSum[1];
    float t1 = partSum[2] + partSum[3];
    sum = t0 + t1;
    for ( ; i < sz; ++i)
        sum += hist[i]*hist[i];

    float scale = 1.f/(std::sqrt(sum)+sz*0.1f), thresh = (float)descriptor->L2HysThreshold;
    i = 0, sum = 0.0f;

#ifdef UseSSE
    __m128 _scale = _mm_set1_ps(scale);
    static __m128 _threshold = _mm_set1_ps(thresh);

    __m128 p = _mm_mul_ps(_scale, _mm_loadu_ps(hist));
    p = _mm_min_ps(p, _threshold);
    s = _mm_mul_ps(p, p);
    _mm_storeu_ps(hist, p);

    for(i = 4 ; i <= sz - 4; i += 4)
    {
        p = _mm_loadu_ps(hist + i);
        p = _mm_mul_ps(p, _scale);
        p = _mm_min_ps(p, _threshold);
        s = _mm_add_ps(s, _mm_mul_ps(p, p));
        _mm_storeu_ps(hist + i, p);
    }

    _mm_storeu_ps(partSum, s);
#else
    partSum[0] = 0.0f;
    partSum[1] = 0.0f;
    partSum[2] = 0.0f;
    partSum[3] = 0.0f;
    for( ; i <= sz - 4; i += 4)
    {
        hist[i] = std::min(hist[i]*scale, thresh);
        hist[i+1] = std::min(hist[i+1]*scale, thresh);
        hist[i+2] = std::min(hist[i+2]*scale, thresh);
        hist[i+3] = std::min(hist[i+3]*scale, thresh);
        partSum[0] += hist[i]*hist[i];
        partSum[1] += hist[i+1]*hist[i+1];
        partSum[2] += hist[i+2]*hist[i+2];
        partSum[3] += hist[i+3]*hist[i+3];
    }
#endif
    t0 = partSum[0] + partSum[1];
    t1 = partSum[2] + partSum[3];
    sum = t0 + t1;
    for( ; i < sz; ++i)
    {
        hist[i] = std::min(hist[i]*scale, thresh);
        sum += hist[i]*hist[i];
    }

    scale = 1.f/(std::sqrt(sum)+1e-3f), i = 0;
#ifdef UseSSE
    __m128 _scale2 = _mm_set1_ps(scale);
    for ( ; i <= sz - 4; i += 4)
    {
        __m128 t = _mm_mul_ps(_scale2, _mm_loadu_ps(hist + i));
        _mm_storeu_ps(hist + i, t);
    }
#endif
    for ( ; i < sz; ++i)
        hist[i] *= scale;
}

Size HOGCache::windowsInImage(const Size& imageSize, const Size& winStride) const
{
//	printf("\nSize = %d x %d", imageSize.width, imageSize.height);
//	printf("\nStride = %d x %d", winStride.width, winStride.height);
    return Size((imageSize.width - winSize.width)/winStride.width + 1,
        (imageSize.height - winSize.height)/winStride.height + 1);
}

Rect HOGCache::getWindow(const Size& imageSize, const Size& winStride, int idx) const
{
    int nwindowsX = (imageSize.width - winSize.width)/winStride.width + 1;
    int y = idx / nwindowsX;
    int x = idx - nwindowsX*y;
    return Rect( x*winStride.width, y*winStride.height, winSize.width, winSize.height );
}

static inline int gcd(int a, int b)
{
    if( a < b )
        std::swap(a, b);
    while( b > 0 )
    {
        int r = a % b;
        a = b;
        b = r;
    }
    return a;
}


void HOGDescriptor_Mod::compute(InputArray _img, std::vector<float>& descriptors,
    Size winStride, Size padding, const std::vector<Point>& locations) const
{
    if( winStride == Size() )
        winStride = cellSize;
    Size cacheStride(gcd(winStride.width, blockStride.width),
                     gcd(winStride.height, blockStride.height));

    Size imgSize = _img.size();

    size_t nwindows = locations.size();
    padding.width = (int)alignSize(std::max(padding.width, 0), cacheStride.width);
    padding.height = (int)alignSize(std::max(padding.height, 0), cacheStride.height);
    Size paddedImgSize(imgSize.width + padding.width*2, imgSize.height + padding.height*2);

   /* CV_OCL_RUN(_img.dims() <= 2 && _img.type() == CV_8UC1 && _img.isUMat(),
        ocl_compute(_img, winStride, descriptors, DESCR_FORMAT_COL_BY_COL, blockSize,
        cellSize, nbins, blockStride, winSize, (float)getWinSigma(), gammaCorrection, L2HysThreshold, signedGradient))
		*/
    Mat img = _img.getMat();
    HOGCache cache(this, img, padding, padding, nwindows == 0, cacheStride);

    if( !nwindows )
        nwindows = cache.windowsInImage(paddedImgSize, winStride).area();

    const HOGCache::BlockData* blockData = &cache.blockData[0];

    int nblocks = cache.nblocks.area();
    int blockHistogramSize = cache.blockHistogramSize;
    size_t dsize = getDescriptorSize();
    descriptors.resize(dsize*nwindows);

    // for each window
    for( size_t i = 0; i < nwindows; i++ )
    {
        float* descriptor = &descriptors[i*dsize];

        Point pt0;
        if( !locations.empty() )
        {
            pt0 = locations[i];
            if( pt0.x < -padding.width || pt0.x > img.cols + padding.width - winSize.width ||
                pt0.y < -padding.height || pt0.y > img.rows + padding.height - winSize.height )
                continue;
        }
        else
        {
            pt0 = cache.getWindow(paddedImgSize, winStride, (int)i).tl() - Point(padding);
//            CV_Assert(pt0.x % cacheStride.width == 0 && pt0.y % cacheStride.height == 0);
        }

        for( int j = 0; j < nblocks; j++ )
        {
            const HOGCache::BlockData& bj = blockData[j];
            Point pt = pt0 + bj.imgOffset;

            float* dst = descriptor + bj.histOfs;
            const float* src = cache.getBlock(pt, dst);
            if( src != dst )
                memcpy(dst, src, blockHistogramSize * sizeof(float));
        }
    }
}
float Evaluate_SVM_Dot_Product(const HOGDescriptor_Mod *hog_Object_ptr, HOGCache &cache, Point pt0);
float Evaluate_SVM_LUT(const HOGDescriptor_Mod *hog_Object_ptr, HOGCache &cache, Point pt0);

void HOGDescriptor_Mod::detect(const Mat& img,
    std::vector<Point>& hits, std::vector<double>& weights, double hitThreshold,
    Size winStride, Size padding, const std::vector<Point>& locations) const
{
    hits.clear();
    weights.clear();
    if( svmDetector.empty() )
        return;

    if( winStride == Size() )
        winStride = cellSize;
    Size cacheStride(gcd(winStride.width, blockStride.width),
        gcd(winStride.height, blockStride.height));

    size_t nwindows = locations.size();
    padding.width = (int)alignSize(std::max(padding.width, 0), cacheStride.width);
    padding.height = (int)alignSize(std::max(padding.height, 0), cacheStride.height);
    Size paddedImgSize(img.cols + padding.width*2, img.rows + padding.height*2);

    HOGCache cache(this, img, padding, padding, nwindows == 0, cacheStride);

    if( !nwindows )
        nwindows = cache.windowsInImage(paddedImgSize, winStride).area();

#ifdef UseSSE
    float partSum[4];
#endif
//	printf("\n%d windows", nwindows); getchar();
//	ftime(&t_start);
    for( size_t i = 0; i < nwindows; i++ )
    {
        Point pt0;
        if( !locations.empty() )
        {
            pt0 = locations[i];
            if( pt0.x < -padding.width || pt0.x > img.cols + padding.width - winSize.width ||
                    pt0.y < -padding.height || pt0.y > img.rows + padding.height - winSize.height )
                continue;
        }
        else
        {
            pt0 = cache.getWindow(paddedImgSize, winStride, (int)i).tl() - Point(padding);
            CV_Assert(pt0.x % cacheStride.width == 0 && pt0.y % cacheStride.height == 0);
        }

		float s = -10;
		switch (SVM_Eval_Method) {
		case SVM_Dot_Product:
			s = Evaluate_SVM_Dot_Product(this, cache, pt0);
			break;
		case SVM_LUT:
			s = Evaluate_SVM_LUT(this, cache, pt0);
			break;
		default:
			break;
		}
//		printf("\n%.2f", s); getchar();
        if( s >= hitThreshold )
        {
            hits.push_back(pt0);
            weights.push_back(s);
        }
    }
//	ftime(&t_end);
}

float Evaluate_SVM_Dot_Product(const HOGDescriptor_Mod *hog_Object_ptr, HOGCache &cache, Point pt0) {
	size_t dsize = (*hog_Object_ptr).getDescriptorSize();
	double rho = (*hog_Object_ptr).svmDetector[dsize];
	float s = (*hog_Object_ptr).svmDetector.size() > dsize ? (*hog_Object_ptr).svmDetector[dsize] : 0;
	const float* svmVec = &((*hog_Object_ptr).svmDetector[0]);
	int nblocks = cache.nblocks.area();
	const HOGCache::BlockData* blockData = &cache.blockData[0];
	int blockHistogramSize = cache.blockHistogramSize;
	std::vector<float> blockHist(blockHistogramSize);
	int indx = 0;
#ifdef UseSSE
	float partSum[4];
#endif
	int j, k;
	for (j = 0; j < nblocks; j++, svmVec += blockHistogramSize)
	{
		const HOGCache::BlockData& bj = blockData[j];
		Point pt = pt0 + bj.imgOffset;

		const float* vec = cache.getBlock(pt, &blockHist[0]);
#ifdef UseSSE
		__m128 _vec = _mm_loadu_ps(vec);
		__m128 _svmVec = _mm_loadu_ps(svmVec);
		__m128 sum = _mm_mul_ps(_svmVec, _vec);

		for (k = 4; k <= blockHistogramSize - 4; k += 4)
		{
			_vec = _mm_loadu_ps(vec + k);
			_svmVec = _mm_loadu_ps(svmVec + k);

			sum = _mm_add_ps(sum, _mm_mul_ps(_vec, _svmVec));
		}

		_mm_storeu_ps(partSum, sum);
		double t0 = partSum[0] + partSum[1];
		double t1 = partSum[2] + partSum[3];
		s += t0 + t1;
#else
		for (k = 0; k <= blockHistogramSize - 4; k += 4)
			s += vec[k] * svmVec[k] + vec[k + 1] * svmVec[k + 1] +
			vec[k + 2] * svmVec[k + 2] + vec[k + 3] * svmVec[k + 3];
#endif
		for (; k < blockHistogramSize; k++)
			s += vec[k] * svmVec[k];
	}

	/*int j, k;
	for (j = 0; j < nblocks; j++)//, svmVec += blockHistogramSize)
	{
		const HOGCache::BlockData& bj = blockData[j];
		Point pt = pt0 + bj.imgOffset;
		const float* vec = cache.getBlock(pt, &blockHist[0]);
		k = 0;
		for (; k < blockHistogramSize; k++)
			s += Linear_LUT_HOG_Optimal_48_96[int(vec[k]*64)][indx++];
			//s += vec[k] * svmVec[indx++];
	}*/
	return s;
}

float Evaluate_SVM_LUT(const HOGDescriptor_Mod *hog_Object_ptr, HOGCache &cache, Point pt0) {
	size_t dsize = (*hog_Object_ptr).getDescriptorSize();
	double rho = (*hog_Object_ptr).svmDetector[dsize];
	const float* svmVec = &((*hog_Object_ptr).svmDetector[0]);
	int nblocks = cache.nblocks.area();
	const HOGCache::BlockData* blockData = &cache.blockData[0];
	int blockHistogramSize = cache.blockHistogramSize;
	std::vector<float> blockHist(blockHistogramSize);
	int indx = 0;
	int j, k;
	float s = (*hog_Object_ptr).svmDetectorBias;

	for (j = 0; j < nblocks; j++)
	{
		const HOGCache::BlockData& bj = blockData[j];
		Point pt = pt0 + bj.imgOffset;
		const float* vec = cache.getBlock(pt, &blockHist[0]);
		k = 0;
		for (; k < blockHistogramSize; k++) {
			float feature_value = vec[k];
			//feature_value = pow(feature_value, 0.25);
			int feature_value_q = int(feature_value * ((*hog_Object_ptr).svmDetectorLUT_Q));
//			if (feature_value_q > 4) feature_value_q = 15;	//use upper value (ceil) when saturating
//			if (feature_value_q < 2) feature_value_q = 0;
			//if (feature_value_q > 10) feature_value_q = 31;	//use upper value (ceil) when saturating
			//if (feature_value_q < 3) feature_value_q = 0;
			s += (*hog_Object_ptr).svmDetectorLUT[feature_value_q][indx++];
		}
	}
//	printf("\n%.2f, %d", s, (*hog_Object_ptr).svmDetectorLUT_Q); getchar();
	return s;
}

void HOGDescriptor_Mod::detect(const Mat& img, std::vector<Point>& hits, double hitThreshold,
    Size winStride, Size padding, const std::vector<Point>& locations) const
{
    std::vector<double> weightsV;
    detect(img, hits, weightsV, hitThreshold, winStride, padding, locations);
}

class HOGInvoker :
    public ParallelLoopBody
{
public:
    HOGInvoker( const HOGDescriptor_Mod* _hog, const Mat& _img,
        double _hitThreshold, const Size& _winStride, const Size& _padding,
        const double* _levelScale, std::vector<Rect> * _vec, Mutex* _mtx,
        std::vector<double>* _weights=0, std::vector<double>* _scales=0 )
    {
        hog = _hog;
        img = _img;
        hitThreshold = _hitThreshold;
        winStride = _winStride;
        padding = _padding;
        levelScale = _levelScale;
        vec = _vec;
        weights = _weights;
        scales = _scales;
        mtx = _mtx;
    }

    void operator()( const Range& range ) const
    {
        int i, i1 = range.start, i2 = range.end;
        double minScale = i1 > 0 ? levelScale[i1] : i2 > 1 ? levelScale[i1+1] : std::max(img.cols, img.rows);
        Size maxSz(cvCeil(img.cols/minScale), cvCeil(img.rows/minScale));
        Mat smallerImgBuf(maxSz, img.type());
        std::vector<Point> locations;
        std::vector<double> hitsWeights;

        for( i = i1; i < i2; i++ )
        {
            double scale = levelScale[i];
			//printf("\n%.2f", scale);
            Size sz(cvRound(img.cols/scale), cvRound(img.rows/scale));
            Mat smallerImg(sz, img.type(), smallerImgBuf.ptr());
            if( sz == img.size() )
                smallerImg = Mat(sz, img.type(), img.data, img.step);
            else
                resize(img, smallerImg, sz);
			//ftime(&t_start);
            hog->detect(smallerImg, locations, hitsWeights, hitThreshold, winStride, padding);
			//ftime(&t_end);
            Size scaledWinSize = Size(cvRound(hog->winSize.width*scale), cvRound(hog->winSize.height*scale));

            mtx->lock();
            for( size_t j = 0; j < locations.size(); j++ )
            {
                vec->push_back(Rect(cvRound(locations[j].x*scale),
                                    cvRound(locations[j].y*scale),
                                    scaledWinSize.width, scaledWinSize.height));
                if (scales)
                    scales->push_back(scale);
            }
            mtx->unlock();

            if (weights && (!hitsWeights.empty()))
            {
                mtx->lock();
                for (size_t j = 0; j < locations.size(); j++)
                    weights->push_back(hitsWeights[j]);
                mtx->unlock();
            }
        }
    }

private:
    const HOGDescriptor_Mod* hog;
    Mat img;
    double hitThreshold;
    Size winStride;
    Size padding;
    const double* levelScale;
    std::vector<Rect>* vec;
    std::vector<double>* weights;
    std::vector<double>* scales;
    Mutex* mtx;
};


void HOGDescriptor_Mod::detectMultiScale(
    InputArray _img, std::vector<Rect>& foundLocations, std::vector<double>& foundWeights,
    double hitThreshold, Size winStride, Size padding,
    double scale0, double finalThreshold, bool useMeanshiftGrouping) const
{
    double scale = 1.;
    int levels = 0;

    Size imgSize = _img.size();
    std::vector<double> levelScale;
    for( levels = 0; levels < nlevels; levels++ )
    {
        levelScale.push_back(scale);
        if( cvRound(imgSize.width/scale) < winSize.width ||
            cvRound(imgSize.height/scale) < winSize.height ||
                scale0 <= 1 )
            break;
        scale *= scale0;
    }
    levels = std::max(levels, 1);
    levelScale.resize(levels);

    if(winStride == Size())
        winStride = blockStride;

    std::vector<Rect> allCandidates;
    std::vector<double> tempScales;
    std::vector<double> tempWeights;
    std::vector<double> foundScales;

    Mutex mtx;
    Mat img = _img.getMat();
    Range range(0, (int)levelScale.size());
    HOGInvoker invoker(this, img, hitThreshold, winStride, padding, &levelScale[0], &allCandidates, &mtx, &tempWeights, &tempScales);
    parallel_for_(range, invoker);

    std::copy(tempScales.begin(), tempScales.end(), back_inserter(foundScales));
    foundLocations.clear();
    std::copy(allCandidates.begin(), allCandidates.end(), back_inserter(foundLocations));
    foundWeights.clear();
    std::copy(tempWeights.begin(), tempWeights.end(), back_inserter(foundWeights));

    if ( useMeanshiftGrouping )
        groupRectangles_meanshift(foundLocations, foundWeights, foundScales, finalThreshold, winSize);
    else
        groupRectangles(foundLocations, foundWeights, (int)finalThreshold, 0.2);
    clipObjects_mod(imgSize, foundLocations, 0, &foundWeights);
}
void clipObjects_mod(Size sz, std::vector<Rect>& objects,
	std::vector<int>* a, std::vector<double>* b)
{
	size_t i, j = 0, n = objects.size();
	Rect win0 = Rect(0, 0, sz.width, sz.height);
	if (a)
	{
		CV_Assert(a->size() == n);
	}
	if (b)
	{
		CV_Assert(b->size() == n);
	}

	for (i = 0; i < n; i++)
	{
		Rect r = win0 & objects[i];
		if (r.area() > 0)
		{
			objects[j] = r;
			if (i > j)
			{
				if (a) a->at(j) = a->at(i);
				if (b) b->at(j) = b->at(i);
			}
			j++;
		}
	}

	if (j < n)
	{
		objects.resize(j);
		if (a) a->resize(j);
		if (b) b->resize(j);
	}
}


void HOGDescriptor_Mod::detectMultiScale(InputArray img, std::vector<Rect>& foundLocations,
    double hitThreshold, Size winStride, Size padding,
    double scale0, double finalThreshold, bool useMeanshiftGrouping) const
{
    std::vector<double> foundWeights;
    detectMultiScale(img, foundLocations, foundWeights, hitThreshold, winStride,
                padding, scale0, finalThreshold, useMeanshiftGrouping);
}

template<typename _ClsName> struct RTTIImpl
{
public:
    static int isInstance(const void* ptr)
    {
        static _ClsName dummy;
        static void* dummyp = &dummy;
        union
        {
            const void* p;
            const void** pp;
        } a, b;
        a.p = dummyp;
        b.p = ptr;
        return *a.pp == *b.pp;
    }
    static void release(void** dbptr)
    {
        if(dbptr && *dbptr)
        {
            delete (_ClsName*)*dbptr;
            *dbptr = 0;
        }
    }
    static void* read(CvFileStorage* fs, CvFileNode* n)
    {
        FileNode fn(fs, n);
        _ClsName* obj = new _ClsName;
        if(obj->read(fn))
            return obj;
        delete obj;
        return 0;
    }

    static void write(CvFileStorage* _fs, const char* name, const void* ptr, CvAttrList)
    {
        if(ptr && _fs)
        {
            FileStorage fs(_fs, false);
            ((const _ClsName*)ptr)->write(fs, String(name));
        }
    }

    static void* clone(const void* ptr)
    {
        if(!ptr)
            return 0;
        return new _ClsName(*(const _ClsName*)ptr);
    }
};

typedef RTTIImpl<HOGDescriptor_Mod> HOGRTTI;

CvType hog_type( CV_TYPE_NAME_HOG_DESCRIPTOR, HOGRTTI::isInstance,
    HOGRTTI::release, HOGRTTI::read, HOGRTTI::write, HOGRTTI::clone);

std::vector<float> HOGDescriptor_Mod::HOG_Caltech_64_128()
{
	static const float detector[] = { 0.01878615f, -0.00224480f, -0.03305347f, -0.01076582f, -0.04761570f, 0.03781769f, 0.01255422f, 0.00814729f, -0.00308390f, 0.01071725f, 0.01519414f, -0.02110975f, 0.00651120f, 0.00977023f, 0.03131209f, -0.02072689f, -0.01867912f, -0.00003144f, 0.05978794f, -0.01111414f, -0.05535992f, 0.04544152f, -0.04320488f, 0.06711376f, -0.00669560f, 0.00394298f, 0.04970749f, 0.06159376f, 0.04355074f, -0.04046136f, 0.06672414f, 0.02844378f, 0.05255666f, -0.01020792f, 0.01744778f, 0.06275862f, 0.05239862f, -0.00075797f, -0.01779786f, 0.00403716f, 0.00125596f, 0.01293445f, -0.02267774f, -0.02031794f, 0.01622410f, 0.04503696f, -0.03087682f, -0.07381556f, 0.01377302f, -0.01573495f, 0.03679933f, -0.02347934f, -0.02800209f, 0.00639900f, 0.05805808f, 0.00514531f, -0.03435531f, 0.00374873f, 0.01171149f, 0.06731043f, 0.00112986f, 0.00519582f, 0.04973477f, 0.03643309f, -0.01442250f, -0.03112136f, 0.03189351f, -0.04499474f, 0.06425123f, 0.00160228f, -0.01600682f, 0.03220969f, 0.03993951f, -0.02855881f, -0.05209760f, 0.00731726f, -0.00644758f, -0.00339974f, -0.03777518f, -0.02009460f, 0.00281665f, 0.04193018f, -0.00109246f, -0.01556983f, 0.07189695f, 0.00812753f, 0.06738249f, -0.01443850f, 0.03561806f, 0.02918687f, 0.02329149f, -0.01523963f, -0.01691724f, 0.03402525f, -0.02232152f, 0.01648022f, -0.03447576f, -0.02797463f, 0.01301329f, 0.00155596f, -0.04383043f, -0.02763272f, 0.05489206f, -0.02750145f, 0.04975490f, 0.02872430f, 0.01264451f, 0.02495661f, 0.02591371f, -0.00654682f, -0.02058307f, 0.00540756f, -0.03459801f, -0.00311912f, -0.00998795f, 0.02257325f, 0.00098594f, 0.04131031f, 0.04236069f, -0.00794797f, 0.00252418f, -0.00775549f, 0.00805873f, 0.03835568f, 0.04137515f, -0.01320607f, 0.02677877f, -0.02265952f, -0.03218769f, -0.00274031f, -0.00321096f, 0.00302229f, 0.01595063f, 0.01094805f, 0.03425932f, -0.01981592f, 0.01075689f, -0.02698620f, 0.02564003f, -0.06231862f, 0.02310784f, -0.02306248f, -0.01246849f, -0.01748855f, -0.01070399f, -0.01125699f, -0.02970074f, -0.00844218f, 0.02160419f, -0.01544753f, 0.01475993f, -0.00905602f, -0.03145740f, 0.00561503f, 0.02137271f, -0.00013095f, 0.00241055f, -0.02211648f, 0.00859307f, 0.02040288f, 0.01259598f, 0.01532595f, -0.02420474f, -0.01002852f, -0.04815746f, -0.02882151f, 0.00261820f, 0.01167782f, -0.00748363f, 0.00464625f, -0.00869305f, -0.03450755f, -0.02010795f, -0.02038800f, 0.01423306f, 0.01798106f, 0.07993855f, 0.04950993f, 0.04288238f, -0.01365891f, 0.01422828f, 0.03627339f, 0.00836020f, -0.00197165f, -0.01070774f, 0.00246732f, -0.01218758f, 0.01190279f, 0.02711796f, 0.00910158f, -0.00423262f, -0.04436307f, 0.01461259f, -0.05863661f, -0.02567520f, -0.04432131f, 0.00691319f, 0.00795899f, 0.04953428f, 0.05837589f, -0.00915023f, 0.00591981f, 0.02566255f, 0.04414917f, -0.00715663f, 0.01241493f, 0.05138842f, 0.01422953f, -0.00504466f, -0.07314556f, 0.00750270f, 0.00093678f, 0.00550983f, -0.04361594f, 0.04875571f, 0.03013145f, 0.00848191f, 0.01031805f, -0.04647661f, 0.01675869f, 0.03131573f, 0.03134316f, -0.01936358f, 0.06038631f, 0.02258528f, -0.02742412f, 0.01116907f, -0.05641141f, 0.00094944f, 0.01525527f, 0.04012037f, -0.00406980f, 0.02618699f, 0.00436621f, 0.04929439f, 0.02043071f, -0.01571116f, 0.04602130f, 0.01655618f, 0.04414966f, -0.03242586f, 0.05560550f, 0.08640176f, 0.01369000f, 0.04210778f, 0.00281001f, 0.01687429f, -0.02318861f, -0.02555736f, -0.03499472f, 0.02562536f, 0.03069020f, 0.01217333f, 0.01885820f, 0.01551607f, 0.08087558f, 0.06168952f, 0.03683213f, 0.00218505f, 0.02280326f, 0.00211232f, 0.00371010f, -0.00804696f, -0.02482755f, 0.00396834f, -0.00320265f, -0.02248860f, -0.01458372f, -0.00806090f, -0.03703907f, 0.05352692f, 0.04959437f, 0.04673141f, 0.04968135f, 0.01989802f, 0.03727668f, 0.02261548f, 0.03582555f, 0.01830862f, -0.03780156f, -0.03968223f, -0.01372534f, 0.00019492f, 0.00264289f, -0.01840128f, -0.01017879f, -0.00249562f, -0.02599895f, 0.03487265f, 0.02682274f, 0.00810733f, 0.00515660f, -0.04277944f, 0.03917482f, 0.01698872f, 0.01315881f, 0.01466563f, -0.01938129f, -0.00452694f, -0.01941501f, -0.03256627f, -0.06716925f, -0.01144764f, -0.00401895f, -0.00003879f, 0.00968020f, 0.00761431f, 0.00790924f, 0.00866762f, 0.01742959f, 0.02793883f, 0.04119500f, 0.02995397f, 0.03820651f, 0.03055941f, -0.00639612f, -0.02001055f, -0.05024068f, -0.00035133f, -0.03891998f, 0.03098672f, 0.00520150f, 0.00049107f, -0.00035156f, -0.04488682f, 0.04498577f, 0.03054162f, -0.00461370f, -0.01266082f, 0.00257561f, -0.01127144f, -0.01507874f, -0.04804970f, -0.02412670f, 0.02201555f, -0.03229836f, -0.03024793f, -0.02338988f, -0.06029305f, -0.03453654f, -0.01640072f, -0.02443036f, -0.02344909f, 0.03319829f, 0.01775788f, 0.01058227f, -0.01642074f, 0.02718535f, -0.02038161f, -0.01546660f, -0.02579355f, -0.04290596f, 0.00708888f, -0.02088509f, -0.01882634f, -0.02395057f, -0.04272034f, -0.06065369f, -0.03324870f, -0.04620404f, -0.00098986f, 0.01275334f, -0.03141110f, 0.00959781f, -0.01425619f, -0.01672342f, 0.00806872f, 0.00666822f, 0.00510972f, -0.05605917f, -0.03861942f, -0.02997671f, -0.04712919f, -0.02585302f, -0.05982759f, -0.05113042f, -0.01916405f, -0.02860450f, -0.02018336f, -0.00849691f, -0.02522062f, 0.01047286f, -0.00649162f, -0.02352597f, -0.02705241f, 0.00239339f, -0.00366033f, -0.02952261f, -0.02420249f, -0.01775941f, -0.01720546f, -0.00036285f, -0.06460640f, -0.05531412f, -0.01274440f, -0.01326469f, -0.05486958f, -0.05062567f, -0.04233640f, -0.03647200f, 0.01570783f, -0.00572922f, 0.01115204f, 0.02112476f, -0.03032519f, -0.09522374f, -0.08682903f, -0.08143655f, -0.02502729f, -0.03944855f, -0.06365364f, -0.04751637f, 0.01160544f, -0.05691842f, -0.00268091f, 0.00573614f, 0.04085666f, 0.03355846f, 0.06143564f, 0.00014056f, -0.01973156f, -0.01350417f, -0.01931319f, -0.01686522f, 0.00437075f, 0.01348825f, 0.02886735f, 0.00658074f, -0.03372642f, -0.02025649f, 0.01631809f, -0.02626850f, -0.04646355f, -0.03681832f, -0.00569603f, 0.02315524f, -0.04862328f, -0.06370025f, -0.03458511f, 0.03235624f, -0.01849243f, -0.06545150f, -0.07886354f, -0.05151476f, -0.05200773f, -0.07820530f, -0.08052730f, -0.02043218f, -0.01624940f, -0.02751788f, 0.01329739f, -0.00149089f, 0.00128997f, 0.02502724f, 0.04567342f, 0.04018838f, 0.05380138f, 0.07079116f, 0.03369334f, -0.01868505f, -0.05108846f, -0.00617552f, 0.02374986f, 0.00003590f, 0.02426955f, 0.06643048f, 0.05872232f, 0.02908830f, -0.01303039f, -0.03125131f, -0.00403865f, -0.00922725f, -0.10686526f, -0.01000386f, -0.01927680f, -0.02910918f, 0.00354354f, -0.05008551f, -0.07685826f, -0.03685815f, -0.01146767f, -0.06096699f, 0.00392004f, -0.09956824f, -0.06797030f, -0.04788693f, 0.04223121f, -0.00502165f, 0.05285521f, 0.08897761f, -0.04355615f, 0.05616595f, 0.03004157f, 0.01666109f, 0.03196988f, 0.03848012f, -0.02513197f, -0.00099939f, 0.03200506f, -0.00512786f, 0.03185532f, -0.06581582f, -0.03791193f, 0.00948701f, -0.02951104f, -0.04869721f, -0.01827341f, -0.05960644f, -0.01388069f, 0.01998290f, -0.04529953f, -0.02802809f, -0.03186630f, -0.04151278f, -0.05244066f, 0.00686334f, -0.05872747f, -0.07613711f, 0.03227504f, -0.02877223f, -0.04596235f, -0.03927031f, 0.08037110f, -0.00231234f, 0.00996984f, -0.02468206f, -0.02159137f, -0.00904433f, -0.07602027f, 0.00573140f, 0.06787559f, 0.03090949f, -0.02982951f, 0.03148168f, -0.02576962f, -0.01866328f, 0.00278969f, -0.04030286f, -0.01880442f, 0.04208690f, 0.03471593f, -0.04076562f, -0.04870032f, 0.06015910f, 0.02627477f, 0.03093049f, -0.00228369f, -0.00604005f, 0.03389845f, 0.03050134f, 0.00185682f, -0.01692066f, 0.05438389f, 0.04475771f, 0.05241127f, -0.01241687f, -0.01239041f, 0.03249672f, 0.06668840f, -0.07052792f, -0.03000933f, 0.04646800f, -0.04391988f, 0.01337189f, -0.00935302f, -0.04729802f, 0.06351820f, 0.02755227f, 0.01879353f, 0.00413875f, 0.05837946f, -0.02286712f, 0.04545523f, 0.00530983f, -0.03953148f, 0.03884873f, 0.04554997f, -0.03166009f, -0.05018013f, 0.01800662f, 0.07191218f, 0.04952503f, 0.00100079f, -0.01824730f, 0.00741504f, 0.06880866f, -0.02108322f, -0.02768181f, 0.01968195f, 0.04997525f, 0.03435347f, 0.01412299f, -0.00497584f, 0.00750561f, 0.01766906f, 0.01363421f, -0.04113400f, 0.01185545f, 0.00345288f, 0.04984643f, -0.01536406f, -0.05662654f, 0.01819404f, 0.07584516f, 0.03185353f, 0.00131180f, 0.03366906f, -0.03805275f, 0.01339542f, -0.02126796f, -0.03527384f, 0.06680050f, 0.04616583f, -0.02935499f, -0.04472774f, 0.00754837f, 0.04595018f, 0.00708662f, -0.01315603f, -0.02768559f, 0.00871233f, 0.09047404f, -0.03404412f, -0.04215214f, 0.04665599f, 0.06776419f, 0.05989181f, 0.03388727f, 0.00810483f, 0.07521911f, 0.08202036f, 0.01053764f, -0.00348686f, 0.03046953f, 0.03565598f, -0.00249432f, -0.03162839f, -0.00334445f, 0.07120519f, 0.04107946f, -0.02022400f, -0.00400223f, 0.04569995f, 0.05388835f, 0.04853686f, -0.01659378f, 0.02115829f, 0.05908127f, 0.03537357f, -0.07249857f, -0.04529286f, -0.03338842f, 0.01928404f, -0.03323337f, -0.00192265f, -0.02171320f, 0.02365060f, 0.03875748f, -0.00244120f, -0.00763417f, -0.02771138f, -0.01006848f, 0.02054860f, -0.00969891f, -0.01690868f, 0.01771504f, 0.00508429f, -0.04909355f, 0.02024281f, 0.01945603f, -0.01450953f, -0.04329425f, -0.03208257f, -0.01926724f, 0.03460552f, 0.07044353f, 0.08196673f, 0.07021362f, 0.04593536f, -0.02241251f, 0.00634200f, 0.00752005f, -0.00007809f, 0.04422848f, 0.01627029f, -0.01318782f, -0.03031459f, -0.06216414f, 0.05250832f, 0.03250695f, -0.01711125f, -0.03033974f, -0.02353684f, 0.01361174f, 0.01693639f, -0.02522164f, -0.02456903f, 0.04712125f, 0.05595002f, 0.01948341f, -0.01475495f, -0.00794222f, 0.04258821f, 0.09379890f, 0.04182860f, -0.03635959f, 0.00035610f, 0.01081854f, -0.01576208f, -0.04977076f, 0.01643952f, 0.01648422f, 0.10993852f, -0.00560671f, -0.06624397f, -0.01208824f, 0.00594833f, -0.02288078f, -0.04595725f, -0.00959603f, 0.05023192f, 0.04517420f, -0.01786974f, -0.00420771f, 0.05629241f, 0.05155307f, 0.00804913f, 0.01057198f, 0.02247748f, 0.00109328f, 0.02200699f, -0.03454457f, -0.00830603f, 0.04979858f, 0.02270294f, -0.02912070f, 0.04594105f, 0.01794234f, 0.00538326f, 0.11094231f, -0.01567710f, -0.03071072f, 0.04186745f, 0.02804785f, 0.02051192f, -0.00967606f, -0.00682115f, -0.02342160f, 0.07137154f, -0.01798659f, -0.03012891f, 0.02563761f, 0.01546795f, 0.01463471f, 0.00662048f, 0.01450856f, 0.02247272f, 0.03090183f, -0.01659340f, 0.01677277f, 0.05202847f, 0.02424237f, -0.02546598f, 0.03357662f, 0.02481197f, 0.01558044f, 0.04399069f, 0.00710036f, -0.02462604f, 0.02279598f, -0.02449357f, -0.05862407f, -0.00657625f, 0.00861676f, 0.04465102f, 0.04640190f, 0.00566383f, -0.02003709f, 0.05088420f, -0.01294057f, -0.01627851f, 0.02321889f, 0.02618158f, 0.06150235f, 0.06487139f, 0.00621005f, -0.08210723f, 0.01871566f, -0.02672593f, -0.07819312f, -0.01187479f, 0.04468893f, 0.02770685f, 0.05309300f, 0.03072974f, 0.01325862f, 0.06124949f, 0.00145960f, 0.01612926f, 0.02460489f, 0.00824257f, -0.04786992f, -0.02351774f, -0.01896202f, -0.00872302f, -0.00005381f, -0.02636168f, -0.00943967f, -0.01161065f, -0.04546176f, 0.08807520f, 0.08018527f, 0.00788398f, -0.00189845f, 0.03128608f, 0.03960265f, 0.01717599f, 0.07106238f, 0.05914197f, 0.02211003f, 0.05819116f, 0.00100594f, 0.00500810f, 0.02728376f, 0.01208866f, 0.00370917f, 0.01055306f, -0.00266357f, -0.01516315f, -0.00327448f, 0.00957515f, 0.00060689f, 0.08241424f, 0.01563829f, 0.00569184f, 0.02256975f, -0.01534976f, -0.01193535f, -0.04596096f, -0.07003870f, -0.02988664f, 0.02845015f, -0.00893630f, -0.05167069f, -0.03244246f, -0.04101194f, 0.06371104f, 0.07249369f, 0.04884499f, 0.03429425f, 0.08914798f, 0.04426282f, 0.02009486f, 0.02856913f, 0.00289265f, 0.00389768f, 0.00138804f, -0.02470823f, -0.00498751f, 0.08587399f, -0.00815698f, -0.03624542f, -0.01263801f, -0.01917417f, 0.01396572f, 0.01471740f, -0.02937133f, -0.01148474f, 0.04366339f, 0.01525570f, -0.03679936f, -0.01236677f, -0.01352501f, -0.03561108f, -0.00405417f, -0.03151723f, -0.03725636f, 0.04221789f, -0.02348614f, -0.04330934f, -0.04562415f, -0.05984882f, 0.03310219f, 0.06823277f, -0.02183874f, -0.01974998f, 0.07003755f, 0.01823736f, -0.01817426f, 0.02902181f, -0.00193999f, -0.02300504f, 0.05376674f, 0.00544442f, -0.01517805f, 0.06468066f, -0.00845607f, -0.01702271f, -0.00808327f, -0.03106012f, -0.03189405f, -0.02025841f, -0.03715266f, -0.03695754f, 0.04951455f, -0.01812366f, -0.03216298f, 0.01189878f, -0.02247317f, -0.02823271f, -0.01495308f, -0.02174564f, -0.05947656f, 0.00348373f, -0.04223281f, -0.05466698f, -0.02617687f, -0.04033536f, 0.00849450f, 0.04942207f, 0.03046953f, 0.00870480f, 0.07235379f, 0.05089927f, 0.00631786f, 0.01810571f, -0.01170337f, 0.00306705f, 0.05032378f, 0.02630168f, -0.00143925f, 0.01352965f, -0.02542269f, -0.04013560f, 0.00273851f, -0.00330418f, -0.02264706f, 0.03429442f, 0.01482990f, -0.02923329f, -0.00812828f, -0.00609001f, -0.03701995f, -0.05158155f, -0.07048850f, -0.02279143f, 0.00101425f, -0.01906559f, -0.04745854f, -0.00229577f, -0.03529403f, -0.00729902f, -0.03536178f, -0.05553189f, 0.01810220f, 0.05157120f, 0.02071786f, 0.00980351f, 0.01618167f, 0.02706701f, -0.01044082f, -0.01607750f, -0.01621159f, 0.03926143f, 0.08153149f, 0.00943576f, -0.03210470f, 0.00626484f, 0.01064815f, -0.00273658f, -0.00135677f, -0.01253019f, -0.00828614f, -0.00703572f, -0.02294152f, -0.02294706f, 0.01701245f, 0.00332740f, 0.05186590f, 0.02163057f, -0.00081042f, -0.05766790f, -0.06949800f, -0.05761674f, -0.04706332f, -0.00858485f, -0.01752658f, 0.04729485f, 0.01225224f, -0.01805644f, 0.05268281f, 0.05398864f, 0.03294851f, 0.05695528f, 0.04652197f, 0.05964293f, 0.02541851f, 0.02059378f, -0.03335733f, 0.01762076f, 0.01349597f, 0.00831240f, 0.02341720f, 0.04725093f, 0.04251639f, 0.05762675f, 0.06324549f, 0.01730930f, 0.00382248f, -0.00718645f, 0.01437234f, 0.03382857f, 0.02619687f, 0.03847860f, 0.05124785f, 0.00603498f, -0.01682648f, -0.00631926f, -0.02841766f, -0.02795481f, -0.00709740f, -0.00951197f, 0.02808291f, -0.03727986f, -0.06153748f, -0.03719382f, 0.02806077f, 0.04762835f, 0.02111524f, 0.03377315f, 0.06668191f, 0.09871583f, 0.05343286f, 0.05420946f, -0.00052302f, -0.02721730f, -0.02694378f, -0.04042410f, -0.01073598f, 0.01844405f, 0.02933091f, -0.05995428f, -0.04777454f, -0.04303703f, 0.03091716f, -0.00107249f, -0.00309212f, -0.03847935f, -0.00685211f, 0.01219424f, -0.05260520f, -0.02613452f, 0.00985969f, -0.02909327f, -0.04224715f, 0.02811654f, -0.01408360f, 0.02271849f, 0.02043948f, -0.04981243f, -0.04831923f, -0.00967067f, 0.01950765f, -0.00104796f, 0.01346978f, -0.03580245f, -0.00255314f, 0.04843141f, -0.01949712f, -0.01113366f, 0.01807568f, -0.03013401f, -0.04109807f, 0.02939116f, -0.04308976f, -0.00192394f, 0.07085702f, -0.01744486f, -0.05049078f, -0.03788495f, -0.00105370f, -0.07143075f, -0.04404131f, 0.04617411f, 0.04923454f, 0.03443615f, -0.01257316f, -0.03022167f, 0.00812424f, 0.00398035f, 0.00664177f, -0.01005086f, 0.02431160f, 0.04778986f, 0.03424769f, 0.00469184f, -0.01432405f, 0.00789674f, 0.05877604f, -0.03384549f, -0.05043270f, 0.02700877f, 0.02727220f, 0.06369454f, -0.00615123f, -0.01657593f, 0.04869708f, 0.06414002f, -0.01213374f, -0.00939974f, 0.00642731f, 0.01811421f, 0.06772272f, -0.00894993f, -0.01147916f, 0.08711022f, 0.01192173f, -0.00506679f, -0.06000935f, -0.01270292f, 0.05959301f, 0.02188842f, -0.02527555f, -0.06378698f, -0.00215907f, 0.10798866f, 0.04382401f, 0.02836247f, 0.03390486f, 0.04579261f, -0.02285426f, -0.03307016f, -0.05415248f, 0.05109280f, 0.04122447f, -0.06364054f, -0.05784198f, -0.00436295f, 0.05061163f, 0.03850084f, -0.03943404f, -0.06381033f, 0.04472810f, 0.11282113f, 0.08465373f, 0.11759328f, 0.13381761f, 0.10297585f, 0.08109049f, 0.06224539f, -0.00028884f, 0.05554008f, 0.01991579f, -0.02085821f, -0.00026629f, -0.04676922f, 0.02762052f, -0.06291547f, -0.06798761f, -0.06110318f, 0.00183294f, -0.00382462f, -0.04701187f, -0.00526891f, -0.00652003f, 0.01361585f, 0.03521463f, -0.01307323f, 0.00113648f, 0.03379195f, 0.09735926f, 0.14702524f, 0.15747252f, 0.08775581f, 0.09128944f, 0.04084205f, 0.00968436f, -0.01024935f, 0.06862513f, 0.12410378f, 0.08567179f, 0.09193633f, 0.04823282f, 0.10039144f, 0.11034909f, 0.07623604f, 0.12453194f, 0.19420144f, -0.07739237f, -0.07032817f, 0.03815327f, 0.04626439f, -0.03326904f, -0.04420659f, -0.05080248f, -0.03708746f, -0.04528029f, -0.00751468f, 0.09323639f, 0.16302247f, 0.08032542f, -0.06528563f, -0.09110985f, -0.04498733f, -0.04962393f, -0.07161536f, 0.02423094f, 0.03496175f, 0.11167847f, 0.09218782f, 0.05219970f, 0.03097210f, 0.03086267f, 0.09996531f, 0.12085772f, -0.00578561f, 0.08654227f, 0.18402892f, 0.09991333f, 0.02580757f, -0.02900945f, 0.00562203f, 0.04306385f, 0.00593101f, 0.08929474f, 0.21356974f, 0.16966728f, 0.04589879f, 0.01956859f, -0.00494311f, 0.00875978f, 0.00533760f, 0.00290998f, 0.04722815f, 0.12108309f, 0.03486136f, -0.03873474f, -0.00698803f, -0.00400079f, 0.00301954f, 0.00352612f, 0.00732205f, 0.01939473f, 0.14094504f, 0.16729519f, 0.05609642f, -0.00353496f, -0.01849359f, 0.03565866f, 0.02498527f, -0.04804130f, -0.02613563f, 0.00943842f, -0.04578623f, -0.09753565f, -0.07318561f, -0.08399214f, -0.03532942f, -0.00668484f, -0.03633925f, 0.05887394f, 0.11917251f, 0.03511264f, 0.01970124f, 0.04771805f, 0.03402291f, 0.02710548f, 0.06007919f, 0.04802724f, 0.02279612f, 0.09487083f, 0.05143212f, 0.03037265f, 0.05985160f, 0.03947429f, 0.04781054f, 0.06996727f, 0.04305828f, 0.01737752f, -0.01449862f, -0.06090984f, -0.06848649f, -0.03503193f, -0.06139164f, -0.02716843f, 0.02565364f, 0.03943714f, 0.03359919f, 0.00226491f, -0.01879661f, -0.04493717f, -0.01891606f, -0.05577957f, -0.02504424f, 0.01092236f, 0.04221837f, 0.10778112f, 0.11982676f, 0.04168952f, -0.00237081f, 0.07086146f, 0.00977770f, 0.01009277f, 0.04470597f, 0.04357202f, 0.10396231f, 0.12788617f, 0.01670403f, -0.03068930f, 0.04230604f, 0.01392035f, -0.01169356f, 0.01631905f, 0.06638083f, 0.07690117f, 0.04147712f, -0.00284332f, -0.02349386f, -0.00439454f, -0.04517610f, -0.03192258f, -0.00835287f, 0.02646200f, 0.03145197f, 0.03085419f, -0.04707114f, -0.04745444f, 0.03917791f, -0.04480892f, -0.04866575f, -0.02282306f, -0.00988333f, 0.10774999f, 0.14390158f, 0.05789705f, 0.02582101f, 0.07878795f, 0.08155783f, 0.06496570f, 0.07565701f, 0.08208815f, 0.10113062f, 0.10890649f, 0.04069236f, -0.00657736f, 0.02304602f, 0.04481221f, 0.08257412f, 0.06497147f, 0.05593672f, 0.03616930f, 0.05038059f, -0.00637183f, 0.00224736f, 0.11565826f, 0.06833739f, 0.02180515f, 0.01927962f, 0.01161375f, 0.01243792f, 0.02016885f, -0.00134421f, -0.02951555f, 0.03452681f, -0.00409717f, -0.00082992f, 0.00289916f, -0.02452033f, 0.07893640f, 0.13467961f, 0.07942202f, 0.01885271f, 0.04090110f, 0.05544413f, 0.09426625f, 0.11389374f, 0.09188815f, 0.02335605f, 0.10687354f, 0.04897968f, -0.00580858f, 0.02477198f, 0.02061114f, 0.02334246f, 0.05521480f, 0.04299230f, 0.01512745f, 0.05413653f, 0.04330312f, -0.02571854f, -0.04380923f, -0.04714603f, 0.02905893f, 0.08544393f, 0.05209743f, 0.01787720f, -0.00856393f, -0.01271492f, -0.06697786f, -0.07640814f, -0.06643784f, -0.01363567f, 0.01371672f, 0.02173055f, 0.01368154f, 0.12323183f, 0.03944393f, -0.01481670f, 0.03527353f, 0.00287916f, 0.00861270f, 0.04213872f, 0.03882627f, 0.02359116f, 0.11900191f, 0.02830939f, -0.00543872f, -0.00559199f, -0.02961755f, 0.02689826f, 0.02882352f, 0.01677009f, 0.05381129f, 0.05405762f, -0.01515895f, -0.06543188f, -0.05402671f, -0.05537873f, -0.03977801f, -0.00230561f, 0.02058043f, 0.09199040f, 0.09063583f, 0.02243174f, -0.01163034f, -0.02332827f, -0.05598468f, 0.03125977f, 0.05348528f, 0.04112348f, 0.00810345f, 0.05972141f, 0.02823945f, -0.03045728f, -0.00570164f, -0.04395656f, -0.00563589f, -0.00979688f, -0.03251627f, -0.02450656f, 0.05005366f, 0.03388465f, -0.03265503f, -0.04027252f, -0.04636829f, -0.03254069f, -0.03367457f, -0.02815954f, 0.05068323f, 0.09280934f, 0.01128072f, -0.04224689f, -0.05156360f, -0.05116552f, 0.00838934f, -0.00873587f, 0.00786036f, 0.01017203f, 0.06616134f, 0.02538756f, -0.03738370f, -0.04451807f, -0.05887061f, -0.01033517f, -0.01619062f, -0.00296818f, -0.02529397f, 0.07350121f, 0.04132021f, -0.02443028f, -0.04032023f, -0.06892137f, -0.03056222f, -0.03174124f, -0.04614157f, 0.02730662f, 0.09929461f, 0.00452469f, -0.08320029f, -0.03166123f, -0.07812914f, -0.05235115f, -0.02596913f, -0.04365650f, 0.02676525f, 0.08355814f, 0.05044545f, -0.02914317f, -0.01235774f, -0.05141609f, -0.01476055f, 0.05296891f, 0.01684485f, 0.08963228f, 0.11198198f, 0.06803401f, 0.00216193f, 0.00645064f, -0.06132017f, -0.04689707f, 0.05723606f, 0.03998201f, 0.07723108f, 0.09450253f, 0.06000432f, 0.00870161f, -0.01058188f, -0.02646866f, -0.01398591f, 0.00630969f, -0.03517729f, 0.02231866f, 0.02032942f, 0.00545634f, -0.04022641f, 0.03163884f, 0.03208057f, 0.04186882f, 0.01057378f, -0.02841343f, 0.09818542f, 0.15418981f, 0.16829987f, 0.12512379f, 0.06032014f, 0.02195900f, 0.00806573f, 0.05841811f, 0.00179797f, 0.05445491f, 0.07933785f, 0.09997354f, 0.06533882f, 0.05273508f, 0.10331168f, 0.05162466f, 0.04234636f, -0.00899526f, 0.00963667f, 0.03356415f, 0.05364791f, 0.06930675f, 0.14255636f, 0.14409402f, 0.06878222f, 0.04621433f, -0.00881069f, -0.07202354f, -0.05116246f, 0.00483115f, -0.03618808f, 0.03364937f, 0.02685122f, -0.03969252f, -0.02311021f, -0.05670812f, 0.03192183f, 0.06190820f, 0.10189351f, 0.14406833f, 0.10466955f, 0.19655377f, 0.05500998f, 0.05411794f, 0.01156997f, -0.07157540f, -0.02660463f, 0.02214019f, -0.03070518f, 0.00485122f, -0.03956399f, -0.08059239f, -0.07722651f, -0.09429585f, -0.01766637f, -0.01154249f, 0.02750854f, -0.01326937f, 0.00113218f, 0.06553180f, 0.00393976f, 0.02623336f, 0.00597417f, -0.04755523f, -0.02237610f, 0.04419893f, 0.03354407f, 0.01175933f, 0.03192454f, 0.00634640f, -0.00824123f, -0.04285603f, -0.02740165f, 0.00918497f, 0.02603822f, -0.03213990f, -0.00089545f, 0.04233510f, 0.00415884f, -0.02405177f, -0.03280947f, -0.03061858f, 0.02655393f, 0.04902287f, 0.02466987f, -0.00259454f, 0.02778154f, -0.00163914f, -0.03353026f, -0.06232664f, 0.04406732f, -0.03235006f, -0.02570207f, 0.01486327f, 0.04770024f, 0.04935245f, -0.00358207f, -0.04466425f, 0.02992461f, 0.10436859f, -0.02622942f, 0.01256166f, -0.00531495f, 0.04785787f, 0.05685732f, -0.03654406f, -0.04211138f, 0.07046730f, 0.04147415f, -0.04862929f, -0.02767102f, 0.05852158f, 0.06458365f, 0.09189710f, -0.00327522f, -0.03754646f, 0.03715980f, 0.05321734f, -0.03326589f, -0.05190075f, -0.00327473f, 0.01688486f, 0.06316090f, 0.01225673f, -0.05170825f, 0.04691209f, 0.01664282f, -0.10916956f, -0.08496947f, -0.07223777f, -0.00353273f, -0.02542571f, -0.10751837f, -0.12415369f, -0.01957306f, 0.04357196f, 0.04488667f, 0.09840552f, 0.16922418f, 0.16229810f, 0.10671634f, 0.08156298f, 0.00881030f, 0.02585973f, 0.00150834f, -0.07570329f, -0.09268396f, -0.06670630f, -0.02334309f, -0.05282683f, -0.09443780f, -0.14696262f, -0.03170263f, 0.04093346f, 0.03103766f, 0.07360602f, 0.11574830f, 0.16276659f, 0.11282683f, 0.12240920f, 0.06483546f, 0.07270351f, -0.03636717f, 0.05120038f, 0.09756405f, 0.08331167f, 0.10365290f, 0.06217380f, 0.06589046f, 0.02568124f, -0.04777792f, 0.09489305f, 0.05857754f, 0.05177006f, -0.00911147f, -0.01525243f, 0.03399069f, 0.07756129f, 0.09399364f, 0.08949814f, -0.04351723f, 0.00898100f, 0.05815406f, 0.06050185f, 0.11514639f, 0.06259922f, 0.12191516f, 0.08582617f, -0.01369801f, 0.11175632f, 0.09791294f, 0.06792039f, 0.00571331f, -0.01731702f, -0.02434690f, 0.05133475f, 0.05749117f, 0.07662351f, 0.02395619f, 0.03726464f, 0.07628133f, 0.07012703f, 0.04305603f, 0.07320541f, 0.08831878f, 0.10489033f, 0.06363916f, -0.07629858f, 0.03968855f, 0.08428438f, 0.06333820f, -0.01459654f, 0.03676257f, 0.09327082f, 0.06501482f, -0.04670194f, 0.06432380f, 0.09218840f, 0.06154499f, 0.03272294f, 0.02129131f, 0.05675820f, 0.06187550f, 0.02504533f, -0.01070930f, -0.09042705f, 0.00819468f, 0.00005657f, -0.05019764f, -0.01610900f, 0.07718444f, 0.10190575f, 0.04057618f, -0.06876002f, 0.01432326f, 0.12439366f, 0.15007235f, 0.09625975f, 0.00360732f, 0.05814260f, 0.14534782f, 0.12286223f, -0.03176561f, -0.05547193f, 0.02068044f, -0.01724304f, -0.04993372f, -0.08376755f, -0.04029682f, 0.00985242f, 0.01782241f, -0.04611649f, -0.04534993f, 0.06639481f, 0.10771843f, 0.02341295f, 0.01949288f, 0.12192010f, 0.18498690f, 0.13731800f, -0.00072032f, -0.06116600f, 0.00762252f, -0.00666006f, -0.03439742f, -0.08171532f, -0.03563251f, 0.02124676f, 0.02113266f, -0.05552494f, 0.00416983f, 0.01663505f, -0.00855902f, -0.02852404f, 0.00660492f, -0.00612183f, 0.03250734f, 0.06294809f, 0.02591045f, -0.00101507f, 0.03065868f, 0.00337484f, -0.05957970f, -0.03964982f, -0.06225329f, 0.00978309f, 0.02754637f, 0.02752263f, 0.01940933f, 0.07794208f, -0.00814499f, -0.01883455f, -0.00960418f, -0.01902483f, 0.02164742f, 0.04388450f, 0.04348737f, 0.01455219f, 0.04852569f, -0.01463443f, -0.08269602f, -0.05137614f, -0.07558227f, 0.00457351f, 0.05511728f, 0.05423633f, -0.01325603f, 0.04312986f, 0.00932003f, -0.02413631f, 0.00149580f, -0.03059079f, 0.00428567f, 0.03034671f, 0.01464907f, -0.07378244f, 0.01796597f, -0.03266862f, 0.00081484f, 0.08633481f, 0.03674961f, -0.01533694f, 0.01015025f, -0.04456920f, 0.00400109f, 0.02686524f, -0.00482310f, -0.02838754f, 0.01194831f, -0.02743774f, 0.00652743f, 0.04516566f, 0.00555174f, -0.06428664f, -0.00844282f, -0.02296902f, 0.01938154f, 0.07355654f, 0.00472665f, -0.03808882f, 0.00880279f, -0.04312545f, -0.05061456f, 0.04871011f, 0.03710382f, 0.10459260f, 0.21259237f, 0.16516612f, 0.05686626f, 0.01840283f, -0.04037043f, -0.06978445f, 0.02262874f, -0.00702803f, 0.03291830f, 0.10896182f, 0.04841346f, -0.00621038f, 0.00243723f, -0.07592908f, -0.04006696f, 0.02858687f, 0.04798275f, 0.09429554f, 0.18584777f, 0.11659620f, 0.01892630f, -0.00403749f, -0.05774479f, -0.04385132f, 0.00012619f, -0.00104521f, -0.00273609f, 0.12395631f, 0.05161817f, -0.01898942f, 0.01326377f, -0.04422498f, 0.00992639f, 0.06030647f, 0.04211000f, 0.03528413f, 0.03887335f, 0.00220248f, 0.02797421f, 0.11595430f, 0.01490079f, 0.05457438f, 0.05271633f, 0.03347463f, -0.01632934f, -0.04206928f, -0.03923973f, 0.01720631f, 0.07144948f, 0.03746372f, 0.05007459f, 0.11894894f, 0.03835204f, 0.01353120f, 0.06927122f, 0.02616358f, 0.05919249f, 0.09014434f, -0.00379607f, 0.04833562f, 0.12380839f, 0.03785451f, -0.01256381f, -0.01841532f, -0.04704299f, 0.03449249f, 0.05299695f, 0.05545035f, 0.05673669f, 0.03994976f, -0.00551109f, -0.04308027f, -0.04070375f, -0.03756178f, -0.00947555f, 0.02748857f, 0.05606917f, 0.06353972f, 0.10667088f, 0.06453649f, 0.00353010f, 0.01247374f, -0.01244471f, 0.05809919f, 0.08617160f, 0.06939770f, 0.06761500f, 0.06254488f, -0.02208818f, -0.04932000f, -0.04313945f, -0.05653498f, -0.00102358f, 0.03197810f, 0.08906870f, 0.06413943f, 0.10728610f, 0.06622158f, -0.00916728f, -0.00993257f, 0.00552479f, 0.07086878f, 0.10801621f, 0.08021523f, 0.01680194f, 0.10453911f, 0.03338826f, -0.03763711f, -0.07003647f, -0.03586691f, 0.02048407f, 0.01620958f, 0.00412122f, 0.00836738f, 0.09008353f, 0.02844230f, -0.02729444f, -0.03137145f, -0.04064164f, -0.00122414f, -0.01311081f, -0.01746561f, 0.00948226f, 0.06624891f, 0.02929843f, -0.05107199f, -0.06771838f, -0.01363042f, 0.03863236f, 0.08301864f, 0.04075867f, 0.01072984f, 0.04698297f, -0.01810272f, -0.07147558f, -0.03288006f, -0.00659891f, 0.04152945f, 0.06422785f, -0.00803192f, 0.03443290f, 0.07478459f, 0.01476454f, -0.03400133f, 0.00773658f, -0.04435433f, -0.03733272f, 0.02418010f, 0.01398767f, 0.06190493f, 0.07638157f, 0.01854121f, -0.04540797f, -0.02011402f, -0.03620786f, -0.02771283f, 0.06047135f, 0.04044989f, -0.00849552f, 0.02352856f, -0.05152931f, -0.06191465f, 0.00985169f, -0.02935451f, -0.01448455f, 0.05381508f, 0.00235255f, 0.02160162f, 0.05030786f, -0.04910153f, -0.05501445f, -0.01486054f, -0.03140164f, 0.01940739f, 0.09004320f, 0.06399587f, 0.05945688f, 0.10755971f, 0.10575369f, 0.05609033f, 0.05640474f, 0.03540174f, 0.02788228f, 0.06595109f, 0.00049228f, 0.01787594f, 0.01005389f, 0.02706068f, 0.01464427f, 0.04856926f, 0.05929133f, 0.04186769f, 0.02467939f, -0.00387438f, 0.00409357f, 0.04840457f, 0.02890640f, 0.01781596f, 0.04093534f, 0.08563755f, 0.10235348f, 0.11583525f, 0.05834925f, -0.01664843f, -0.00318004f, 0.03819735f, 0.03105332f, 0.03060931f, 0.03894599f, 0.00895227f, 0.00110011f, -0.00877323f, 0.03524015f, 0.01929016f, 0.07499408f, 0.14656073f, 0.14056875f, 0.15220113f, 0.06913990f, 0.04061252f, 0.02783580f, -0.09016433f, -0.07186056f, -0.02485273f, -0.07529513f, 0.05051856f, -0.05368253f, -0.06256274f, -0.08154239f, -0.09394524f, 0.02510623f, 0.03595429f, 0.11095440f, 0.18710712f, 0.12797122f, 0.15107032f, 0.02279114f, 0.00606861f, 0.02123511f, -0.11579403f, -0.10089463f, -0.06303992f, -0.07258220f, 0.02850608f, -0.08682448f, -0.07860538f, -0.08544735f, -0.11703883f, -0.02886173f, -0.00323293f, 0.02236884f, -0.00290943f, 0.06386214f, 0.02282310f, -0.00093058f, -0.03382245f, -0.03684840f, -0.06388099f, 0.00915902f, 0.04206568f, 0.01026608f, 0.01639525f, 0.03519815f, 0.00995563f, -0.04719955f, -0.07004933f, -0.04628178f, -0.01861522f, 0.00073649f, 0.02220411f, 0.06563437f, -0.01753162f, -0.03481225f, -0.01752510f, -0.03525945f, -0.08694344f, -0.03176261f, 0.00217032f, 0.01070187f, -0.01998636f, -0.02880801f, -0.01488969f, -0.02021388f, -0.07630657f, 0.05035524f, -0.03375863f, -0.00945288f, 0.04357570f, 0.01977461f, 0.04790861f, 0.00442404f, -0.00726297f, 0.04980541f, 0.10193414f, -0.01790230f, -0.02857782f, 0.00773310f, 0.05878996f, 0.04649506f, 0.00280679f, -0.01674613f, 0.02705150f, -0.01234178f, -0.02942676f, -0.02955891f, -0.00470174f, 0.01894186f, 0.06284760f, 0.00260991f, -0.01094607f, 0.00902814f, 0.01346351f, -0.01221925f, -0.01309742f, 0.00596166f, 0.03114615f, 0.06445997f, 0.00360138f, 0.02927897f, -0.00177791f, 0.06440394f, -0.05016073f, -0.05576530f, -0.02342031f, 0.05490494f, 0.02695617f, -0.01716610f, -0.04598488f, 0.01589427f, 0.07816962f, -0.03104341f, 0.01709717f, 0.06041404f, 0.08703089f, 0.11600252f, 0.12447668f, 0.08202502f, 0.09096028f, 0.03507766f, -0.06742048f, -0.05567162f, -0.00813275f, 0.03118060f, 0.03457983f, -0.03630587f, 0.00175412f, -0.01232715f, 0.05820030f, -0.05143072f, -0.05149064f, -0.01241935f, 0.01134693f, 0.05748134f, 0.03438174f, 0.01712768f, 0.04861096f, 0.06094474f, -0.02051703f, -0.00547421f, 0.01672493f, 0.08728299f, 0.07230956f, 0.14315229f, 0.14494743f, 0.09535486f, 0.17161071f, 0.13298450f, 0.07340580f, 0.06875427f, 0.08515487f, 0.03456015f, 0.07170303f, 0.08333233f, 0.14292335f, 0.01253682f, -0.08779790f, -0.08142612f, -0.03445361f, 0.05638599f, -0.00079888f, -0.00304658f, -0.01439300f, 0.02107921f, 0.05823994f, -0.03049938f, -0.03763191f, 0.03111196f, 0.03174811f, 0.03164924f, 0.00694550f, -0.01100827f, 0.04498068f, 0.11714933f, 0.09843507f, 0.02273647f, 0.04699400f, 0.05460724f, 0.08989334f, 0.10526445f, 0.04302370f, 0.06721442f, -0.00904132f, 0.03744108f, -0.00935757f, -0.03612002f, 0.04976771f, 0.15749406f, 0.21959518f, 0.11536847f, 0.02231060f, -0.01603033f, -0.04418570f, -0.05608305f, -0.02692722f, -0.01774709f, 0.02264049f, 0.03796296f, -0.02475626f, -0.02222871f, -0.02722463f, -0.05839970f, -0.05629733f, -0.05969364f, -0.02948549f, 0.10599960f, 0.16815283f, 0.14001150f, -0.00012630f, -0.04421182f, -0.01002540f, -0.00136852f, -0.04852171f, -0.02051162f, 0.08673927f, 0.17305234f, 0.15772800f, 0.02551902f, -0.03324610f, 0.00715945f, -0.03122238f, -0.07368951f, -0.10125770f, -0.06557858f, -0.00926538f, 0.03465261f, -0.00621253f, 0.02985158f, -0.01640721f, -0.01806155f, -0.01008639f, 0.01657785f, 0.11034003f, 0.17219786f, 0.22021889f, 0.10784917f, -0.00469297f, 0.01119645f, 0.01970841f, -0.00055292f, -0.00821511f, 0.01202488f, 0.06473410f, 0.12895488f, 0.05349378f, 0.01333230f, 0.03106872f, -0.02749675f, -0.04444170f, -0.05855188f, -0.06399076f, -0.02806905f, 0.00907902f, 0.01008963f, 0.04874921f, 0.01381811f, -0.05092364f, -0.06139919f, -0.03327654f, -0.05625738f, -0.03508002f, 0.02765319f, 0.04636516f, 0.07229569f, 0.07757282f, 0.05007491f, 0.04081300f, 0.03090716f, 0.01119824f, 0.04272068f, 0.12712563f, 0.09228944f, 0.07038593f, 0.08885931f, 0.03282142f, 0.02823482f, 0.05907503f, 0.01360872f, 0.02798159f, 0.08661647f, 0.04866919f, 0.00539952f, -0.01055831f, -0.05630352f, -0.02180248f, 0.02847690f, 0.00290656f, -0.02251076f, 0.05044009f, 0.03087916f, -0.01948754f, -0.03133969f, -0.08039800f, -0.01233478f, 0.03657931f, -0.03369846f, -0.05414160f, 0.02605504f, 0.03330262f, 0.05707821f, 0.06872384f, -0.02314039f, 0.01908854f, 0.09354928f, 0.04175201f, 0.03622885f, 0.12148409f, 0.07259596f, 0.05224030f, 0.02019172f, -0.02845541f, 0.03791345f, 0.07793228f, -0.00495926f, 0.01824082f, 0.10472871f, 0.08047205f, 0.00619418f, -0.00571357f, -0.01414253f, 0.03239704f, 0.07859010f, 0.02353015f, -0.01063665f, 0.04032790f, 0.03705401f, 0.03057634f, 0.00663529f, -0.03161947f, -0.06420478f, 0.00463103f, -0.04012336f, -0.03776760f, 0.02164264f, 0.05138955f, 0.09975899f, 0.05972116f, 0.04196614f, 0.07041353f, 0.08319245f, 0.06048293f, 0.07053352f, 0.13991996f, 0.13018220f, 0.11168552f, 0.06098702f, 0.06767290f, 0.03457330f, 0.02006337f, 0.00426310f, 0.06877855f, 0.11464842f, 0.09716248f, 0.03289990f, 0.07273298f, 0.01384424f, -0.04733631f, -0.05311680f, -0.06247411f, -0.00563570f, 0.04375050f, 0.01844996f, -0.03199433f, 0.03517090f, -0.03393241f, -0.04523511f, -0.03580920f, -0.09847873f, -0.03944333f, -0.04634899f, 0.00948363f, 0.11838052f, 0.08408561f, 0.07026835f, 0.06164432f, 0.03502299f, -0.01099100f, 0.08506758f, 0.13687943f, 0.10924995f, -0.00534313f, 0.04015687f, 0.01723742f, 0.05518818f, 0.04423920f, 0.01926496f, 0.05225484f, 0.07056526f, 0.02362442f, 0.00788536f, 0.03857690f, -0.06919457f, -0.06614482f, -0.04418750f, -0.09745897f, -0.03565117f, 0.02973182f, 0.06299650f, 0.02087593f, 0.04722182f, -0.01499164f, -0.06634402f, -0.02327127f, -0.06822773f, -0.01214228f, 0.06548406f, 0.05811495f, 0.01019760f, 0.05787808f, -0.01161746f, 0.02162851f, 0.05504127f, -0.00841942f, 0.00244081f, 0.09515335f, 0.03927907f, 0.01466648f, 0.02900550f, 0.01222623f, -0.01738617f, 0.02934737f, -0.02816646f, -0.01670235f, 0.05728840f, 0.03318266f, 0.00562568f, 0.03589498f, 0.00910532f, -0.03418965f, -0.02276698f, -0.04113830f, 0.01712258f, 0.08239755f, 0.02033529f, 0.00487530f, 0.02894953f, -0.03199302f, -0.06788903f, -0.02655786f, -0.00326373f, 0.05356836f, 0.11608828f, 0.00039241f, -0.03485726f, 0.00024253f, -0.00007924f, -0.03373309f, 0.00719833f, -0.04285954f, 0.00399491f, 0.03558076f, 0.00008117f, -0.02234532f, 0.01257380f, -0.05936953f, -0.06662379f, -0.02702709f, -0.02091710f, 0.03745216f, 0.06053908f, -0.02150538f, 0.01591954f, 0.07424237f, -0.01579655f, -0.05527902f, -0.00502863f, -0.00103155f, 0.04447044f, 0.12823912f, 0.01153186f, 0.03301948f, 0.06395266f, -0.03742392f, -0.04751715f, -0.01626798f, -0.01327775f, 0.04036925f, 0.15462637f, 0.09468430f, 0.00408405f, -0.00645355f, -0.07328470f, -0.09232280f, -0.03056419f, 0.01461383f, 0.04502214f, 0.09547142f, -0.02038681f, -0.02782831f, -0.03770049f, -0.06000668f, -0.01930557f, -0.02157656f, -0.02295284f, -0.01456595f, 0.08073894f, 0.00915940f, 0.04500522f, 0.05982021f, 0.01927795f, 0.04598815f, 0.05256517f, 0.12536565f, 0.17104222f, 0.20004830f, 0.12629589f, -0.01942723f, 0.03452773f, 0.05154300f, 0.07578918f, 0.04208466f, 0.06462251f, 0.10723652f, 0.10794933f, 0.04196192f, -0.05140986f, -0.00727579f, 0.01535666f, 0.04049721f, -0.01224699f, 0.03822108f, 0.06787455f, 0.09375329f, 0.04747551f, -0.02864795f, 0.01241916f, 0.03639593f, -0.00686160f, -0.02291553f, -0.05006283f, -0.00430713f, 0.03660318f, -0.00240884f, 0.04657570f, 0.08400640f, 0.07861945f, 0.14908456f, 0.11796017f, 0.11461946f, 0.10010253f, 0.07059533f, 0.06036563f, -0.05215824f, -0.04404761f, -0.04977129f, -0.07621128f, 0.00760547f, -0.03628982f, 0.00265469f, -0.02351526f, -0.04222752f, 0.02716092f, 0.08459901f, 0.08743033f, 0.08962674f, 0.07767694f, 0.07947683f, 0.04065634f, 0.02887897f, 0.02581510f, -0.04705546f, -0.03807860f, -0.04848518f, -0.04510013f, 0.02208879f, -0.03693392f, -0.01712791f, -0.05239125f, -0.04903450f, -0.01183810f, 0.01332648f, 0.02643973f, 0.00749597f, 0.03259270f, -0.03182332f, 0.00992202f, 0.01652397f, -0.00807491f, -0.03611385f, -0.01392696f, 0.01196057f, 0.01192175f, -0.07502472f, -0.01515977f, 0.00859387f, 0.00678212f, -0.02965466f, 0.00805367f, 0.03284098f, 0.00357543f, 0.00511591f, 0.04642539f, 0.01335906f, 0.01956807f, -0.01819744f, -0.03016259f, -0.00536399f, 0.01324922f, 0.01057991f, -0.01975175f, -0.02189755f, 0.05683979f, 0.06249428f, -0.00869579f, -0.03937346f, 0.05823261f, -0.03760637f, -0.00249481f, -0.03207283f, 0.02962863f, 0.06579671f, -0.01574932f, -0.02596675f, 0.02452602f, 0.05583577f, -0.01608392f, -0.00451214f, -0.02727193f, 0.05297499f, 0.05678963f, -0.01850835f, -0.00862089f, 0.02545626f, 0.04840067f, -0.00718734f, -0.00948675f, 0.03100146f, 0.02020636f, 0.04415858f, -0.04542315f, -0.00588458f, 0.01684640f, 0.02968280f, 0.03118289f, -0.00060581f, 0.02939589f, 0.00856396f, 0.06790884f, -0.04062854f, 0.00908660f, 0.03236137f, 0.06160330f, -0.02522266f, -0.01078952f, -0.02165897f, 0.01768494f, 0.00649432f, -0.04099318f, -0.01540302f, -0.01293830f, 0.05446461f, -0.00334961f, 0.00221246f, -0.00306624f, -0.00962116f, 0.02730571f, -0.00197791f, -0.02452815f, 0.03103301f, 0.05501927f, -0.00797242f, -0.02749486f, 0.00895752f, -0.00264785f, 0.04911258f, -0.03946627f, -0.01545661f, 0.03706520f, 0.04934560f, -0.02727608f, 0.00354607f, 0.02285564f, -0.01444103f, 0.03241967f, -0.02444916f, -0.01853317f, 0.06571836f, 0.06953436f, -0.01782526f, -0.01179675f, -0.00543244f, 0.05683804f, 0.01125038f, -0.00117559f, -0.01139441f, 0.05245981f, 0.08634598f, 0.00151724f, -0.01044376f, 0.03580709f, 0.02994892f, 0.04616996f, 0.01307016f, -0.00027343f, 0.05085357f, 0.02357900f, -0.02541789f, -0.01313619f, 0.02030613f, 0.04351678f, 0.02294291f, -0.02510332f, -0.01252423f, 0.00231169f, 0.08052724f, 0.01442871f, 0.02364987f, 0.01723319f, 0.00657401f, 0.03795626f, -0.00903154f, -0.02468148f, 0.03411500f, 0.05245522f, -0.02062435f, -0.00668229f, -0.03426522f, 0.01357177f, 0.02778693f, 0.01059957f, -0.05200283f, 0.01667525f, 0.04834663f, -0.02854114f, 0.00412988f, 0.01356912f, 0.02696671f, 0.06587073f, 0.07264230f, 0.08032317f, 0.07213508f, 0.05576490f, -0.03400527f, -0.02829336f, -0.04373093f, 0.03417938f, -0.02594596f, -0.03096772f, -0.07811092f, 0.01631913f, 0.03964983f, -0.01875781f, -0.02099333f, 0.00684724f, 0.01850494f, -0.01279190f, -0.00806193f, -0.01484296f, 0.01483517f, 0.00685591f, -0.04232691f, -0.04008456f, -0.02178351f, 0.01492841f, -0.00298162f, 0.04305293f, 0.10122582f, 0.05373550f, -0.03568370f, -0.01598641f, -0.02818564f, -0.01367014f, 0.03286201f, -0.03153314f, 0.01116592f, 0.13432735f, 0.06872091f, 0.02157683f, -0.00698929f, -0.04382117f, -0.03334776f, 0.02069707f, -0.04995008f, -0.02183115f, -0.02316111f, 0.01241026f, 0.02610712f, 0.02719125f, 0.00026639f, 0.04162894f, 0.03524503f, -0.01887861f, -0.01615216f, 0.01134322f, 0.01099854f, -0.00020613f, 0.01078558f, 0.00478231f, 0.01793033f, 0.05177408f, -0.02744724f, -0.00922947f, 0.11410876f, 0.07123929f, -0.00294827f, 0.03792190f, -0.02269942f, 0.01424666f, 0.04930967f, -0.01410185f, -0.00996062f, 0.04506830f, 0.00023641f, 0.04106047f, 0.02440421f, -0.00945793f, 0.03290221f, 0.04596138f, 0.01200415f, -0.00881161f, 0.03618791f, 0.01162407f, -0.02357907f, 0.03973287f, -0.05070475f, -0.00095780f, 0.01967817f, -0.00788112f, -0.02252490f, -0.00387697f, -0.05247399f, 0.02566480f, 0.02149697f, -0.04278180f, -0.00562731f, 0.08613562f, 0.03030757f, 0.03277295f, 0.05941161f, 0.02399220f, 0.03073975f, -0.00137154f, -0.07529084f, -0.04245184f, 0.02639078f, -0.05510592f, 0.02256968f, 0.07374550f, 0.05330564f, -0.00974155f, 0.00510023f, -0.08164987f, 0.02001205f, 0.04641761f, 0.02364675f, 0.02090996f, 0.02224614f, -0.04815739f, 0.00768631f, 0.00336581f, -0.06962127f, -0.02244921f, 0.01743253f, -0.00739071f, 0.00793969f, 0.04038312f, -0.00955908f, 0.08558581f, 0.05664980f, -0.00391376f, 0.01204981f, 0.03208879f, 0.02230554f, 0.05695049f, 0.09189817f, 0.07757331f, 0.04557866f, 0.01085430f, 0.00796619f, -0.00821078f, 0.01619350f, 0.02622035f, 0.04508241f, 0.04607864f, 0.02350019f, 0.03707235f, 0.04071007f, 0.00066443f, -0.01234382f, 0.01586661f, 0.02806045f, 0.04172990f, 0.04026979f, 0.01974485f, -0.03342275f, 0.01697428f, -0.01391837f, -0.03563229f, -0.02673430f, -0.00127530f, -0.00409038f, -0.02033447f, -0.06172148f, 0.03729594f, 0.03297069f, 0.02160916f, 0.03517433f, 0.07624896f, 0.03491277f, 0.08331205f, 0.08202682f, 0.02836405f, -0.03774613f, -0.01289721f, -0.03223725f, 0.00788759f, 0.10044965f, 0.01783835f, 0.00821172f, 0.02799601f, -0.01523708f, 0.01245349f, 0.05379189f, -0.00646332f, -0.00197591f, 0.03704947f, 0.01468394f, -0.02258818f, -0.01242818f, -0.02210738f, -0.01028609f, 0.03478316f, -0.04082056f, -0.02030331f, 0.01625447f, -0.00905890f, -0.04869362f, -0.04374754f, -0.04405381f, -0.04646268f, 0.02129956f, -0.00743081f, 0.04882723f, 0.11813779f, 0.02291917f, -0.00055906f, 0.08403242f, 0.01438065f, -0.02922087f, 0.01140340f, -0.01987780f, -0.03561805f, 0.09659570f, -0.00935558f, 0.02786991f, 0.06646252f, -0.00432101f, -0.00334204f, 0.01242500f, -0.02600162f, 0.02128612f, 0.03436245f, 0.00666056f, -0.02616754f, -0.00479148f, -0.02109305f, -0.03996861f, -0.04092338f, -0.05895368f, -0.02874729f, 0.06379664f, -0.02488858f, -0.01474263f, 0.01308497f, -0.03612628f, -0.00386590f, 0.02352955f, -0.00604491f, 0.00753074f, 0.11412999f, 0.02691511f, 0.04434998f, 0.06786779f, 0.00314222f, 0.00112500f, -0.01110519f, -0.06987393f, -0.05767714f, 0.05021254f, 0.01943841f, 0.05191694f, 0.06523419f, -0.01621864f, 0.01090056f, 0.00109568f, -0.06969674f, -0.01434532f, 0.07083049f, 0.00413273f, -0.02287742f, -0.00099027f, -0.01577105f, -0.02503839f, -0.04915646f, -0.09140889f, -0.05653142f, -0.01236419f, -0.02785027f, -0.01765751f, -0.00889133f, -0.01420238f, 0.02192227f, -0.01577913f, -0.03175435f, 0.00099893f, 0.04863589f, 0.05301309f, 0.03934159f, 0.07968278f, 0.02261282f, -0.02721842f, -0.01649118f, -0.01530369f, 0.01586952f, 0.02430798f, 0.00365790f, -0.01818779f, 0.06085609f, 0.03451937f, -0.03570643f, -0.05578092f, -0.04649417f, -0.00857880f, -0.00485314f, -0.00967480f, -0.01115386f, 0.02783757f, -0.00564101f, -0.04600395f, -0.06345836f, -0.05407679f, -0.05044785f, -0.03181873f, -0.01460280f, -0.07421742f, -0.01640853f, -0.01927266f, 0.00623902f, 0.05101486f, 0.04269380f, 0.09817528f, 0.05402468f, 0.11035052f, 0.05238701f, 0.07820073f, 0.05572548f, 0.02310778f, 0.04631245f, 0.06332659f, 0.01948171f, 0.05543391f, 0.07149528f, 0.01540171f, 0.02227325f, -0.01422997f, -0.00486719f, 0.01817859f, 0.01980460f, -0.00283663f, -0.01898155f, 0.03599556f, -0.02103449f, 0.01154796f, -0.00385195f, -0.02623957f, -0.02926745f, -0.00482407f, -0.06924202f, -0.01326842f, -0.01343341f, -0.06372957f, -0.06307679f, -0.07616893f, 0.05857002f, 0.09353325f, 0.09879264f, 0.08239578f, 0.08320633f, 0.10505463f, 0.02420357f, 0.03238476f, 0.03625487f, -0.01904267f, -0.03052113f, -0.01399615f, 0.03162518f, 0.07821721f, -0.00486734f, -0.03794096f, -0.04715840f, -0.02347112f, 0.04292195f, 0.02112478f, 0.03359469f, -0.00932854f, 0.01807251f, 0.05681467f, 0.00708457f, -0.00727594f, 0.00501775f, -0.02801870f, -0.06098434f, -0.03998020f, -0.01318023f, 0.01427125f, 0.00943012f, -0.05144325f, -0.05181018f, -0.02871134f, 0.01221230f, 0.00173835f, 0.01605833f, 0.03524509f, 0.04394568f, -0.02783944f, 0.01846399f, -0.01277527f, -0.01003297f, -0.00532444f, -0.02363567f, 0.00026267f, 0.02437971f, 0.04796951f, 0.03091120f, 0.03321073f, -0.01999232f, -0.03471395f, -0.03033794f, -0.02999753f, -0.01482711f, 0.02581802f, 0.00242945f, -0.00971269f, -0.02785725f, -0.01713163f, -0.01247091f, -0.03175541f, -0.04381396f, -0.00784815f, -0.00521646f, 0.01169384f, 0.00918897f, -0.01805611f, -0.03668087f, -0.03729904f, 0.04738782f, -0.01361197f, -0.00569464f, 0.06453477f, 0.02111610f, 0.02302704f, -0.01371679f, 0.01879722f, 0.03773638f, 0.05989062f, 0.02249751f, -0.01791807f, 0.03369345f, 0.01346113f, 0.03301729f, -0.04218162f, 0.00447993f, 0.04982991f, 0.03456219f, -0.03564013f, -0.01101283f, 0.03264554f, 0.00166872f, 0.03075006f, -0.00030385f, 0.02047605f, 0.03541711f, 0.00035158f, 0.01552482f, -0.02387243f, 0.03826675f, -0.06139391f, 0.02683273f, -0.02310314f, -0.03463944f, -0.00565878f, 0.03424580f, 0.00809238f, -0.03752724f, 0.04206236f, 0.03092431f, 0.04170616f, -0.02579157f, -0.00273713f, 0.01127893f, 0.04112407f, -0.01928848f, -0.02655316f, 0.05934175f, 0.02740042f, 0.02616494f, -0.04043882f, -0.03337565f, 0.03907538f, 0.01912159f, -0.00824227f, -0.04789111f, 0.01204626f, -0.02273761f, 0.04069171f, -0.03604286f, -0.04675605f, 0.00628104f, 0.05359628f, -0.01809217f, -0.03363368f, 0.00575847f, -0.04815764f, 0.04269510f, -0.08587814f, -0.04890182f, 0.05267826f, 0.03855951f, -0.02049722f, -0.04825587f, 0.00684720f, 0.03782525f, -0.00336884f, -0.02740897f, 0.00214816f, 0.01596953f, 0.00852673f, 0.03046265f, 0.00669154f, 0.03457516f, 0.01634935f, 0.00305046f, -0.00525691f, -0.02337077f, -0.01850894f, 0.02200189f, -0.03518162f, -0.02814184f, -0.06600507f, -0.03837941f, 0.00853054f, -0.07311551f, -0.04711448f, 0.02987258f, 0.01275753f, 0.03468731f, -0.00131854f, 0.04354443f, -0.01266366f, 0.06132745f, -0.00970769f, -0.00835323f, 0.04921725f, 0.03037232f, -0.00760517f, -0.01722874f, -0.00000747f, -0.00251828f, -0.03782246f, -0.00591813f, -0.02989419f, 0.00751602f, -0.01408851f, 0.00979313f, -0.03537198f, 0.00840786f, 0.00308763f, -0.00497116f, -0.03189218f, -0.00843390f, -0.02191830f, 0.01580682f, -0.00900947f, -0.04233203f, -0.00994901f, -0.03717872f, -0.00741131f, 0.00233228f, -0.01823768f, 0.02283363f, -0.00029260f, 0.03067602f, -0.01121606f, -0.00826640f, -0.00986405f, 0.01301492f, 0.01028010f, 0.04725009f, 0.01820990f, 0.01555933f, 0.02889342f, -0.01900051f, -0.02925536f, 0.00197500f, -0.03898763f, -0.02749844f, 0.00275458f, 0.00472023f, 0.04110301f, 0.07326284f, 0.03711174f, 0.04311446f, 0.03744451f, 0.00342830f, 0.00276028f, -0.02820069f, -0.01010636f, -0.01739480f, 0.01007574f, -0.00271486f, -0.04932312f, 0.01797952f, -0.00639141f, 0.01780590f, 0.03134913f, -0.02200604f, 0.00927214f, 0.02612590f, 0.01983889f, -0.01138821f, 0.00045417f, 0.02924121f, 0.01927993f, 0.03831020f, -0.00210703f, 0.07661331f, 0.04821881f, 0.00505587f, 0.02809476f, 0.07280443f, 0.04973956f, 0.00157713f, 0.02648265f, 0.02924448f, 0.03172894f, 0.04233724f, -0.05285650f, -0.00775849f, 0.00201294f, -0.01714894f, -0.03297575f, 0.00105687f, 0.00411668f, 0.00064761f, 0.02459151f, 0.00136226f, -0.00944282f, 0.01180843f, 0.01828858f, 0.01623177f, 0.04505251f, 0.01755650f, -0.00990396f, 0.02528178f, -0.04830532f, -0.03293060f, -0.06882352f, -0.03465189f, -0.03719869f, -0.00427847f, 0.01577876f, 0.04358880f, 0.04062099f, -0.04302058f, 0.05563285f, 0.07169953f, 0.03358020f, 0.03120543f, 0.02834369f, -0.01928765f, 0.04463584f, 0.03802984f, -0.03397391f, 0.01970156f, 0.07417133f, -0.00022064f, -0.01579328f, 0.01717530f, -0.01798490f, -0.00482874f, 0.04058108f, -0.04083295f, 0.01959339f, -0.00615119f, 0.01789991f, -0.01990647f, -0.00571394f, -0.01637731f, 0.00808691f, -0.01116586f, -0.06513184f, -0.01172903f, -0.02262384f, -0.02557676f, -0.07629207f, -0.02028062f, -0.06792990f, 0.06840893f, 0.04459779f, 0.00628803f, 0.04119278f, 0.06140901f, 0.02312765f, 0.03970009f, 0.04616587f, 0.03215369f, 0.02307247f, 0.02353263f, -0.01225684f, 0.00107569f, 0.01594861f, -0.00866409f, 0.02408426f, -0.00255351f, -0.01385974f, 0.01160292f, 0.01496015f, -0.03058372f, 0.02123344f, 0.00912332f, 0.00457739f, -0.02429619f, -0.00796603f, -0.00638799f, -0.03307646f, 0.00535036f, -0.03966535f, -0.02150640f, -0.03841149f, -0.04489772f, -0.05860925f, -0.05518189f, -0.02636503f, 0.02853669f, 0.04766671f, -0.01348339f, 0.00277954f, 0.03109388f, 0.01921960f, 0.03640298f, 0.01373783f, -0.01059019f, 0.04294094f, 0.06706217f, -0.02115162f, -0.03920882f, 0.00331341f, 0.00693213f, -0.01609012f, -0.00870185f, -0.02912634f, 0.01688934f, 0.02537179f, -0.03313936f, 0.00066127f, -0.01572538f, 0.00825502f, -0.00433293f, -0.01807858f, -0.00842589f, -0.00729294f, 0.02000674f, -0.05637068f, -0.00577311f, -0.06516575f, -0.02386949f, -0.03620352f, -0.01629398f, -0.04548480f, -0.00191752f, 0.02725411f, -0.01559660f, 0.01884577f, 0.01770338f, -0.01777633f, 0.00655420f, 0.00916774f, -0.05589752f, -0.01508255f, -0.00485254f, -0.05465161f, -0.02507524f, 0.00293861f, -0.06503414f, -0.00227944f, 0.01674633f, -0.01419216f, -0.01292281f, 0.01402731f, -0.03360571f, 0.03692609f, -0.03581885f, -0.02259249f, -0.00295893f, 0.02029527f, -0.04428556f, -0.01761039f, -0.00451766f, -0.02582847f, -0.00903784f, -0.02792272f, -0.07710719f, -0.02735334f, 0.01842053f, -0.03495527f, 0.04549083f, 0.04637969f, -0.01222602f, 0.02058290f, 0.05175173f, 0.01707179f, -0.00243104f, 0.00996749f, -0.01408597f, -0.01362303f, 0.00758005f, -0.03244358f, -0.02107503f, 0.02671075f, -0.01324591f, -0.01519111f, 0.00091436f, -0.04972227f, 0.00481992f, 0.03917996f, 0.01669471f, 0.04214123f, -0.02309367f, -0.03900089f, -0.04031215f, 0.00038413f, -0.03533260f, -0.02744528f, -0.01540801f, -0.02082232f, -0.02149994f, -0.03342665f, -0.03224736f, -0.05548627f, -0.05369843f, -0.09183941f, -0.01036556f, 0.03474237f, 0.01945615f, 0.04134807f, 0.09098073f, 0.04265560f, 0.04632885f, 0.04409271f, 0.01473905f, -0.00818702f, 0.00843878f, -0.00260551f, -0.00744471f, 0.01480319f, 0.01807285f, -0.00999056f, -0.00336640f, -0.00475619f, -0.02428139f, 0.00278543f, 0.00814181f, 0.01699213f, 0.01141056f, 0.00325422f, -0.02056812f, -0.02549216f, -0.04302592f, -0.04246019f, -0.03583694f, -0.08308261f, -0.02703734f, -0.03077884f, 0.00490426f, -0.06961292f, -0.05381527f, -0.07509100f, 0.06615981f, 0.07119699f, 0.05980883f, 0.03393572f, 0.06017216f, 0.05547001f, 0.01822241f, 0.03181805f, 0.04835779f, 0.02454496f, 0.03836241f, 0.05503915f, -0.01141289f, -0.02123850f, 0.00291733f, 0.00789237f, -0.00602251f, 0.00514024f, 0.00058448f, 0.03398811f, -0.00693249f, -0.02747576f, -0.02216334f, 0.06632200f, -0.00145502f, 0.00096544f, -0.04574344f, -0.04522492f, -0.00242772f, 0.00006261f, -0.06149092f, -0.08578241f, -0.04618456f, -0.04622852f, -0.05087251f, -0.07677597f, 0.07175997f, 0.05259425f, 0.06235531f, 0.04274979f, -0.01450479f, 0.05605650f, 0.06729395f, 0.03237069f, 0.05002569f, 0.01968140f, -0.00984909f, -0.02103294f, -0.00787862f, 0.00493400f, 0.02153343f, 0.00653491f, -0.01557499f, 0.02177604f, -0.00509178f, 0.01576185f, 0.01875064f, -0.00141228f, -0.05653560f, 0.02235726f, 0.01249101f, -0.00227329f, -0.03121114f, -0.05694037f, -0.04348139f, -0.04671987f, 0.00621519f, -0.03461743f, -0.01388734f, -0.03964582f, -0.04386381f, -0.05904256f, 0.05138507f, 0.01274178f, 0.00120371f, 0.00543445f, -0.02279082f, -0.03406797f, 0.00794799f, 0.02129365f, 0.07883125f, 0.05587815f, -0.01012229f, -0.00342389f, 0.00882288f, 0.04226666f, -0.02137096f, 0.00897658f, 0.00267523f, 0.06837290f, -0.04577796f, -0.03949954f, -0.01604776f, -0.01049543f, -0.01129267f, -0.06067382f, -0.03188060f, -0.00726918f, -0.01951312f, -0.01597191f, -0.04068583f, -0.01415112f, -0.04168606f, -0.06343972f, -0.08818689f, -0.00895443f, -0.02519328f, -0.00991010f, -7.52162767f };
	return std::vector<float>(detector, detector + sizeof(detector) / sizeof(detector[0]));
}

std::vector<float> HOGDescriptor_Mod::HOG_Optimal_64_128()
{
	static const float detector[] = { 0.06960074f, -0.07910255f, -0.06437013f, 0.04607003f, 0.10450290f, -0.00586068f, 0.01865242f, -0.00537472f, 0.10567640f, 0.11804003f, -0.01261269f, -0.01596124f, 0.05620911f, 0.06970171f, 0.13417541f, 0.07938405f, -0.02585638f, 0.05457407f, 0.08104900f, -0.06033890f, -0.02407077f, 0.09277076f, 0.02295698f, 0.06168086f, -0.02827385f, -0.04398259f, 0.08228857f, 0.10209226f, 0.00494431f, 0.00265260f, 0.10032313f, 0.04726348f, 0.06062548f, 0.02344172f, 0.01288481f, 0.01969737f, 0.02974369f, -0.01381898f, -0.01045350f, -0.00725927f, 0.02024396f, 0.00178194f, -0.00023828f, -0.04505149f, 0.02023832f, 0.06166428f, 0.00584761f, 0.04678846f, 0.07192736f, 0.07133866f, 0.03796736f, 0.04983972f, 0.07169937f, 0.04940093f, 0.06764664f, -0.05326992f, -0.02930772f, -0.01212818f, -0.02376785f, 0.02940384f, -0.03488080f, -0.00750339f, 0.05887952f, 0.10149691f, -0.05050758f, -0.00643477f, 0.05894846f, 0.01680244f, 0.05271001f, 0.01610390f, -0.02605498f, 0.03282842f, 0.02418945f, -0.01306722f, 0.01240543f, 0.03399486f, 0.04912702f, 0.01213922f, 0.06297701f, 0.07375181f, 0.03055856f, 0.03052112f, 0.01456601f, 0.01761672f, 0.04895630f, 0.04497886f, 0.00344149f, 0.00997411f, 0.04155576f, 0.10666752f, 0.01431996f, -0.05965952f, -0.00948065f, 0.02302527f, -0.02155631f, -0.00714321f, 0.00405737f, -0.02482333f, -0.00245854f, -0.03339281f, -0.04856359f, -0.03631790f, 0.01929850f, -0.05194499f, 0.00372117f, -0.00232599f, 0.04094534f, -0.02640645f, 0.02033898f, 0.07385735f, 0.02974924f, 0.04644308f, 0.08857111f, 0.00960392f, 0.03781292f, -0.02534788f, 0.02227156f, -0.01437129f, 0.03984442f, -0.05590172f, 0.01604015f, -0.01508070f, -0.00359818f, 0.00608212f, -0.02045309f, 0.00300021f, -0.01373532f, 0.04084454f, 0.04150361f, -0.02467513f, -0.04307213f, -0.01892224f, 0.00457642f, 0.04094762f, -0.02435527f, -0.03971873f, -0.05930152f, -0.06831138f, -0.02979579f, -0.08603250f, -0.00667215f, 0.00802225f, -0.01981694f, -0.03137015f, 0.00338195f, 0.11632528f, 0.01005481f, 0.04091873f, 0.07054788f, -0.01195079f, 0.05142254f, 0.09572970f, -0.00431714f, -0.03975862f, -0.03683252f, -0.09791683f, -0.03187240f, 0.02006094f, -0.06738127f, 0.01166512f, -0.00220883f, -0.06368266f, 0.05352161f, 0.09073208f, 0.01114644f, 0.08744769f, 0.09396043f, 0.07253585f, 0.12693140f, 0.11254290f, 0.05276652f, -0.05203199f, -0.01657395f, -0.09094737f, -0.05001587f, 0.00613750f, 0.02955803f, 0.06669997f, 0.02096979f, -0.01660619f, -0.11843991f, -0.08048791f, -0.08057443f, 0.04024167f, 0.11391305f, 0.00889569f, 0.00654018f, 0.00708368f, -0.05202452f, -0.08707163f, -0.07463144f, -0.07435784f, -0.03913238f, -0.02755833f, -0.02293216f, -0.10179774f, -0.02376985f, -0.08029328f, 0.04585383f, 0.02685099f, -0.05556581f, 0.06571198f, 0.10861532f, 0.04250555f, -0.03168184f, 0.03011994f, -0.01511626f, 0.04234898f, 0.01085239f, -0.04181188f, 0.00483441f, 0.03455166f, 0.01880864f, -0.10143370f, -0.00281951f, -0.04195251f, 0.03584442f, -0.03733712f, -0.04114711f, 0.01085210f, 0.01846596f, 0.02066835f, -0.06258660f, 0.03653050f, 0.04921238f, -0.08524538f, -0.08704899f, -0.07592314f, -0.00133689f, 0.01903479f, -0.01725872f, -0.12558660f, 0.03616411f, -0.01197440f, 0.12979132f, 0.04391966f, 0.01234232f, 0.06200677f, 0.12438298f, 0.03434636f, -0.05497097f, 0.05531288f, 0.06196250f, -0.05935057f, -0.06238275f, -0.03683921f, -0.01576730f, 0.01146300f, 0.05414963f, -0.08907664f, -0.00231442f, -0.07803914f, -0.01396276f, 0.05754373f, -0.06128905f, 0.07419887f, 0.08034113f, -0.01310517f, -0.07453193f, 0.09557124f, 0.07647574f, -0.08810071f, -0.03940562f, -0.08486685f, -0.00391592f, -0.03282853f, -0.03780516f, -0.10506383f, 0.01718735f, -0.08164467f, 0.02258125f, 0.01931810f, 0.03963929f, 0.10480297f, 0.06523681f, 0.06175649f, -0.00908779f, 0.07881026f, 0.04672651f, -0.04987736f, -0.05926617f, -0.04429869f, 0.00742671f, -0.03872892f, -0.00733125f, -0.04713406f, -0.00668811f, -0.04523082f, -0.02723457f, -0.04849768f, 0.01116395f, 0.03717132f, -0.02725594f, 0.01558580f, 0.01643124f, 0.06199597f, -0.00060132f, -0.05503221f, -0.06429205f, -0.06115689f, -0.03118242f, -0.07986848f, -0.04300274f, 0.01452845f, 0.01124224f, -0.01972953f, 0.07919165f, -0.04448921f, 0.04708096f, 0.06548114f, 0.03119136f, 0.09699393f, 0.03827638f, 0.05062755f, 0.02008723f, -0.03295087f, -0.01154608f, -0.01377520f, 0.00512636f, -0.06325997f, -0.02295671f, -0.02236337f, 0.00924799f, 0.00066625f, -0.01509641f, 0.04530014f, -0.05234995f, -0.00624648f, -0.03576847f, -0.03125661f, -0.05569574f, 0.01867164f, 0.04657721f, -0.02875693f, -0.03842692f, -0.11628526f, -0.11020558f, 0.02480212f, -0.01589854f, -0.08806840f, -0.00232823f, 0.03228751f, 0.01662660f, 0.03766960f, 0.04508738f, 0.09455802f, -0.00103042f, 0.08096416f, 0.03879942f, 0.09172562f, 0.02241904f, -0.03893210f, 0.00219296f, -0.07389283f, 0.00271286f, 0.01303298f, 0.04139126f, -0.03841353f, 0.02059950f, -0.02596098f, -0.02288342f, 0.01455757f, -0.01130868f, -0.01548489f, 0.06613617f, 0.02474101f, -0.05936618f, -0.02735984f, -0.00763537f, -0.03727250f, -0.03418258f, -0.02895881f, 0.03013540f, -0.01012524f, 0.00305108f, -0.02735645f, 0.04905102f, 0.00459389f, 0.00379407f, 0.05095290f, 0.03701726f, 0.03130617f, 0.03093460f, 0.03208000f, -0.02719921f, -0.00605023f, -0.04386398f, -0.04548312f, 0.03469705f, 0.05469540f, 0.06486343f, -0.03726215f, 0.03067910f, 0.02562421f, 0.05831153f, -0.04702030f, 0.00500099f, -0.02819646f, -0.02341967f, 0.04278446f, 0.01363667f, -0.02990129f, -0.03473084f, 0.06363503f, 0.03293840f, 0.00139445f, -0.01881368f, -0.02899660f, 0.03158714f, -0.09709465f, -0.03605020f, -0.08825810f, -0.06049157f, -0.02397004f, -0.02248443f, 0.04080577f, -0.02062077f, 0.04664751f, 0.00257309f, 0.03581612f, 0.03546309f, 0.03640778f, -0.02544853f, -0.03489899f, -0.04649322f, -0.03318461f, -0.01562552f, 0.00650999f, 0.00420965f, 0.02779845f, -0.07672394f, -0.04105153f, -0.06844650f, -0.00574437f, 0.00799230f, 0.01547132f, -0.01101068f, -0.03577148f, -0.00505739f, -0.01518662f, -0.01860208f, -0.09344950f, 0.00946149f, -0.03103602f, -0.06489965f, 0.01983058f, -0.02554591f, -0.02880522f, 0.12005348f, -0.04794249f, 0.01882248f, -0.01942754f, -0.00608123f, 0.11884168f, 0.11506599f, 0.01164149f, 0.06090829f, 0.05981008f, -0.02447448f, 0.01939334f, 0.06145078f, -0.01194518f, 0.01445580f, 0.04119866f, 0.03168549f, 0.01525902f, 0.08297217f, -0.01287615f, -0.06728674f, 0.08859281f, 0.05692705f, 0.07329954f, 0.01864715f, -0.02954922f, -0.03238055f, 0.15187715f, -0.02149784f, -0.00651292f, 0.03845980f, 0.08812981f, 0.01252998f, 0.00862821f, -0.04762332f, -0.03994296f, -0.02458221f, -0.05100199f, 0.00861806f, 0.04613860f, 0.00591460f, 0.02059587f, -0.01560224f, 0.07189245f, -0.00921535f, 0.09445974f, 0.03989306f, -0.01219691f, -0.00941382f, 0.03252027f, 0.03569459f, 0.09150652f, 0.01737602f, -0.04159127f, -0.03383612f, 0.04260914f, 0.03084239f, 0.03576593f, 0.01244146f, 0.06455856f, -0.03328300f, 0.09544178f, 0.00835738f, -0.00248925f, 0.01085978f, 0.03007352f, -0.09330325f, 0.06030752f, 0.09643646f, -0.10116150f, 0.05128449f, -0.03315620f, -0.06667903f, 0.04833730f, 0.02178794f, -0.03696589f, -0.02364572f, 0.09327405f, -0.00680306f, 0.13361430f, 0.00696933f, 0.01656982f, 0.04769811f, 0.03035261f, -0.11202969f, -0.03844355f, 0.01986645f, -0.13874409f, 0.05461767f, -0.06321133f, -0.04622944f, 0.06725765f, 0.08167562f, -0.03366510f, -0.02163895f, 0.06817791f, 0.11813218f, 0.05172892f, -0.04426898f, -0.09811206f, 0.05347703f, 0.10855933f, -0.00531976f, 0.01175883f, 0.02587716f, 0.04203554f, 0.09653664f, 0.02145993f, 0.03839565f, 0.06057927f, 0.06043516f, -0.10207824f, -0.05971764f, 0.06402207f, -0.00564996f, 0.02403841f, -0.01925407f, -0.05676549f, 0.06829813f, 0.05319206f, -0.00220463f, -0.01708261f, 0.04024707f, -0.04542009f, 0.07369622f, 0.02532946f, 0.03460226f, 0.02724286f, 0.09044345f, -0.06110506f, -0.02554124f, -0.07963590f, 0.02944800f, 0.01737883f, -0.01138925f, 0.00434876f, 0.03788580f, 0.06216901f, -0.02524021f, -0.01982287f, 0.01623533f, 0.05643840f, 0.03450578f, 0.01817477f, -0.07409399f, -0.01299853f, 0.06650191f, -0.03668137f, -0.00803914f, -0.04933684f, 0.03324857f, 0.05994915f, -0.01174999f, -0.07422469f, -0.02336119f, 0.06531696f, -0.06065569f, 0.05606217f, 0.06919837f, -0.06129608f, 0.00322706f, 0.03053039f, -0.09108629f, -0.00956994f, 0.04199366f, -0.02381545f, -0.00454903f, 0.00186041f, 0.04898115f, -0.04063163f, -0.01083457f, -0.05641827f, -0.02722128f, -0.02894478f, -0.00075000f, -0.03483834f, 0.01368399f, -0.01174365f, 0.01009418f, 0.01803325f, 0.01791049f, -0.00867728f, 0.03796733f, -0.06582177f, -0.03309285f, 0.05587475f, 0.05657276f, -0.03541853f, -0.04388442f, -0.00755128f, 0.06738478f, -0.05171345f, -0.20789213f, -0.13444085f, 0.01814083f, 0.04567640f, 0.00758124f, -0.00441769f, 0.05916842f, 0.02366673f, -0.02483655f, -0.03853156f, -0.03572855f, -0.10212882f, -0.03696109f, -0.06489362f, 0.01225732f, -0.00744844f, -0.02272703f, 0.01538818f, 0.02882133f, -0.01709451f, -0.08238053f, 0.00951158f, 0.00971718f, -0.02853625f, -0.02510527f, -0.02760956f, -0.03688918f, 0.04128409f, 0.08914263f, 0.00466468f, -0.01142627f, -0.04441936f, 0.04449618f, 0.05583890f, 0.05406217f, -0.06906101f, 0.14953725f, 0.00524110f, -0.04700478f, -0.05303912f, 0.01283664f, -0.02719899f, -0.07909084f, -0.02461116f, 0.12437804f, 0.16075112f, 0.03020657f, -0.04037953f, 0.05887264f, -0.01146285f, 0.02164629f, 0.01626474f, 0.01457685f, 0.02769515f, 0.13660448f, -0.10870128f, -0.09057141f, 0.05373538f, -0.03148268f, -0.02914509f, -0.03388715f, -0.01651153f, 0.07939537f, 0.23861266f, -0.04400615f, -0.01145773f, 0.03455182f, 0.00745242f, 0.00683111f, 0.02052265f, 0.01183270f, 0.04535913f, 0.07448519f, -0.12807863f, -0.04379979f, 0.01830180f, -0.04347844f, 0.00688690f, -0.02035962f, -0.01519204f, 0.06060176f, 0.15378401f, -0.05110507f, -0.02300957f, 0.07148504f, 0.03868479f, 0.01884508f, -0.01683958f, -0.00708237f, 0.02954396f, 0.11302351f, -0.02860935f, -0.00558425f, 0.02755653f, 0.01339550f, -0.00421389f, 0.00930553f, 0.00373716f, 0.04544627f, 0.10102851f, -0.00046553f, 0.01046394f, 0.06866586f, 0.04866524f, 0.09446286f, -0.00168333f, -0.08089137f, -0.02275773f, 0.08008642f, -0.03162927f, -0.03207872f, -0.03576378f, -0.01596744f, 0.00606315f, -0.05009763f, -0.09176443f, 0.07660228f, 0.07982963f, 0.00815900f, 0.00687980f, 0.05651961f, -0.02241939f, -0.02908284f, 0.07615300f, 0.07826436f, 0.00609035f, -0.01585576f, -0.00045398f, -0.06712896f, -0.01295133f, -0.07752959f, -0.06482802f, 0.03104064f, -0.01427409f, 0.07459928f, 0.18171735f, 0.02144636f, 0.00046766f, 0.08488859f, 0.09497325f, 0.02421590f, 0.06594469f, 0.01603512f, 0.05739294f, 0.08794581f, -0.02308071f, -0.03743531f, 0.09326668f, 0.00869198f, -0.03684134f, 0.00854264f, -0.03153613f, 0.07233900f, 0.07372520f, 0.04685167f, 0.00160380f, -0.02334387f, -0.03804147f, 0.03221885f, 0.10057424f, 0.08221770f, -0.00805337f, 0.01777914f, -0.03174724f, -0.08544096f, -0.03045937f, -0.03852118f, 0.05841822f, 0.09461058f, 0.03187372f, 0.16764661f, 0.17638428f, 0.05445889f, 0.06525106f, 0.09263208f, 0.08391844f, 0.06328176f, 0.12976746f, -0.00318421f, 0.08005284f, 0.01396492f, -0.03200523f, 0.01918371f, 0.07041435f, 0.09015123f, 0.03298481f, 0.05376414f, 0.08737821f, 0.03734978f, 0.02819695f, -0.04178618f, -0.01233209f, 0.05720787f, 0.02487367f, -0.01751917f, 0.04371063f, 0.02097893f, -0.00144264f, 0.05038103f, -0.02370138f, -0.03224995f, -0.01264137f, -0.08037745f, -0.05193098f, -0.00230810f, -0.08375254f, 0.08917055f, 0.01519985f, -0.04329246f, 0.11548568f, 0.18923959f, 0.16297438f, 0.07309161f, 0.08583172f, 0.06695853f, 0.02068656f, -0.03613257f, -0.03557049f, 0.05799189f, 0.10313621f, 0.09529394f, -0.01778218f, -0.00841283f, 0.04997647f, -0.01584882f, 0.01032213f, -0.00077680f, 0.02614496f, -0.02908483f, 0.01034443f, 0.02035438f, 0.07238750f, -0.05374283f, -0.01796641f, 0.01717458f, -0.05961612f, -0.10247042f, -0.04846918f, 0.01254844f, -0.03402378f, 0.06539351f, -0.04633787f, 0.01333455f, 0.05624354f, -0.03496066f, 0.07830934f, 0.10608245f, 0.09276765f, -0.04826369f, -0.01544803f, -0.04944588f, -0.05049294f, 0.02492731f, -0.07248493f, 0.00049714f, 0.02112267f, -0.01108904f, -0.09430600f, -0.10559062f, -0.02140550f, -0.00457797f, 0.02974440f, -0.06749954f, -0.11792414f, -0.04647963f, 0.02735898f, -0.02380046f, 0.08399769f, 0.02020549f, -0.07388193f, -0.01908477f, -0.02637865f, -0.05633124f, -0.11482161f, 0.00656764f, -0.00258727f, 0.03427686f, -0.04501021f, 0.10407639f, 0.08061978f, 0.00419704f, 0.04959927f, 0.12798217f, 0.04701756f, 0.02662782f, 0.00752255f, 0.00159933f, 0.02613113f, -0.00357272f, -0.03562596f, 0.05904064f, 0.11502474f, 0.07488943f, 0.02365775f, -0.09488027f, 0.01791992f, 0.00808710f, 0.06181631f, 0.04134722f, 0.00491630f, -0.00717360f, -0.00302900f, -0.04393506f, 0.01396761f, -0.03282378f, -0.00212437f, 0.02535436f, 0.03287282f, -0.07873380f, -0.10823823f, -0.12169826f, -0.00023439f, -0.02013377f, -0.11445801f, 0.09613814f, 0.05445349f, -0.00329248f, 0.10041431f, 0.16960889f, 0.11335701f, -0.01981261f, -0.07992320f, 0.00352661f, 0.02055095f, 0.07705337f, 0.01535690f, 0.03724914f, 0.09470704f, 0.04591078f, -0.02041724f, -0.01616977f, -0.01287048f, -0.01522663f, -0.02495845f, 0.00931630f, -0.00770206f, -0.02598781f, -0.09064976f, 0.03418028f, 0.04117175f, -0.05282710f, -0.07323144f, -0.08029628f, -0.06049079f, -0.04233640f, -0.09303133f, -0.05374222f, -0.04474872f, -0.01559040f, -0.06557288f, 0.02474565f, 0.10667597f, 0.06002563f, 0.11232587f, 0.18106725f, 0.15170854f, 0.09399565f, 0.12378372f, 0.03219204f, -0.02925270f, 0.02376204f, 0.00252551f, 0.05362532f, 0.08713104f, 0.12201885f, 0.04144180f, 0.06934386f, -0.04113593f, -0.00677848f, -0.02807392f, 0.02922932f, -0.01149268f, -0.13679624f, 0.05307414f, -0.00701904f, 0.02003185f, 0.02679813f, 0.01314745f, -0.02590988f, 0.03118020f, -0.01779674f, -0.08943168f, -0.05339945f, -0.05484541f, 0.00143522f, 0.10642606f, 0.00797921f, 0.00231373f, 0.02989047f, 0.09819689f, -0.04144063f, 0.16352389f, 0.00980752f, 0.07758064f, 0.00577374f, -0.10056936f, -0.13700867f, 0.00288882f, -0.00911683f, -0.01669748f, 0.03687228f, -0.02733851f, -0.06324351f, -0.01131413f, 0.05921374f, -0.02364782f, -0.02104324f, 0.07763209f, -0.00840097f, 0.01290263f, -0.04250304f, 0.02786358f, 0.12887579f, 0.04376828f, -0.07858816f, -0.03731761f, 0.04056940f, -0.09367214f, -0.06022060f, -0.10545053f, -0.12175232f, 0.08003715f, 0.02452034f, -0.07306524f, -0.00763540f, 0.09737959f, -0.01439565f, 0.07279002f, -0.00347974f, 0.02734687f, 0.05450626f, -0.00734909f, -0.09455590f, -0.03077139f, 0.15314244f, -0.05525741f, 0.01528207f, -0.03889421f, -0.09813819f, -0.00334519f, -0.01295653f, -0.11548741f, -0.03067251f, 0.09121321f, 0.05880432f, -0.00594771f, -0.00200834f, -0.05562357f, 0.01455468f, 0.07627108f, -0.02150783f, 0.02954642f, 0.10554297f, 0.06411020f, 0.05006259f, 0.02542589f, 0.00528754f, 0.09539164f, 0.03380587f, -0.04401643f, 0.01250404f, 0.06118274f, 0.02720583f, -0.05857123f, -0.04373591f, -0.09409721f, 0.03121506f, 0.05529752f, -0.09351661f, 0.00988854f, 0.05279034f, 0.02492944f, 0.00320658f, -0.03868426f, -0.05608573f, 0.10348159f, -0.00239112f, -0.12988874f, -0.03204302f, -0.07388566f, -0.02833838f, -0.15400823f, -0.07971009f, -0.13101488f, -0.02775406f, 0.14115304f, 0.07432867f, 0.08994732f, 0.09596008f, -0.01449583f, 0.00675850f, 0.00076710f, -0.02941752f, 0.00167080f, -0.03786029f, -0.01903632f, 0.28061192f, 0.17761046f, 0.04582863f, -0.10168030f, -0.02863333f, -0.17658263f, -0.05114053f, 0.19254447f, 0.17714414f, 0.25634045f, 0.20536253f, 0.00517343f, -0.02326372f, -0.01336097f, -0.08396951f, -0.00984022f, 0.11835370f, 0.00603654f, -0.05385999f, -0.00195236f, 0.00230337f, -0.07100704f, -0.01526925f, 0.09316690f, 0.13304370f, -0.01635832f, -0.09602780f, -0.00440822f, 0.10726507f, -0.01611074f, -0.07200804f, 0.04553917f, 0.17119280f, 0.09361311f, 0.18586376f, 0.06839576f, -0.01874620f, 0.02835977f, 0.05484151f, -0.02086519f, 0.00205041f, 0.21254700f, 0.20121197f, 0.06659029f, 0.01702241f, 0.11750215f, 0.13223360f, 0.07094526f, 0.01856550f, 0.15923735f, 0.32807409f, 0.26829863f, -0.01403030f, 0.23622344f, 0.42190397f, 0.32000939f, 0.03522340f, -0.10172980f, 0.05933426f, 0.09037292f, 0.01042787f, -0.05004960f, 0.19258585f, 0.16661775f, 0.02939760f, -0.04318697f, -0.12148365f, -0.13517769f, -0.12231070f, -0.18656097f, -0.04272203f, 0.20447802f, 0.40059808f, 0.30333641f, 0.04207285f, -0.03082788f, 0.12330266f, 0.22659461f, 0.09273879f, -0.17037098f, -0.06520060f, 0.00170100f, -0.05521762f, -0.11810159f, -0.08199317f, -0.01144764f, -0.02562937f, -0.10814133f, 0.08765217f, 0.25765670f, 0.00224881f, 0.07047440f, 0.10162493f, 0.02706992f, 0.08270607f, 0.08260721f, 0.00532284f, -0.01074478f, 0.05756406f, -0.03607767f, 0.00901902f, 0.01419247f, -0.01177657f, 0.09066866f, 0.12441589f, 0.06011432f, 0.05076726f, -0.03991549f, -0.09992832f, -0.08371614f, -0.05586747f, 0.01529317f, 0.08735966f, 0.06822567f, -0.01332651f, -0.03912928f, -0.08635812f, -0.08882964f, -0.02652092f, -0.08881731f, -0.07423050f, -0.03606897f, 0.00687873f, 0.07268100f, 0.04385706f, 0.12769891f, 0.03211974f, 0.01300909f, 0.00274292f, -0.00392251f, 0.02069605f, 0.07260749f, 0.06419776f, 0.06897861f, 0.21285249f, 0.03340761f, -0.00448121f, -0.05286799f, 0.04703905f, -0.00349360f, 0.10184880f, -0.05962481f, 0.04108404f, 0.04167766f, -0.05839589f, 0.05031641f, -0.00853674f, -0.05377093f, -0.10610685f, -0.05613273f, 0.05750779f, 0.06619822f, 0.02564794f, -0.08115233f, -0.11943945f, -0.08817159f, -0.08459294f, -0.07183663f, -0.02269165f, -0.00495797f, 0.05103676f, 0.25100222f, 0.07254552f, 0.03183750f, -0.04893135f, 0.06079649f, 0.04942617f, 0.11291160f, -0.00165548f, 0.08674279f, 0.10068677f, -0.06777998f, -0.06557640f, 0.02431690f, -0.00199580f, 0.05942251f, 0.04604861f, -0.05338942f, 0.04366274f, -0.00301893f, -0.04224145f, -0.03472164f, -0.07353562f, -0.02156851f, 0.00912359f, 0.01958570f, -0.02697408f, -0.00421565f, -0.04208474f, -0.12131405f, -0.06146094f, -0.06682025f, -0.03062361f, -0.03734079f, -0.06887945f, -0.05500408f, 0.15414730f, 0.17416174f, 0.00621537f, -0.00654827f, 0.08901984f, 0.08006841f, 0.05066208f, 0.08275032f, 0.05270678f, 0.12695638f, 0.03301486f, 0.01193789f, 0.00137572f, 0.06244359f, 0.05446614f, 0.00157733f, 0.08920864f, 0.13371805f, 0.05825487f, 0.05591298f, -0.05548166f, 0.02024262f, -0.02884784f, 0.03715298f, -0.01443080f, -0.01868618f, -0.06108704f, -0.10577924f, -0.06803061f, -0.04470763f, -0.06362580f, -0.00117467f, -0.03377224f, -0.06869606f, -0.06722922f, -0.05857905f, 0.10452976f, 0.08621247f, 0.00520305f, 0.01713345f, 0.06219872f, 0.08489912f, 0.00104784f, 0.19861759f, 0.24111567f, 0.05127507f, 0.02813143f, -0.06796147f, -0.00625424f, 0.08667467f, 0.11099239f, -0.01683205f, 0.16230972f, 0.10856808f, -0.06361820f, 0.02803698f, -0.04696204f, -0.09614237f, -0.05601963f, -0.04836019f, -0.03248577f, 0.05248131f, 0.07445758f, -0.01025675f, -0.00460952f, -0.13553335f, -0.09856925f, 0.00683270f, 0.02574812f, -0.12798328f, 0.03955332f, 0.03221257f, 0.01794393f, 0.04776020f, -0.06445035f, 0.00916071f, 0.02693541f, 0.03378900f, -0.06965358f, 0.10093983f, 0.09200808f, 0.02197915f, 0.06627928f, -0.07755812f, -0.06271934f, 0.02045279f, -0.03904405f, -0.08056513f, 0.05051399f, 0.02420075f, 0.01080262f, 0.05583039f, -0.09473789f, -0.00266427f, -0.01483345f, 0.00581058f, -0.13842497f, 0.07816215f, -0.00254015f, 0.00419949f, 0.09800242f, -0.07415507f, -0.07156817f, -0.08680350f, -0.00810248f, -0.04305274f, 0.08644483f, -0.00716540f, -0.05553313f, 0.10253932f, -0.09770679f, -0.08892951f, -0.00324666f, -0.04849985f, -0.05454857f, 0.01538604f, -0.07181175f, -0.03809754f, 0.01523939f, -0.10870024f, -0.01186004f, -0.03356268f, -0.03632705f, -0.07549796f, -0.02727906f, 0.01846765f, -0.03563254f, 0.19811852f, -0.06388418f, -0.06882755f, -0.03078205f, -0.01131457f, -0.01718448f, 0.07207056f, -0.02840447f, 0.00650126f, 0.05132307f, -0.09239545f, -0.02906800f, -0.05330393f, -0.03761130f, -0.06575563f, -0.01621895f, 0.03828311f, -0.04428537f, 0.01343388f, -0.07819474f, -0.03922791f, -0.02843613f, -0.03346031f, -0.14959151f, -0.05469074f, -0.01753938f, -0.11383430f, 0.00110295f, -0.11999028f, -0.13703332f, -0.09120263f, -0.08554915f, -0.14097652f, -0.03103814f, -0.04019416f, -0.01175187f, 0.04756390f, -0.10375603f, -0.05550424f, 0.01849070f, -0.03722610f, -0.13711964f, 0.05780812f, 0.02329045f, 0.03214633f, 0.12074963f, -0.04335040f, -0.06041257f, -0.02422041f, -0.07330122f, -0.11441324f, 0.09837819f, 0.03005112f, -0.04494757f, 0.04753376f, -0.04514168f, -0.05434216f, -0.02385532f, -0.01774866f, -0.03262895f, 0.04509021f, -0.00586314f, -0.12664779f, -0.06099739f, -0.04927407f, -0.09716125f, -0.10501535f, -0.13160032f, -0.08293201f, -0.02646802f, -0.11977829f, 0.10174614f, 0.21282150f, 0.18045636f, 0.12185562f, 0.10992785f, -0.00084021f, 0.02838312f, 0.15336230f, 0.03335326f, 0.10375317f, 0.15869828f, 0.13477042f, 0.03063769f, 0.07326529f, 0.03441737f, -0.01468933f, 0.11120885f, 0.00734541f, 0.02541766f, -0.02323823f, -0.04002049f, -0.00893040f, 0.06380278f, 0.16349081f, 0.05304978f, 0.06820717f, 0.05219530f, -0.12485037f, -0.17000162f, -0.08060718f, -0.16334976f, -0.09512093f, -0.05174159f, -0.09740466f, -0.10831099f, -0.06053162f, 0.19426946f, 0.20493386f, 0.07309886f, 0.08002682f, 0.20577240f, 0.29354363f, 0.09028221f, 0.13839460f, 0.11885204f, -0.09271783f, -0.11477423f, -0.06206481f, -0.17113071f, 0.09017638f, 0.06175470f, -0.12165855f, -0.09732720f, -0.06949533f, -0.00006433f, -0.01018705f, 0.07978140f, 0.10420472f, -0.01123709f, 0.09467249f, 0.00256792f, 0.05138805f, 0.05457307f, -0.05763914f, -0.09524970f, 0.00194025f, 0.06952175f, -0.12511535f, 0.03882967f, -0.03387679f, -0.07309376f, -0.01985399f, -0.05351670f, 0.01093779f, 0.10074844f, 0.06903393f, -0.06102320f, 0.07946286f, -0.00242635f, 0.07359197f, 0.02531302f, -0.00752257f, -0.11155552f, -0.00698449f, 0.07179832f, -0.07817563f, 0.10387887f, 0.00048559f, -0.03468842f, 0.03167307f, 0.10292331f, -0.07793021f, -0.08415280f, -0.08339339f, 0.01647992f, -0.04904274f, -0.05939008f, -0.07599731f, 0.11615392f, 0.16038609f, -0.05146302f, -0.03385259f, 0.02818935f, 0.09587006f, 0.03730140f, -0.00004537f, 0.01013889f, 0.15074077f, 0.10273107f, -0.07283400f, -0.09015583f, -0.02490961f, 0.02376870f, -0.02112401f, -0.05420239f, -0.04758819f, 0.11993233f, 0.19315455f, 0.00291638f, -0.03907344f, 0.07058710f, 0.04111342f, 0.05529915f, -0.01694724f, -0.07393680f, 0.11659611f, -0.11942983f, -0.03807777f, 0.25231481f, 0.17270091f, 0.11258026f, 0.09519437f, 0.10052715f, -0.10051223f, -0.08773164f, -0.01146547f, 0.23780833f, 0.31990073f, 0.19293504f, 0.01995098f, 0.09311732f, 0.09966740f, 0.01580437f, -0.18183737f, -0.10202997f, -0.09031059f, 0.06595499f, 0.08600569f, 0.16088604f, 0.19916702f, 0.25401129f, -0.04354094f, -0.12326746f, -0.11945769f, -0.00152793f, 0.07221148f, 0.07764767f, 0.00122611f, 0.12126420f, 0.28187991f, 0.21828611f, -0.01886932f, 0.12866444f, 0.19271771f, 0.03987121f, 0.04880741f, 0.07191844f, -0.00758365f, 0.02855291f, 0.19857241f, 0.08070691f, 0.01079173f, 0.05522430f, 0.09164755f, 0.05933057f, 0.01950224f, 0.01326083f, 0.10558845f, 0.18055661f, 0.10640845f, 0.10107291f, 0.21461466f, 0.05660295f, 0.02778826f, 0.02161117f, 0.01745544f, 0.04202072f, 0.14025270f, 0.14117041f, 0.05753179f, 0.15805494f, 0.09757030f, -0.04308265f, 0.01331392f, 0.00436299f, 0.12289649f, 0.06362046f, 0.05825461f, -0.08654437f, 0.15317151f, 0.30641749f, 0.21005283f, 0.07195550f, 0.07441432f, 0.17387201f, 0.12147784f, 0.02276881f, -0.15712526f, -0.10620222f, -0.02694243f, -0.07373337f, 0.00816514f, -0.00989296f, 0.04207446f, -0.01967459f, -0.09486740f, -0.00263296f, 0.15137986f, 0.17169810f, 0.03263650f, 0.03490729f, 0.17832727f, 0.29185171f, 0.11716523f, -0.02801437f, -0.12254806f, -0.04524283f, 0.02964921f, -0.06324411f, -0.05332625f, -0.07190440f, -0.03318333f, -0.11961899f, -0.19949737f, 0.02858808f, 0.10207970f, 0.04197144f, -0.05626654f, -0.08867483f, -0.00656615f, 0.06305695f, 0.07862643f, -0.03969377f, -0.07308021f, -0.01794955f, -0.05649801f, -0.03779962f, -0.14533126f, -0.07029242f, -0.00488660f, -0.04361404f, -0.02507398f, -0.02695305f, 0.11906534f, 0.10504466f, 0.00536339f, -0.10630674f, -0.05965621f, -0.04583737f, -0.01130052f, -0.07146119f, -0.01776615f, 0.00038528f, -0.02080515f, -0.07024461f, -0.17622866f, -0.04533916f, -0.04812050f, -0.03287394f, -0.07534602f, -0.04427852f, 0.07306324f, -0.02787158f, 0.05399184f, 0.03089826f, 0.05327852f, 0.01389698f, 0.01666629f, -0.00679842f, -0.12081788f, 0.02952255f, 0.01384276f, -0.00054027f, -0.07684225f, 0.03831281f, 0.02884319f, -0.02107201f, -0.15999767f, 0.05491090f, 0.03280618f, -0.02957561f, 0.02459365f, -0.04936536f, 0.00573113f, 0.00123551f, -0.01139668f, -0.06741109f, -0.11972249f, 0.00458574f, 0.00559379f, 0.02915014f, -0.16418178f, -0.00287156f, -0.02784588f, -0.07138587f, -0.08164674f, -0.09599400f, 0.00877254f, 0.04215108f, 0.05453249f, -0.02689122f, 0.10939529f, 0.08350197f, -0.01363157f, -0.05985903f, -0.12938296f, -0.02703287f, 0.06541289f, 0.04483433f, 0.01499951f, 0.13312776f, 0.05300064f, -0.05747301f, -0.09034509f, -0.10891399f, -0.04336747f, 0.06082394f, 0.08588144f, -0.08539488f, 0.06013373f, -0.00418427f, -0.04105450f, -0.06016132f, -0.05826549f, -0.03024450f, 0.06923017f, 0.04288722f, -0.00697807f, 0.02407149f, -0.03633687f, -0.09021858f, -0.10423931f, -0.08114948f, 0.00440633f, 0.06042244f, 0.13901643f, 0.08647812f, 0.19167719f, 0.08030390f, 0.00979762f, -0.09345684f, -0.09123032f, 0.01394800f, -0.03958338f, 0.05079907f, 0.06869063f, 0.06796470f, -0.01772572f, 0.01042151f, -0.16005683f, -0.10012556f, 0.03226485f, 0.06895665f, 0.16363196f, 0.06208247f, 0.09984669f, -0.03720446f, 0.01178258f, -0.09198026f, -0.17237095f, 0.00323442f, -0.06486858f, 0.02371144f, 0.02047168f, 0.10773748f, -0.03112350f, -0.02172671f, -0.06842096f, -0.04708510f, 0.12663754f, -0.03529928f, -0.02500511f, -0.02185654f, 0.01327127f, -0.03071343f, 0.06448326f, -0.02950232f, 0.04143313f, 0.20358848f, -0.05697340f, 0.00053433f, -0.00442017f, 0.05827927f, -0.10304058f, 0.04876167f, -0.05192426f, -0.03362214f, 0.07543677f, -0.05867552f, 0.01513148f, -0.01179516f, -0.01785403f, -0.02519645f, 0.10182801f, -0.11821658f, -0.01742461f, 0.06623460f, -0.07816827f, -0.00608129f, -0.03501660f, 0.01416280f, -0.05811937f, 0.16048570f, 0.00421529f, 0.01936119f, 0.23541244f, -0.03709727f, 0.02102232f, -0.00723646f, 0.05380127f, -0.13041898f, -0.00473427f, -0.06011183f, 0.00470585f, 0.24052738f, 0.02930882f, -0.02763449f, -0.05324966f, -0.00031706f, -0.03772441f, 0.07470012f, 0.02468493f, -0.02670685f, 0.08533870f, -0.07131892f, 0.01241858f, 0.00231282f, 0.02163916f, -0.04844932f, 0.20515250f, 0.00634505f, -0.00416100f, 0.07804284f, -0.06619755f, -0.07107280f, -0.09851331f, -0.07119722f, 0.01128407f, 0.19674685f, -0.04319961f, 0.00009385f, 0.22259123f, -0.03038747f, -0.03150792f, -0.00180931f, -0.02912046f, -0.03283918f, 0.12533608f, 0.02409988f, -0.03068426f, 0.09274552f, -0.12982943f, -0.01973899f, -0.03454953f, -0.04645352f, -0.10057050f, 0.03136052f, 0.01483206f, -0.06363741f, 0.05144011f, -0.07875995f, -0.06742685f, 0.01771086f, -0.10184382f, -0.01526040f, 0.23836677f, 0.00992562f, -0.04861739f, -0.00656852f, -0.07876445f, -0.03086379f, -0.03153542f, -0.09179447f, -0.13080959f, 0.04362699f, 0.02163382f, -0.01493748f, 0.04819011f, -0.08224124f, -0.04186504f, 0.04375767f, -0.02918759f, -0.12934743f, -0.07629411f, 0.02112591f, 0.03580419f, 0.09050460f, -0.07409742f, -0.02820434f, -0.04573944f, -0.06680953f, -0.06786027f, 0.04108844f, 0.09367219f, 0.03683874f, -0.06316276f, -0.12095901f, -0.04695820f, -0.00497153f, -0.02646817f, -0.07753929f, 0.03282242f, -0.00832964f, 0.06607114f, 0.04430161f, -0.08947018f, -0.00752382f, -0.04587108f, 0.02876395f, -0.06588798f, 0.12367038f, 0.03964109f, 0.03128797f, 0.11547073f, 0.05666630f, 0.09204114f, 0.01808474f, 0.01860938f, -0.02160734f, 0.06034340f, 0.03647325f, 0.05347255f, 0.06311174f, -0.00459985f, -0.01596406f, 0.03143472f, -0.04409248f, -0.02512990f, 0.09559279f, 0.04760602f, 0.06010541f, 0.07690862f, -0.01082603f, 0.08630450f, -0.01988392f, 0.07377793f, 0.01101513f, 0.12209721f, -0.03964164f, 0.04295103f, 0.03600949f, -0.05075186f, -0.03060690f, 0.05829701f, -0.02190045f, -0.01603481f, 0.09074600f, -0.00037687f, 0.07744967f, 0.07690504f, -0.01133995f, 0.09021314f, 0.21962075f, 0.21361436f, 0.02597334f, 0.09734499f, 0.08787274f, -0.08212815f, -0.16121426f, -0.10738640f, -0.13701255f, 0.05106525f, -0.04705105f, -0.07307115f, -0.10086887f, -0.08941233f, 0.09149156f, 0.06052165f, 0.00748705f, 0.19388339f, 0.17424825f, 0.05238127f, 0.00938571f, 0.10719862f, 0.05969285f, -0.08991310f, -0.13884286f, -0.11425460f, -0.00332614f, 0.05789022f, -0.09345028f, -0.09866739f, -0.18418872f, -0.12007637f, -0.04444163f, -0.03382920f, 0.03470082f, 0.09202360f, -0.02667250f, 0.09966510f, 0.03298095f, 0.00611168f, 0.04402279f, 0.00244075f, -0.06971583f, 0.00856666f, 0.03966585f, -0.08443543f, 0.07180966f, -0.01150654f, -0.06620313f, 0.04181406f, 0.03631763f, -0.01043362f, 0.00634368f, 0.12954883f, -0.03429291f, 0.09935336f, -0.01520640f, -0.05503233f, 0.05749114f, 0.08724609f, -0.00838402f, 0.01964378f, 0.02979225f, -0.06996921f, 0.06216251f, -0.00040732f, -0.06511527f, 0.11828013f, 0.07780413f, -0.07448745f, -0.03280845f, 0.02363497f, 0.06594944f, 0.01383110f, -0.02073825f, -0.02900470f, 0.02313251f, 0.14993552f, -0.03867538f, -0.03308271f, 0.04574119f, -0.03078860f, 0.02107682f, -0.07146915f, -0.12554239f, 0.02947701f, 0.01847818f, -0.02038505f, 0.03369550f, 0.10079644f, 0.03741151f, 0.08557806f, -0.03105729f, -0.10330289f, -0.02177212f, 0.11490655f, -0.01229820f, -0.02648579f, 0.11775443f, -0.01909017f, 0.06206708f, -0.00194526f, -0.02648846f, 0.07140717f, 0.01556338f, -0.11925004f, -0.05098362f, -0.05644641f, 0.12423866f, 0.15739407f, 0.25194250f, 0.01161163f, -0.00386908f, 0.02717758f, -0.02618168f, 0.00129668f, 0.05111488f, 0.02398808f, 0.13575761f, 0.30496783f, 0.31007567f, 0.14157274f, 0.00912257f, -0.14294561f, -0.08791643f, -0.09761954f, 0.02614912f, -0.06035551f, 0.06256478f, -0.09763664f, -0.01894396f, 0.05374833f, -0.06094854f, 0.02959564f, 0.03657380f, 0.09901246f, 0.03076346f, 0.08663071f, 0.13060452f, 0.12019518f, 0.20660459f, 0.18529843f, 0.05063603f, 0.00740385f, 0.02186974f, -0.04098804f, -0.00118549f, 0.10771449f, 0.16492280f, 0.17887430f, 0.32175215f, 0.20066533f, -0.01644843f, -0.01422279f, 0.12869505f, 0.16358192f, -0.01557141f, 0.01738089f, 0.14883249f, 0.05782865f, 0.02260461f, -0.01952158f, 0.06159754f, -0.01889720f, -0.02728768f, 0.03076523f, 0.08632641f, 0.05110924f, 0.14384717f, 0.08680327f, 0.03175224f, 0.00012248f, 0.02197364f, -0.00033389f, -0.13017541f, -0.04627188f, 0.02447563f, 0.16258908f, 0.12096296f, -0.04910552f, -0.00404668f, 0.28672789f, 0.47300769f, 0.16503821f, -0.04559182f, -0.10297841f, -0.04048641f, -0.05764887f, -0.11779028f, -0.20638060f, -0.07862667f, 0.08867226f, 0.02459718f, -0.07640229f, -0.03790470f, 0.00875273f, -0.02254240f, -0.05585640f, 0.06906526f, 0.23521090f, 0.40076438f, 0.21623823f, -0.00061668f, -0.12482277f, -0.12089125f, -0.13199824f, -0.07317499f, -0.10525327f, -0.01598733f, 0.13632903f, 0.13492676f, -0.00821310f, -0.02796430f, 0.05746475f, 0.07936828f, 0.00188877f, -0.13542633f, -0.09568137f, -0.08114823f, 0.03139451f, 0.05853696f, 0.05420789f, -0.03167516f, -0.02223451f, -0.06969854f, -0.13699838f, -0.00221413f, -0.06331685f, -0.01914816f, -0.03471554f, 0.04833941f, 0.01089578f, 0.09277242f, 0.00975869f, -0.01177073f, 0.04987520f, 0.05192855f, 0.25735631f, 0.15159274f, 0.04417421f, 0.02633451f, 0.02187802f, 0.01003945f, -0.02328917f, 0.00693238f, -0.02528633f, 0.07433506f, 0.00765840f, 0.03692597f, -0.06030025f, -0.06971697f, -0.04613630f, -0.12155053f, -0.02659272f, -0.02228282f, 0.04172626f, 0.02847929f, 0.00216756f, 0.00325547f, -0.00632273f, -0.03331637f, -0.14269657f, -0.09929789f, -0.00519574f, 0.03511999f, 0.12341338f, 0.09255277f, 0.05716334f, 0.05934072f, 0.09194064f, -0.06639927f, -0.04783789f, 0.04546397f, 0.14844228f, 0.01377795f, -0.01438089f, 0.08082539f, -0.00445815f, 0.07047576f, -0.03601759f, -0.00326744f, 0.07909167f, 0.17376120f, 0.11520382f, -0.11753351f, 0.02676783f, 0.03165923f, 0.06689873f, -0.11370735f, -0.03319100f, 0.00031876f, 0.03952449f, 0.09152398f, -0.08967950f, -0.03543836f, -0.02567889f, 0.03521001f, -0.09434917f, -0.10770399f, -0.12000975f, -0.00424642f, 0.00212454f, 0.05285718f, 0.05393134f, 0.03865501f, 0.11752654f, 0.00980432f, 0.04402908f, 0.05227639f, 0.21250208f, 0.12195572f, 0.01813186f, 0.02551362f, 0.02311620f, 0.04022338f, 0.05650661f, -0.07526952f, -0.06478292f, 0.04838710f, 0.07044793f, -0.03578930f, -0.00279667f, 0.00045215f, 0.10461357f, -0.07651924f, -0.04727585f, -0.09648691f, 0.04092519f, 0.04874024f, -0.01212918f, -0.02523023f, -0.05134040f, 0.04735219f, -0.04232176f, -0.03795357f, -0.06805790f, -0.04867241f, -0.09103389f, 0.10040036f, 0.03077996f, 0.01381365f, 0.02064436f, 0.07970981f, 0.01425486f, 0.00776379f, 0.07929828f, 0.15592486f, 0.19225316f, 0.09653739f, -0.03135792f, 0.03298209f, 0.06121983f, 0.06002913f, 0.00204095f, 0.07995912f, 0.14935336f, 0.09761392f, 0.09042480f, 0.01239649f, 0.04167225f, -0.06356078f, -0.09788739f, -0.08327811f, -0.01741114f, -0.08341862f, 0.05538291f, 0.03614012f, -0.07978211f, 0.02392992f, 0.02879704f, -0.06077531f, -0.12287057f, -0.00100944f, 0.01242626f, 0.20135017f, 0.16298437f, -0.02213512f, 0.11312697f, 0.04824551f, -0.00225256f, -0.04321414f, 0.04687848f, 0.11817231f, 0.08544870f, 0.07499681f, -0.02842437f, 0.08090107f, 0.06469367f, -0.03177627f, -0.10989808f, -0.02963185f, 0.05512539f, 0.06467012f, 0.11837558f, -0.08636627f, -0.02731784f, 0.04488910f, 0.00672770f, -0.04447897f, 0.12559956f, 0.05688091f, 0.02758328f, 0.07570382f, -0.09044038f, -0.04884742f, -0.04933984f, -0.07790067f, -0.02777399f, 0.13356830f, 0.04606266f, 0.07577106f, 0.05958395f, -0.08676675f, -0.00392092f, 0.03255095f, -0.03161076f, -0.07972698f, 0.03993317f, 0.06202025f, 0.03145055f, 0.00493697f, -0.13105536f, -0.05656229f, -0.03107150f, -0.07929187f, -0.06936998f, 0.06891852f, 0.03068066f, -0.04123224f, 0.05457777f, -0.03923945f, -0.02001868f, 0.01308929f, -0.05756735f, -0.02086291f, 0.22410977f, -0.01635973f, 0.03539555f, 0.02088146f, -0.06258141f, -0.00645860f, -0.03080365f, -0.02646933f, -0.03401057f, 0.04211707f, 0.02280802f, 0.00050276f, -0.01310374f, -0.06227501f, -0.01144003f, -0.00373236f, -0.07193143f, -0.04065586f, 0.07362510f, -0.02570302f, 0.04376765f, -0.12079733f, -0.09165417f, -0.05294072f, 0.03391567f, -0.01752590f, -0.06620767f, -0.00473611f, -0.00909861f, 0.07584234f, 0.12178880f, -0.10973383f, -0.04237178f, -0.04937171f, -0.01570138f, -0.04258844f, 0.03966948f, 0.00448837f, 0.11433693f, 0.16699387f, -0.02699916f, -0.00467927f, -0.01592597f, 0.00999871f, 0.00370731f, 0.15603456f, 0.04038297f, 0.00117885f, -0.11109817f, -0.16969387f, -0.04760998f, 0.00318629f, -0.05473995f, -0.03718256f, -0.00285478f, -0.05916800f, -0.00851040f, -0.07761902f, -0.13204363f, -0.04826541f, -0.05158439f, -0.11438823f, -0.05067594f, 0.01520052f, -0.01937737f, 0.09202506f, 0.17205036f, 0.12293913f, 0.08985617f, 0.06305674f, 0.18034808f, 0.15034667f, 0.27291506f, 0.10651634f, 0.06816939f, 0.12998729f, 0.03191341f, 0.01942793f, 0.07372946f, 0.07660253f, 0.15465908f, 0.13725378f, 0.08444093f, 0.03120447f, 0.04869642f, -0.00517585f, 0.00563045f, 0.02969653f, -0.02980796f, 0.00291610f, 0.11223097f, -0.01683288f, -0.06600036f, -0.05405377f, -0.08975101f, -0.10585102f, -0.01627420f, -0.07360179f, -0.02132113f, -0.02749483f, -0.04803162f, 0.12932087f, 0.16223457f, 0.16429324f, 0.24547130f, 0.25686076f, 0.05192611f, 0.07680058f, 0.13704111f, 0.11878061f, -0.02281259f, -0.11215034f, -0.05513098f, 0.07403090f, 0.05206315f, -0.12479066f, -0.02936605f, -0.14707217f, -0.07246971f, 0.03677058f, 0.08965594f, 0.08685893f, 0.20203110f, 0.05398323f, 0.02132746f, 0.00834007f, -0.00655706f, 0.03546836f, -0.12530037f, -0.13771773f, -0.05529463f, 0.00077451f, -0.01593161f, -0.11293926f, -0.09823421f, -0.18250394f, -0.11239359f, 0.00877928f, -0.01447410f, 0.01019818f, 0.11731300f, -0.09465579f, 0.03783100f, 0.04955132f, -0.00790123f, 0.05517091f, 0.03640712f, -0.03885457f, 0.01778435f, 0.07055699f, -0.07001548f, 0.07150275f, -0.01405703f, -0.05647650f, 0.07206064f, -0.01748137f, -0.02088665f, 0.00211279f, 0.19224770f, 0.01483833f, 0.06821041f, 0.05640741f, 0.06853651f, 0.09365608f, -0.02953630f, -0.04635301f, 0.00638507f, 0.09131537f, -0.03577773f, 0.04666570f, 0.05456919f, -0.01982532f, 0.05511567f, 0.05126690f, -0.04022847f, 0.00140293f, 0.05972781f, -0.06422983f, -0.01990930f, -0.02542812f, -0.07631298f, 0.03459851f, 0.06170678f, -0.02404884f, -0.00460791f, 0.09110413f, -0.01566346f, -0.02573891f, 0.04242319f, 0.02001830f, 0.07744758f, 0.02642985f, -0.10125594f, -0.02697371f, 0.05931539f, 0.02605295f, 0.05193194f, 0.01606822f, -0.00718781f, 0.07398010f, 0.01074372f, -0.02662525f, -0.01410661f, 0.10701500f, -0.01071612f, 0.09762820f, 0.03376242f, -0.00266970f, 0.07015234f, 0.01546659f, -0.12456782f, -0.05703831f, 0.04882844f, 0.06592278f, -0.10215403f, 0.05399718f, -0.05928804f, 0.02195458f, 0.07278533f, -0.05518551f, 0.00148812f, 0.01682375f, -0.01769302f, 0.00638791f, 0.00043305f, -0.06460557f, 0.04293777f, 0.08328449f, -0.07016292f, -0.04008625f, 0.01607154f, -0.01419184f, -0.07477041f, -0.00841369f, -0.06294272f, 0.11906829f, 0.05415695f, -0.03234858f, -0.01011530f, 0.01544627f, 0.00077004f, -0.02257728f, -0.03827853f, -0.05249288f, 0.09512976f, 0.02670269f, -0.06365975f, -0.06992068f, -0.03178315f, 0.10246229f, 0.05899500f, -0.01232802f, -0.00172888f, 0.01943260f, 0.03947857f, 0.08425832f, 0.01136940f, -0.00216261f, -0.01094648f, 0.02447063f, -0.06291520f, -0.06314138f, -0.00405855f, -0.00257458f, -0.06426260f, -0.04267001f, 0.00978777f, 0.03945051f, 0.05987449f, -0.03653191f, -0.01043037f, 0.06699710f, 0.03869988f, 0.05986310f, 0.04199718f, 0.04177591f, -0.03027069f, 0.00908537f, -0.02423875f, 0.02223419f, 0.03294752f, 0.02895455f, 0.06020164f, 0.00449395f, -0.00555740f, 0.00666782f, 0.00283647f, 0.10145228f, 0.08385781f, -0.00199564f, -0.08128358f, -0.02919289f, -0.00068403f, 0.05563250f, -0.03263485f, -0.01534677f, 0.00001040f, 0.14009304f, -0.02075094f, -0.00234205f, 0.02387686f, 0.02211221f, -0.01842640f, -0.00876941f, -0.03742717f, -0.04657061f, 0.00310528f, -0.02310651f, -0.09426490f, -0.02353563f, 0.00224644f, 0.03148432f, 0.01338611f, -0.06806378f, -0.00966875f, -0.00627354f, 0.02491370f, -0.05985156f, -0.00141190f, 0.05457696f, 0.03191011f, 0.00061253f, -0.04820896f, -0.05725765f, 0.17926725f, 0.16266477f, -0.10668666f, -0.05741468f, 0.01311288f, 0.02741323f, -0.01258803f, -0.01280302f, -0.06899014f, 0.08806330f, -0.00697358f, -0.04942954f, 0.01372094f, 0.05308352f, 0.01581815f, 0.05620795f, -0.10314704f, 0.03270067f, 0.11820683f, 0.13119177f, -0.02546691f, -0.07603096f, -0.06228166f, 0.02207847f, 0.02778905f, -0.08769730f, -0.10220946f, 0.01697143f, -0.02778685f, -0.08459762f, 0.00943028f, 0.09251255f, 0.08924549f, -0.00391860f, -0.04222243f, 0.09615519f, 0.12348043f, 0.07523759f, -0.09949198f, -0.08243528f, -0.03442745f, 0.00986509f, -0.02643799f, -0.04590241f, -0.01063566f, -0.02308777f, 0.02109999f, 0.04256762f, -0.00081106f, -0.01721874f, 0.02392387f, 0.05029749f, -0.04961122f, -0.02257588f, 0.08179309f, 0.07140425f, -0.00673075f, 0.03204660f, -0.00615491f, -0.01192862f, -0.01222855f, -0.05579372f, -0.04600772f, -0.00756298f, 0.02695104f, -0.00268784f, 0.04930915f, 0.04967819f, 0.09870021f, 0.09018282f, -0.02285757f, -0.03889151f, 0.08622626f, 0.10572799f, 0.01086582f, -0.00692002f, 0.01113239f, -0.01466085f, -0.00169321f, -0.07248947f, -0.03150058f, 0.05662395f, 0.07090907f, 0.02438699f, 0.10858547f, 0.00138480f, 0.04815994f, 0.04536665f, -0.00201392f, 0.01750649f, 0.05679186f, 0.03588928f, -0.04078550f, 0.00180806f, -0.11629055f, -0.02308436f, -0.03045140f, -0.10762823f, 0.02407582f, -0.02904625f, 0.02431415f, -0.01359889f, 0.08649004f, 0.08328352f, 0.06245186f, 0.04488518f, 0.07722312f, 0.08792400f, 0.09842309f, 0.17965319f, 0.10478604f, 0.04219538f, 0.04514331f, 0.07529080f, 0.06641500f, 0.08232303f, 0.01253750f, 0.06931547f, 0.07737581f, 0.08745403f, 0.06742893f, -0.03632117f, 0.00623473f, -0.06779292f, 0.00633573f, 0.06414778f, -0.01689558f, 0.07971027f, -0.06400506f, 0.01337344f, 0.00372533f, -0.03713677f, -0.01628285f, 0.00372226f, -0.04999543f, -0.02999665f, -0.01380400f, 0.09878890f, 0.10341703f, 0.12005604f, 0.19629938f, 0.16662853f, 0.11323071f, 0.02783217f, 0.09526659f, 0.07772791f, 0.01358648f, -0.05701669f, 0.02117923f, 0.08604493f, 0.08590570f, 0.06045859f, 0.03530448f, 0.01122127f, -0.03068958f, 0.04951508f, 0.00312444f, -0.01454423f, 0.01510179f, 0.05308136f, 0.01751884f, -0.01711460f, 0.02492954f, 0.03629026f, 0.00955719f, -0.07434170f, -0.07053896f, -0.06576537f, -0.01921543f, -0.06559519f, -0.00476161f, 0.02425415f, -0.03815187f, -0.01445139f, -0.03209167f, -0.02406338f, 0.12129918f, 0.10397682f, 0.06219499f, 0.00315374f, 0.10678074f, 0.07035769f, -0.04378671f, -0.05481448f, -0.01678143f, 0.07035962f, 0.04519099f, 0.03085512f, -0.00209396f, 0.10865364f, 0.01307177f, 0.00617688f, 0.01578572f, -0.02103796f, 0.02774128f, 0.00550925f, 0.03520722f, -0.01583245f, 0.01460111f, -0.00495283f, -0.05720889f, 0.00051548f, -0.00717017f, -0.01231430f, -0.03036904f, -0.03856851f, -0.09312072f, 0.01621700f, -0.06420529f, 0.00594402f, 0.02687971f, 0.03612623f, 0.09865613f, 0.09935468f, 0.06166869f, 0.04276267f, 0.11178316f, 0.10526737f, 0.03443849f, -0.06289048f, 0.03176697f, 0.10728198f, 0.08989346f, 0.06583374f, 0.00402474f, 0.05350331f, 0.06175857f, -0.05602473f, 0.00043313f, -0.00317887f, 0.05195401f, -0.00187452f, 0.00254092f, -0.00806638f, 0.06737163f, -0.02426367f, -0.07654529f, -0.02136170f, -0.05431660f, -0.02692314f, -0.07002737f, 0.00537109f, -0.07869008f, -0.01507321f, -0.03002469f, 0.00874386f, -0.03344424f, -0.00035723f, 0.15123105f, 0.09171680f, 0.04628133f, 0.01371532f, 0.05908495f, 0.08646357f, -0.02074429f, 0.00096338f, 0.01280852f, 0.11336023f, 0.06669608f, 0.04512350f, 0.03428375f, 0.06689270f, 0.02233004f, -0.02144820f, -0.01387122f, -0.03359580f, 0.02772809f, 0.00613166f, 0.00906537f, -0.02130049f, 0.00903039f, 0.04904733f, -0.07697379f, -0.05207606f, -0.00756545f, -0.07777117f, -0.02716734f, -0.03044947f, -0.03458166f, -0.03150161f, -0.03307513f, 0.03216114f, 0.13795457f, 0.12096051f, 0.15010481f, 0.15890164f, 0.11320857f, 0.08437110f, 0.11145695f, 0.02420640f, 0.00234613f, 0.07668296f, 0.01360975f, 0.03983148f, 0.04842614f, 0.03624843f, 0.05118165f, 0.05930717f, 0.02484152f, 0.00777658f, 0.03962625f, 0.01319798f, -0.03945669f, -0.06494773f, 0.00966004f, -0.00506354f, -0.01307226f, -0.03876081f, -0.02048741f, -0.05020925f, -0.05776278f, -0.09874771f, -0.06042099f, -0.09686722f, -0.08944936f, -0.08357095f, -0.08394368f, 0.03536021f, 0.10318908f, 0.03160422f, 0.14618893f, 0.02496346f, 0.06266686f, 0.08312149f, 0.00159774f, 0.00056455f, -0.01096443f, -0.01916946f, -0.01895691f, 0.03187361f, 0.00016936f, 0.01209978f, -0.03067785f, -0.16345563f, -0.05403291f, 0.05077079f, 0.03664935f, -0.01247741f, 0.06128871f, -0.05300566f, -0.05125454f, 0.03459055f, -0.04873117f, -0.01727398f, 0.08300597f, -0.03169001f, -0.08882238f, -0.05437348f, -0.05580830f, -0.00301252f, 0.00692653f, -0.07300256f, 0.05014007f, 0.04072524f, -0.00243933f, -0.00254752f, 0.06335841f, -0.03383624f, 0.02921929f, -0.00918848f, -0.05156612f, 0.06692978f, -0.00147811f, -0.11453002f, -0.04513944f, 0.01221492f, -0.02290678f, 0.07138535f, 0.04375181f, -0.09848911f, 0.04316647f, 0.07368413f, -0.01527978f, -0.03651018f, 0.02953764f, -0.06461046f, 0.09426832f, 0.04763540f, -0.01803348f, 0.06520470f, 0.08736709f, -0.14159931f, -0.07150494f, 0.04122389f, 0.01349801f, 0.08989650f, 0.01901345f, -0.14820179f, 0.09746036f, 0.04858420f, -0.06412746f, -0.04811560f, 0.02917172f, 0.10100805f, 0.08578257f, -0.01264069f, -0.02054251f, 0.07974006f, 0.01978294f, -0.03350524f, 0.04870400f, 0.08709350f, 0.08698494f, 0.15323304f, 0.00245060f, -0.04334929f, 0.06902597f, 0.09440794f, -0.03013231f, -0.02442016f, 0.01666784f, 0.14685592f, 0.02316693f, -0.09042187f, -0.10169069f, 0.07683369f, 0.07439947f, -0.01365741f, 0.05434036f, 0.14685532f, 0.08309903f, 0.07936610f, 0.03850328f, -0.01266610f, 0.11190132f, 0.05930893f, -0.01303845f, -0.01135209f, 0.01263788f, 0.01525760f, -0.02592500f, -0.00396471f, -0.08252745f, 0.04173068f, 0.02738098f, 0.00316953f, -0.03129131f, 0.09641257f, 0.07435547f, 0.03987011f, 0.00578019f, -0.02038733f, 0.07665873f, 0.10597556f, -0.00003596f, 0.00312059f, 0.02728482f, 0.04241881f, -0.03899328f, 0.05196139f, -0.05751556f, 0.06510739f, 0.10705996f, 0.06977969f, 0.00578770f, 0.04543766f, -0.00177436f, -0.01542040f, 0.03889017f, 0.02149616f, 0.04956149f, -0.06178464f, -0.06147530f, -0.02109074f, 0.04343963f, -0.00465696f, 0.00195381f, -0.00428284f, -0.02008324f, 0.00906641f, -0.03174205f, 0.04883545f, 0.05725868f, 0.03571314f, 0.01775419f, -0.01121360f, -0.00888351f, -0.07551778f, 0.00755293f, -0.00191575f, -0.02084952f, -0.00859538f, 0.05327542f, 0.07128105f, 0.03767267f, 0.02745518f, 0.02297385f, -0.03946099f, 0.06946340f, 0.00968116f, 0.01859126f, 0.08187267f, 0.08931219f, -0.00141674f, 0.04132653f, -0.00400427f, 0.04821702f, -0.05895619f, 0.04778179f, 0.05745847f, -0.06118201f, -0.00771203f, -0.02689872f, 0.00975896f, -0.00496666f, -0.00063108f, -0.07311285f, -0.01020161f, 0.06865984f, -0.01289370f, 0.01657782f, -0.04137051f, -0.04479357f, -0.04852494f, -0.03356623f, 0.04521124f, 0.01127183f, 0.01454908f, 0.00993851f, 0.02755057f, -0.07925580f, 0.01168163f, 0.07320482f, 0.07325880f, 0.02692076f, -0.06611881f, 0.01770247f, -0.04345263f, -0.03690769f, -0.08867888f, -0.00781112f, 0.02864720f, -0.03249141f, -0.00091247f, 0.11560919f, 0.10926908f, 0.06156125f, 0.11335696f, -0.01635325f, 0.00975981f, 0.04051253f, 0.03192897f, 0.00665306f, 0.02017907f, -0.02335453f, -0.00955298f, 0.05164771f, -0.05587854f, -0.05951431f, -0.01962328f, -0.02742696f, -0.02933722f, 0.00222812f, 0.01079894f, 0.02060383f, 0.05076305f, -0.00353417f, 0.01657521f, 0.05925264f, -0.00419127f, -0.03125946f, -0.03375033f, -0.06776831f, -0.07945322f, -0.00958221f, -0.07205689f, -0.03476785f, -0.02456305f, -0.05404102f, -0.03371027f, 0.05723449f, -0.02862993f, 0.01213271f, 0.13826783f, 0.06715247f, -0.03253553f, 0.02029418f, 0.07692305f, 0.01067394f, 0.06183939f, -0.02441886f, 0.00736742f, -0.01906579f, -0.03355664f, -0.05490618f, -0.02140820f, 0.00089183f, -0.00249228f, 0.01455382f, 0.00055321f, 0.00777367f, 0.05001228f, 0.00997055f, -0.05285150f, -0.04705689f, -0.03790681f, -0.04384088f, -0.04317218f, -0.05685417f, -0.07389181f, -0.06686842f, -0.03165647f, -0.03232950f, -0.06776566f, -0.06517301f, 0.05425164f, 0.13190722f, 0.00779245f, 0.06247890f, 0.10288361f, 0.03703718f, 0.05326220f, 0.03917966f, 0.05446563f, -0.03372449f, 0.05665889f, -0.08062032f, 0.03592828f, -0.02031070f, -0.02189794f, -0.00436992f, -0.07077914f, -0.02840801f, 0.07176514f, 0.02407468f, -0.04394663f, -0.00271569f, 0.07017033f, 0.02334619f, 0.04745530f, -0.01235851f, 0.01167196f, 0.00021723f, 0.04681873f, -0.10496349f, 0.02301458f, 0.10646504f, 0.03498973f, -0.03981096f, -0.05589848f, -0.03891961f, 0.03898562f, 0.06786369f, -0.02109518f, 0.08242611f, 0.05344888f, 0.07213008f, 0.02331763f, 0.00874232f, 0.06453672f, -0.05522647f, 0.03578067f, -0.04002948f, -0.07933517f, -0.10422286f, 0.01880783f, -0.03811129f, 0.01210908f, -0.00289408f, 0.03241474f, 0.08560010f, -0.02649107f, 0.06242505f, 0.08655133f, 0.10411413f, -0.01357521f, 0.07422342f, -0.02112751f, -0.05283977f, 0.07288672f, -0.05068916f, -0.04525795f, -0.05931441f, -0.05872113f, -0.02308374f, -0.03603240f, -0.02445908f, 0.03506242f, 0.02391927f, -0.01252849f, -0.04490556f, -0.01266028f, 0.00443981f, 0.06219439f, 0.01198670f, 0.03142727f, -0.00412226f, -0.05445879f, -0.00648362f, -0.02188753f, -0.05216705f, -0.02424087f, 0.05212284f, -0.01365820f, -0.12786656f, -0.03672874f, 0.03749405f, 0.03842020f, 0.04148532f, 0.02735533f, 0.01749949f, 0.06188807f, -0.02013487f, -0.01354105f, -0.02978138f, -0.02719969f, 0.02723540f, 0.07880572f, 0.05705519f, 0.01621853f, -0.00936404f, -0.00962505f, -0.01398413f, 0.02391578f, 0.04039878f, 0.03749137f, 0.06939799f, -0.02627328f, 0.12146418f, 0.05300519f, 0.00862534f, 0.03894858f, -0.01731469f, -0.00480157f, -0.01984027f, 0.00564418f, -0.10888943f, 0.02848656f, -0.05953423f, -0.03749741f, -0.04943031f, 0.04656687f, 0.00837898f, -0.02965496f, 0.02993908f, -0.01214411f, 0.02434639f, -0.03708838f, 0.00598565f, -0.00911937f, 0.00833348f, -0.03578941f, -0.08002330f, 0.00870846f, -0.00188986f, -0.08317931f, -0.07605060f, -0.04528422f, -0.01079808f, 0.01491741f, 0.05089770f, -0.00701233f, 0.04355436f, -0.01802227f, 0.09318885f, 0.06760123f, 0.06766933f, -0.00668864f, 0.00042157f, 0.02667601f, 0.01155076f, -0.03163995f, -0.06216601f, 0.09711801f, 0.08189553f, 0.04368705f, -0.05683333f, 0.01854320f, 0.01598285f, -0.09516472f, 0.04519751f, 0.08831177f, 0.03816774f, 0.04147697f, 0.00197807f, -0.00462748f, 0.01613879f, 0.05487724f, -0.01640672f, -0.01510044f, -0.00896254f, 0.03711082f, 0.02652455f, -0.02504985f, 0.01939070f, 0.03747070f, 0.00615903f, 0.02593753f, 0.03941788f, -0.02318452f, 0.07138584f, -0.01606880f, -0.01301007f, 0.01146650f, -0.03394867f, -0.04826492f, 0.01262555f, -0.00115329f, 0.00853636f, 0.01610529f, -0.00134145f, -0.09349508f, -0.10308200f, 0.10244647f, 0.05515219f, 0.05122533f, -0.00634470f, -0.04741848f, 0.03121251f, 0.04938342f, 0.00523211f, 0.05314858f, -0.02077827f, -0.01375024f, -0.01954797f, 0.01558023f, -0.08348758f, -0.05784082f, -0.03774295f, -0.03223610f, 0.03237992f, 0.01192684f, 0.04758885f, 0.05749580f, 0.02792954f, 0.01860661f, 0.12022117f, 0.08741776f, 0.04690545f, 0.03170096f, -0.03822266f, 0.00904343f, 0.04296792f, -0.06898976f, 0.05424230f, -0.03459317f, -0.01615975f, 0.02970295f, 0.01881135f, 0.02464534f, 0.02448340f, 0.01123997f, 0.02393870f, -0.02575149f, -0.00419762f, 0.03101368f, 0.03329124f, 0.04404832f, -0.09025499f, 0.03032819f, -0.03711392f, -0.07338585f, 0.04121698f, -0.08254785f, -0.05096268f, -0.03830166f, -0.07285463f, -0.01798379f, 0.01374313f, 0.00068222f, 0.00929154f, 0.02740952f, 0.00143656f, -0.00177032f, 0.05602658f, -0.04306196f, 0.00969761f, -0.06430114f, -0.04795189f, -0.02285127f, 0.07593319f, 0.00905392f, 0.00766384f, -0.00234519f, -0.00410634f, 0.00712007f, 0.12822742f, -0.02050950f, -0.04662718f, 0.05418217f, 0.06966794f, 0.02029632f, 0.09517912f, -0.00726917f, -0.00207721f, -0.04211664f, -0.05970208f, -0.00653556f, 0.02445142f, 0.01291477f, 0.01172366f, 0.01131001f, 0.02488475f, 0.02754492f, -0.01810695f, 0.00311197f, 0.07923090f, -0.05468553f, 0.04732427f, 0.04482146f, 0.00988839f, 0.03004625f, 0.09067549f, -0.03421581f, 0.05377790f, 0.05153683f, -0.10079493f, 0.02053871f, 0.01682475f, -0.12681678f, 0.02260341f, 0.05245870f, 0.00363150f, 0.02056721f, 0.13073415f, -0.03787893f, 0.07482434f, -0.02225227f, 0.03841474f, 0.04720273f, 0.09699413f, -0.04319270f, 0.03603907f, -0.00099184f, -0.09721787f, 0.07524490f, -0.01543837f, -0.09541203f, 0.01514664f, -8.0554609620f };
	return std::vector<float>(detector, detector + sizeof(detector) / sizeof(detector[0]));
}

std::vector<float> HOGDescriptor_Mod::HOG_Optimal_48_120()
{
	static const float detector[] = { 0.10724598f, -0.05959907f, -0.01903786f, 0.08997558f, 0.09998365f, 0.03895523f, -0.02590491f, -0.01413603f, 0.10629999f, 0.10593224f, -0.03408622f, 0.01933394f, 0.04234937f, 0.03420821f, 0.07723962f, -0.03509188f, 0.00423142f, 0.12083490f, 0.06153451f, -0.04316871f, -0.02437665f, 0.06644530f, 0.07600208f, 0.05937478f, -0.00717047f, -0.00712439f, 0.06295265f, 0.04753391f, -0.05681077f, 0.00320409f, 0.00786580f, 0.02577461f, 0.08963341f, -0.00264137f, -0.01325855f, -0.00978177f, 0.05069628f, -0.06038514f, -0.01991284f, 0.00466671f, 0.06622727f, 0.03987465f, -0.03158823f, -0.02997270f, 0.07759066f, 0.03522353f, -0.01491681f, 0.00487679f, 0.04708212f, 0.02560531f, 0.08365740f, 0.02189726f, 0.00130515f, 0.05742098f, 0.03625659f, -0.03322664f, 0.00867131f, 0.02416795f, 0.06964408f, 0.05508366f, -0.01428307f, -0.03735764f, 0.05735240f, -0.00634656f, -0.05110590f, -0.05197045f, 0.03249989f, 0.01768555f, 0.04482667f, -0.02744044f, -0.05397707f, 0.05595112f, 0.02664052f, -0.04000114f, -0.00978581f, -0.00797793f, 0.01677964f, 0.03038455f, 0.04178916f, 0.01477252f, 0.03674034f, -0.01332583f, -0.03538326f, -0.03127637f, -0.03241057f, 0.02055151f, 0.04676702f, 0.01756332f, 0.01685529f, 0.01701364f, -0.00949670f, -0.05397360f, -0.02530884f, 0.02836168f, 0.01995947f, -0.00173769f, 0.00484137f, 0.02143099f, 0.05407029f, -0.04119340f, 0.00102232f, 0.05973569f, 0.01667265f, -0.03601054f, 0.01131755f, -0.00699247f, -0.00265005f, 0.00616447f, 0.01991799f, 0.03444241f, -0.01350414f, -0.03438223f, 0.05993613f, 0.03060229f, 0.00963667f, 0.01823579f, -0.01226081f, 0.03451821f, 0.07101371f, -0.05176434f, -0.02963907f, 0.08107141f, 0.02026949f, -0.02864067f, -0.01740499f, -0.04003492f, 0.06856248f, 0.18481541f, 0.06453749f, -0.03470277f, -0.04435909f, 0.00759188f, -0.02327124f, -0.04563728f, -0.02972068f, 0.03643362f, 0.11901934f, -0.07639214f, -0.07394440f, -0.06217054f, -0.02925396f, -0.03850910f, -0.05050232f, -0.05344409f, 0.08208390f, 0.08334386f, -0.05257784f, -0.00719427f, 0.08207347f, 0.01918275f, -0.02924271f, 0.00161769f, 0.01625467f, 0.03751673f, 0.08271780f, 0.00225076f, -0.01826169f, 0.09867277f, -0.00511061f, -0.02736469f, 0.00306084f, -0.01367707f, 0.04945173f, 0.12823023f, -0.03980599f, -0.01163439f, 0.01251907f, 0.00440074f, -0.00497300f, 0.00090539f, -0.02103652f, 0.00158181f, 0.11729253f, 0.00050927f, -0.04770110f, -0.00895714f, 0.00628746f, -0.00514138f, -0.02233053f, -0.00621011f, 0.02081908f, 0.03631954f, -0.05934174f, 0.01297612f, 0.09411004f, 0.00362916f, 0.01837150f, 0.09115352f, 0.01203336f, -0.00246667f, -0.02685052f, -0.09117413f, -0.04107060f, 0.06553227f, -0.05424287f, -0.04625160f, 0.05653217f, 0.03157072f, 0.02977675f, 0.09399695f, 0.00289802f, 0.00435972f, 0.00946808f, 0.01704093f, 0.04420315f, 0.01933214f, -0.01897288f, 0.00735464f, 0.07092317f, -0.07313304f, -0.05565627f, -0.01158511f, -0.01669848f, 0.00712737f, 0.01016093f, 0.00794516f, 0.02596224f, 0.00784844f, 0.01084305f, 0.03601325f, 0.06742924f, -0.02150021f, -0.05295098f, 0.02646375f, 0.06041063f, -0.00448845f, -0.01924650f, -0.03706663f, 0.00511147f, 0.03542694f, -0.03318001f, -0.02943791f, -0.03138807f, -0.01160453f, 0.06639596f, 0.09426854f, 0.01199107f, -0.00769861f, 0.01808293f, 0.01561124f, 0.04559132f, 0.05599248f, 0.02073629f, 0.04512814f, 0.07828380f, -0.00347025f, -0.01373849f, 0.02358075f, 0.03102553f, 0.01045901f, 0.03417076f, 0.02608936f, 0.00701159f, 0.01242175f, -0.02916500f, 0.01706172f, 0.04108985f, 0.05004413f, 0.04985512f, 0.04979578f, 0.03226667f, -0.03807435f, -0.04316471f, -0.06584239f, -0.05803214f, -0.01783984f, -0.02573101f, -0.04105658f, -0.01023608f, -0.05039654f, 0.09525549f, 0.07513710f, 0.00216779f, 0.05640208f, 0.09003124f, 0.13422399f, 0.08271651f, 0.10324018f, 0.05847085f, 0.01945782f, -0.02219159f, -0.02968370f, 0.01866560f, 0.07081342f, 0.07344398f, 0.03279619f, 0.03819146f, 0.04604509f, -0.03288713f, 0.02130459f, -0.00748000f, 0.03765573f, 0.03595558f, 0.02004241f, -0.03894007f, -0.01314434f, -0.01043319f, -0.04755989f, -0.03864519f, -0.04688190f, -0.02800307f, -0.03370639f, -0.02566728f, -0.03398991f, 0.01891920f, -0.06129218f, 0.02728968f, 0.03932308f, 0.02659000f, 0.07203520f, 0.09480432f, 0.12223895f, 0.02631629f, 0.01779206f, 0.03663005f, -0.01181916f, 0.03534526f, -0.03553119f, -0.00024212f, 0.04441102f, 0.02358014f, -0.04363871f, -0.02132649f, -0.02713996f, -0.05868517f, -0.01567017f, -0.03900575f, -0.03084715f, 0.00205947f, 0.02690097f, 0.00613579f, 0.01485016f, -0.01402443f, -0.05924838f, 0.00794516f, -0.04635981f, -0.02596745f, 0.00247636f, -0.01200041f, -0.04140674f, -0.00610670f, -0.07697088f, 0.05092302f, 0.08803406f, -0.02882360f, 0.02750414f, 0.06586990f, 0.05279708f, -0.01342588f, -0.02569440f, -0.00338131f, -0.00595578f, 0.03255532f, -0.02847530f, 0.03955443f, 0.09073319f, 0.04071950f, -0.03019901f, -0.02023172f, 0.00653210f, -0.02350045f, 0.01431905f, -0.02269610f, 0.04826017f, -0.00057046f, 0.01285829f, -0.03308598f, 0.00865020f, -0.02233215f, -0.03343977f, -0.01425623f, -0.03456066f, 0.01222596f, 0.00338375f, -0.02147873f, -0.06763352f, -0.03682174f, -0.05214999f, 0.02499079f, 0.05173657f, -0.01799371f, 0.09396866f, 0.08315813f, 0.09350803f, 0.01539242f, -0.03053043f, 0.03114355f, 0.00564423f, 0.01841539f, -0.03564091f, -0.00162048f, 0.08870343f, 0.06994213f, -0.04401099f, -0.06310677f, 0.03554273f, -0.02045684f, 0.00440969f, -0.00779431f, 0.01513571f, -0.02207567f, -0.01466828f, -0.03692588f, 0.00725702f, -0.00283461f, -0.07002103f, -0.02193050f, -0.03700506f, -0.01723334f, -0.06946168f, -0.07182955f, -0.02451983f, 0.01071195f, -0.04068979f, -0.00654595f, 0.08185032f, -0.01600419f, 0.04239967f, 0.07459192f, 0.10258400f, 0.01464011f, 0.04069209f, 0.00554301f, -0.01187318f, 0.06487007f, 0.03979476f, 0.06415261f, 0.09661663f, 0.04790271f, -0.00010661f, 0.03110680f, 0.00956106f, -0.03649997f, 0.00462135f, 0.00052192f, 0.03431069f, -0.02516550f, 0.01281585f, 0.06257622f, 0.05210619f, -0.00624150f, -0.03939710f, -0.03135704f, -0.02273539f, -0.01317623f, -0.07918194f, -0.04862542f, 0.00090704f, -0.00842111f, -0.00548176f, 0.02674218f, 0.09610031f, 0.08971589f, 0.11499822f, 0.08217855f, 0.10486243f, 0.11778926f, 0.08267295f, -0.00979330f, -0.05195889f, -0.01739670f, 0.02133650f, 0.06433904f, 0.06456758f, 0.10147920f, 0.05553225f, 0.03221763f, -0.01645501f, 0.02401448f, 0.00857705f, 0.00097601f, 0.07325117f, -0.03609977f, 0.01917898f, -0.00433358f, 0.03153728f, 0.09347114f, 0.08610722f, -0.02855551f, 0.01857987f, 0.09523048f, -0.03166406f, -0.01980329f, -0.07960090f, -0.03231050f, 0.08208942f, -0.01689678f, -0.05365824f, -0.00084659f, 0.07187790f, 0.01251256f, 0.06845815f, 0.02109268f, 0.01890268f, 0.04050225f, 0.03122579f, -0.03217875f, 0.01456121f, 0.05151735f, -0.04325384f, 0.04353865f, -0.04877373f, -0.05655584f, 0.01528860f, 0.03167520f, -0.08335466f, -0.09036249f, 0.00550667f, 0.11714760f, -0.03046292f, -0.04472455f, -0.04698398f, 0.08733182f, 0.08381744f, 0.00152119f, 0.06245542f, 0.03234304f, 0.09051480f, 0.03567774f, -0.00323515f, -0.04577609f, 0.07144936f, 0.05272936f, -0.11985728f, -0.08474910f, 0.00020978f, 0.06693579f, -0.02278223f, -0.07574118f, -0.07531099f, 0.08967623f, 0.06163262f, 0.02558480f, 0.21539094f, 0.12238820f, 0.01238689f, 0.03430167f, -0.00681279f, -0.06939505f, 0.02424417f, 0.05317433f, 0.03291040f, 0.01479342f, -0.06808028f, 0.03318218f, -0.03809754f, -0.02178681f, -0.07634928f, 0.00624443f, 0.09590766f, 0.02643260f, -0.03256097f, -0.02139881f, 0.00803821f, -0.00194843f, -0.01379843f, 0.02805880f, 0.09110619f, 0.12817272f, 0.18917530f, 0.19620324f, 0.06777284f, 0.03617392f, -0.00035236f, -0.02490171f, -0.06041528f, -0.01418312f, 0.15778319f, 0.12704621f, 0.01990183f, -0.01334248f, 0.01275771f, 0.03452087f, 0.04344137f, 0.17702784f, 0.17248388f, -0.00217630f, -0.01259515f, 0.04395744f, 0.09209669f, -0.00301484f, -0.06357345f, -0.00193061f, 0.09089994f, 0.06963276f, -0.04705874f, 0.08621294f, 0.20305922f, 0.14733464f, 0.00537891f, -0.07015627f, -0.02925369f, -0.04387395f, -0.07613594f, 0.08051648f, 0.05287251f, 0.10742327f, 0.12781378f, 0.00185050f, -0.02407443f, 0.08446516f, 0.19138731f, 0.19287433f, -0.05317991f, 0.01457163f, 0.17615633f, 0.12338971f, -0.02410067f, -0.01997527f, 0.06427425f, 0.09912043f, 0.02973422f, 0.11550064f, 0.28412175f, 0.27401931f, 0.13009426f, -0.00179805f, 0.00361931f, -0.00744251f, -0.00876631f, -0.04344801f, 0.00782531f, 0.11443395f, -0.02736931f, -0.04962283f, -0.05747234f, -0.01263648f, 0.04876266f, 0.01772048f, -0.05606695f, -0.05897533f, 0.04212877f, 0.12574365f, 0.05100610f, -0.02373270f, 0.03418980f, 0.06958645f, 0.03581157f, -0.06187286f, -0.13314185f, -0.11363571f, -0.16265814f, -0.14741514f, -0.10163389f, -0.01831422f, 0.02345721f, 0.00112610f, -0.07213626f, -0.01153121f, 0.07679853f, -0.00954420f, -0.01236787f, -0.00191636f, 0.02102156f, 0.09320207f, 0.11440229f, 0.01200765f, 0.01580170f, 0.13776397f, 0.04377210f, -0.02117637f, -0.02774677f, 0.00839943f, 0.04606172f, 0.07113678f, 0.04059410f, -0.00236888f, -0.04146146f, -0.07293811f, -0.03551126f, -0.05288199f, -0.01050904f, 0.05211464f, 0.08135778f, 0.05215934f, 0.05659533f, 0.05073829f, -0.04290589f, -0.03299994f, -0.05228200f, -0.06686912f, -0.03125264f, 0.02447885f, 0.07646390f, 0.08630959f, 0.15308443f, 0.03863923f, -0.01336908f, -0.01504630f, 0.02745804f, 0.06676539f, 0.04549179f, -0.00941141f, 0.06277441f, 0.21210904f, 0.01944606f, -0.00775366f, -0.02749250f, 0.03489776f, 0.06194960f, 0.01280329f, -0.03003625f, 0.08178132f, 0.04557126f, -0.02647011f, -0.05107037f, -0.04354807f, -0.03168634f, -0.01695719f, -0.01069118f, 0.00423954f, 0.01960752f, 0.00363192f, -0.08398365f, -0.06244669f, -0.05414137f, -0.02079875f, -0.01478226f, -0.04117374f, -0.05396009f, 0.10903294f, 0.20462797f, 0.05021287f, 0.00088519f, -0.00989290f, 0.02808106f, 0.08375212f, 0.06341691f, 0.03874941f, 0.09262669f, 0.11308999f, 0.02603594f, -0.03170430f, 0.01007232f, 0.03212558f, 0.04408872f, 0.08306846f, 0.02734099f, 0.04850299f, 0.04376443f, -0.03025161f, -0.00017485f, -0.01779346f, 0.00987933f, 0.01174144f, -0.01284634f, -0.03332717f, -0.03375625f, -0.03794997f, -0.04246108f, -0.02869293f, -0.02534422f, 0.00286679f, 0.00794620f, -0.01651066f, -0.04104292f, 0.10473653f, 0.09967461f, -0.00166463f, -0.01173101f, 0.01574206f, 0.04202032f, 0.03230069f, 0.11235283f, 0.11727340f, 0.05125457f, 0.02764031f, -0.03295885f, -0.01170240f, 0.05899599f, 0.06661803f, 0.03626407f, 0.11988996f, 0.12787615f, -0.02242351f, 0.00504938f, -0.01282758f, 0.00330486f, 0.01116149f, 0.02623526f, -0.00875115f, 0.01360551f, 0.01634169f, -0.03075921f, -0.02334645f, -0.08308145f, -0.05809294f, 0.01360069f, 0.00890510f, -0.03995065f, -0.01916020f, 0.05768820f, -0.03338022f, 0.03072185f, -0.03856144f, -0.01999118f, 0.05235477f, 0.05962528f, -0.01069340f, 0.07775559f, 0.12355455f, 0.03877542f, 0.03381226f, -0.06658885f, -0.02333074f, 0.00552062f, -0.01563264f, -0.06550736f, 0.06281615f, 0.02345994f, -0.00942014f, 0.02513063f, -0.08453318f, -0.02726737f, 0.02259305f, 0.04397870f, -0.06590197f, 0.01320565f, 0.06444215f, 0.06403526f, 0.06492008f, -0.06592464f, -0.00679350f, -0.01786066f, -0.03046658f, -0.09879498f, 0.06335532f, 0.03356900f, 0.02538753f, 0.08074788f, -0.04916835f, -0.03442308f, -0.01976235f, -0.02637749f, -0.06599230f, 0.00699125f, 0.03784747f, -0.01709697f, 0.04628171f, -0.07918559f, -0.04664379f, -0.05882357f, -0.02181337f, -0.07816813f, 0.01716982f, -0.02093918f, 0.06363778f, 0.14294344f, -0.00729700f, -0.02086586f, -0.03975954f, -0.02718286f, -0.03099527f, 0.06644224f, 0.00453680f, -0.00279812f, 0.14207097f, -0.03063236f, -0.03306897f, -0.03976284f, -0.01326272f, -0.03910180f, 0.03865838f, 0.02860238f, -0.00159905f, 0.02891019f, -0.08038869f, 0.00275223f, -0.03523654f, -0.03847580f, -0.07990568f, -0.03240742f, 0.00655186f, -0.00950451f, 0.03030633f, -0.06175885f, -0.04463698f, -0.03461041f, -0.05414447f, -0.11361448f, -0.04271584f, 0.00990494f, 0.01173655f, 0.07401961f, -0.05073611f, -0.00048585f, -0.03209816f, -0.03296361f, -0.07225237f, 0.02401655f, 0.03325837f, 0.00359078f, 0.03343356f, -0.05799064f, -0.03423673f, -0.01249530f, -0.04778390f, -0.11357858f, 0.03089653f, 0.06114212f, -0.04869358f, 0.02725726f, -0.06141309f, -0.04257458f, -0.03391343f, -0.04044024f, -0.08997650f, -0.02048483f, -0.01597157f, -0.06356172f, 0.02032759f, -0.02796919f, -0.05024252f, -0.02750763f, -0.07926772f, -0.08230081f, -0.03515042f, -0.04406974f, 0.01506530f, 0.05815461f, -0.04168106f, -0.00503547f, 0.01215259f, -0.00634988f, -0.05367854f, 0.09594094f, 0.06159157f, 0.03066549f, 0.14109437f, 0.07611588f, 0.03362676f, 0.01964200f, -0.02531438f, -0.00871042f, 0.12809873f, 0.03854195f, 0.02103229f, 0.03497566f, 0.02236829f, 0.00068557f, 0.00576055f, 0.03268923f, 0.04991757f, 0.04547716f, -0.00576077f, -0.10345632f, -0.11471539f, -0.10170760f, -0.16513671f, -0.07206152f, -0.08180868f, -0.04864718f, -0.06863556f, -0.06775289f, 0.13955666f, 0.21625922f, 0.17648549f, 0.13802288f, 0.12034226f, 0.12174142f, 0.08082704f, 0.17838860f, 0.06070135f, 0.01779694f, 0.04124157f, -0.03233995f, -0.06127840f, 0.12701339f, 0.04952988f, -0.03077957f, 0.00566800f, -0.00840636f, 0.01727667f, -0.04252973f, -0.00579454f, 0.00657250f, 0.05456345f, 0.14085164f, 0.04251496f, 0.02055925f, 0.04897321f, 0.00173676f, -0.03500521f, 0.00007397f, -0.01063325f, -0.07386503f, 0.02745165f, -0.01821077f, -0.02513711f, -0.00890430f, 0.01613530f, -0.00724885f, 0.01162412f, 0.09423027f, 0.10670270f, 0.18136554f, 0.01641618f, 0.02189014f, 0.00837821f, -0.04884737f, -0.05228189f, -0.02032306f, -0.00911725f, -0.07537987f, -0.00637520f, -0.04318022f, -0.03719686f, -0.05347254f, 0.10525957f, -0.14086957f, -0.10114672f, -0.05001454f, 0.00490366f, -0.06093127f, -0.12168697f, -0.09039295f, 0.12629120f, -0.03866391f, -0.01389998f, 0.24737462f, 0.21979051f, 0.09266596f, 0.13695539f, 0.05886318f, -0.03756755f, -0.00611080f, 0.13422406f, -0.09974454f, -0.09558716f, -0.00076124f, 0.03297782f, 0.00098333f, -0.09826375f, -0.12095454f, 0.12068651f, -0.01791501f, -0.07313050f, 0.04583494f, 0.13944114f, 0.08065329f, 0.21466844f, 0.23678333f, 0.03281764f, 0.00166105f, -0.03938716f, 0.12993016f, 0.22559471f, 0.15321696f, 0.05670407f, 0.06785019f, 0.07359262f, 0.00611562f, -0.12457312f, 0.08075396f, 0.07907013f, 0.02732013f, -0.01407095f, -0.04609820f, 0.00548012f, 0.01100920f, 0.11395174f, 0.04595756f, -0.12267547f, -0.02487511f, 0.05671187f, 0.07328320f, 0.04185171f, 0.12791861f, 0.23983444f, 0.16326690f, -0.03866825f, 0.09483738f, 0.10766032f, 0.01185728f, -0.02559055f, -0.03377063f, 0.01918188f, 0.03934670f, 0.10637551f, 0.05713366f, 0.04597195f, 0.05099368f, 0.12865020f, 0.11968852f, 0.02722912f, 0.04911642f, 0.09358116f, 0.14725431f, 0.08402478f, -0.10826845f, 0.00073817f, 0.08956420f, 0.05109856f, -0.02174436f, 0.02685142f, 0.07730557f, 0.05701752f, -0.02231336f, 0.10976779f, 0.17893505f, 0.11191328f, 0.06936906f, 0.05609476f, 0.13435180f, 0.12975388f, 0.05146990f, 0.04550557f, -0.06373665f, 0.04207178f, 0.08848577f, 0.03318804f, -0.02377519f, 0.08092027f, 0.09166491f, -0.00148523f, -0.09732184f, -0.07396782f, 0.01551105f, 0.04265712f, 0.00845616f, 0.00205897f, 0.09496128f, 0.10993712f, 0.03344473f, -0.05876167f, -0.08651651f, -0.04609407f, -0.03998192f, -0.05276594f, -0.09887476f, -0.00165423f, 0.05109006f, -0.00474297f, -0.07508182f, -0.02510697f, 0.05581969f, 0.11423494f, 0.06566761f, -0.02215771f, 0.01487986f, 0.04514476f, -0.00665142f, -0.08467108f, -0.06495210f, 0.01926343f, 0.05746061f, -0.00752790f, -0.09922579f, -0.03985320f, -0.01778563f, -0.04437042f, -0.09643686f, 0.00839632f, 0.00000003f, -0.02249254f, 0.00395790f, -0.05076798f, -0.00214162f, 0.02934254f, 0.04424550f, 0.02136807f, -0.02189512f, -0.00136183f, -0.00262694f, 0.00068793f, -0.03544258f, -0.03261003f, -0.02071419f, -0.01019022f, -0.00968886f, 0.05037246f, 0.03807054f, 0.05185946f, 0.02910318f, -0.08456747f, 0.00130287f, -0.02555543f, -0.01743256f, -0.02590962f, 0.05332902f, 0.02667266f, -0.02561306f, -0.02389163f, -0.05996433f, -0.00891502f, -0.01948255f, -0.03667188f, -0.01743306f, 0.01027151f, 0.01617083f, 0.03160640f, 0.01772609f, 0.00615548f, 0.04134491f, 0.01355164f, 0.02763402f, -0.03511004f, -0.08037053f, -0.02108188f, 0.04666260f, 0.02869894f, 0.00407638f, 0.05603933f, 0.00310372f, -0.04572687f, -0.08391380f, -0.01237899f, 0.02812295f, -0.00764517f, 0.03140251f, -0.01952859f, 0.04467414f, 0.03384958f, -0.00261538f, -0.03583918f, -0.08496789f, -0.04557842f, 0.03650739f, 0.05210478f, -0.00982684f, 0.02487117f, -0.04047078f, -0.03581297f, -0.07269186f, -0.03104114f, 0.01526856f, 0.04112554f, 0.06030381f, 0.09602299f, 0.10584307f, 0.05083776f, 0.00516964f, -0.01555764f, -0.07620039f, -0.04068996f, -0.02640504f, 0.06772925f, 0.07946116f, 0.08100740f, 0.03015750f, -0.01452982f, -0.07192916f, -0.01197824f, 0.00708646f, 0.04133136f, 0.07694827f, 0.09329003f, 0.04954912f, -0.01164895f, -0.00559937f, -0.00915932f, -0.06592101f, -0.02508148f, -0.01118354f, 0.03072639f, 0.07252927f, 0.06178024f, -0.03848466f, -0.06197235f, -0.04322416f, -0.06515582f, 0.02498031f, 0.01610568f, 0.11621507f, 0.08729621f, 0.07952595f, 0.01631199f, 0.03820659f, -0.05692835f, -0.01170174f, 0.08268365f, -0.01547763f, -0.00127521f, 0.06878420f, 0.04001095f, -0.01820089f, 0.05549363f, -0.01561756f, -0.03780331f, 0.03116065f, -0.00558418f, 0.05058428f, 0.07234902f, 0.06874058f, 0.01696176f, -0.02699820f, -0.08790866f, -0.00475972f, 0.04977786f, -0.02287321f, 0.00788080f, 0.06631130f, -0.00920112f, -0.00503366f, 0.04932084f, -0.03397763f, -0.01677677f, 0.10858294f, -0.04584461f, 0.00175899f, 0.03112592f, 0.01282750f, -0.05830748f, -0.00001515f, -0.02336738f, 0.05253844f, 0.16112209f, 0.01230715f, 0.02504474f, 0.02029933f, -0.01901412f, -0.07672866f, 0.04053434f, -0.00610074f, 0.00150222f, 0.00423188f, -0.06601086f, 0.01138694f, -0.01339311f, -0.03077153f, -0.04820887f, 0.09555214f, -0.01190642f, -0.00051372f, 0.07174371f, -0.06447052f, 0.00621654f, 0.02118773f, -0.01731864f, 0.00711133f, 0.13789094f, 0.04807130f, 0.04239473f, 0.18315955f, 0.01969009f, 0.01007879f, -0.01533023f, 0.00453615f, -0.02185988f, 0.04472318f, -0.00340814f, -0.00282423f, 0.15384753f, -0.00864331f, 0.00093011f, -0.02312893f, -0.01230641f, -0.06737984f, 0.02518332f, -0.01798957f, 0.01029321f, 0.04819970f, -0.04014233f, 0.00870287f, 0.00323040f, -0.00276518f, 0.02089614f, 0.18461840f, 0.04162447f, -0.03956050f, 0.00689805f, -0.05136883f, -0.02640154f, -0.04291697f, -0.03339553f, -0.02994196f, 0.15580723f, 0.00208344f, 0.00006145f, 0.08944876f, -0.05968593f, -0.00125841f, -0.03159946f, -0.01416559f, -0.07064263f, -0.00732787f, 0.01876730f, 0.01911270f, 0.01266492f, -0.07393005f, -0.02702696f, -0.05225755f, -0.02979923f, -0.08481974f, -0.00443864f, 0.03943988f, 0.05273076f, 0.00824213f, -0.05812100f, -0.01560609f, -0.03523251f, -0.02294356f, -0.04478757f, 0.09661599f, 0.05096852f, 0.08825145f, -0.00678405f, -0.06662003f, -0.04173945f, -0.03487521f, -0.03198292f, -0.05995188f, 0.03413445f, 0.02281388f, 0.01683302f, 0.02309085f, -0.03615615f, 0.00922749f, -0.03747601f, -0.00151699f, -0.05867106f, 0.01538846f, 0.03670397f, 0.07319083f, 0.09630558f, 0.02163523f, 0.04464880f, -0.03511577f, 0.02465440f, -0.01028953f, 0.08757360f, 0.08901113f, 0.06468205f, 0.00850178f, -0.03188605f, -0.02073816f, -0.03692611f, -0.00916474f, -0.06099901f, 0.05010365f, 0.04025919f, 0.03308198f, 0.05279352f, -0.01230072f, -0.00069767f, -0.02963162f, 0.03888060f, 0.01274940f, 0.08216323f, 0.04902829f, 0.09835809f, 0.14686380f, 0.06364880f, 0.07974184f, 0.08196809f, 0.09722158f, 0.06060648f, 0.11554286f, 0.09287435f, -0.01149923f, -0.00959010f, -0.04990456f, -0.05717653f, 0.09810881f, 0.00074832f, -0.06776313f, -0.04807139f, -0.04327408f, 0.06457861f, 0.10177048f, 0.03649410f, 0.08982882f, 0.08246144f, 0.09912353f, 0.09559776f, 0.13123523f, 0.05183118f, -0.00139036f, -0.02777161f, -0.06454733f, -0.00869735f, 0.05320297f, -0.04379272f, -0.03385603f, -0.02249998f, -0.03868563f, 0.02397124f, -0.01828895f, 0.01789295f, 0.12304450f, 0.13019817f, 0.16138068f, -0.00133326f, 0.00508584f, 0.02091708f, -0.01193229f, -0.07827248f, -0.01619965f, -0.00658052f, -0.11435752f, -0.05644886f, -0.01717006f, -0.03154029f, -0.01344207f, 0.04116127f, -0.00364583f, 0.02499248f, 0.19623038f, 0.09446634f, 0.09872220f, 0.02685408f, -0.01323739f, 0.00595940f, -0.00570945f, -0.07862936f, -0.01934205f, 0.02464510f, -0.08216548f, -0.01121107f, -0.00065075f, -0.07341437f, -0.04781108f, 0.10432183f, -0.07539319f, -0.07730502f, 0.02848239f, 0.10046745f, 0.06309269f, -0.04988202f, -0.09678241f, 0.08283736f, 0.00972853f, -0.07669644f, -0.04095363f, 0.00726648f, 0.02549775f, 0.12544624f, 0.18051207f, 0.02678776f, 0.07455426f, 0.03856341f, -0.08069542f, -0.06112178f, 0.02460854f, 0.05435756f, 0.07002785f, -0.04448111f, -0.09121361f, 0.01841631f, 0.06682483f, -0.04165618f, -0.02782957f, 0.04030683f, -0.00211646f, 0.02782884f, 0.05591145f, -0.00356601f, 0.09205087f, -0.02041870f, -0.07068150f, -0.07661374f, -0.04737741f, 0.00885447f, 0.08417277f, 0.20158002f, 0.21884722f, 0.11267920f, 0.18830722f, 0.16959935f, 0.03459606f, -0.00862981f, -0.00637755f, 0.00267114f, 0.00957657f, 0.09564630f, 0.15519289f, 0.02596730f, -0.08958856f, -0.06477788f, -0.04668788f, 0.05421089f, -0.02159811f, 0.00059251f, 0.05003057f, 0.07320426f, 0.11090724f, 0.02390158f, 0.00664463f, 0.00821145f, 0.04697831f, -0.01591132f, -0.06400283f, 0.02614006f, 0.08612062f, 0.19332012f, 0.21459422f, 0.09108026f, -0.00961999f, 0.02679868f, 0.12885627f, 0.08722519f, -0.01027233f, 0.04925112f, 0.00027938f, 0.09244009f, 0.06301554f, -0.02141696f, -0.00710239f, 0.18238071f, 0.21032452f, 0.04714587f, -0.06550160f, 0.06431734f, 0.07848973f, 0.04391511f, -0.02023944f, 0.02010939f, 0.05648153f, 0.02702592f, -0.05613040f, -0.01494226f, -0.04216101f, -0.05010295f, -0.04018654f, -0.05006826f, -0.00114598f, 0.12616008f, 0.19536765f, 0.02624171f, -0.06603454f, -0.04924516f, -0.00335113f, 0.03602555f, -0.02341281f, -0.05972806f, 0.09662103f, 0.15892493f, 0.09341360f, -0.02494603f, -0.10250979f, -0.03842479f, 0.00392524f, -0.04216918f, -0.12967976f, -0.13397740f, -0.13385532f, -0.06431475f, -0.11087623f, -0.02658508f, -0.03109550f, -0.01734704f, -0.01010824f, 0.00204162f, 0.13343727f, 0.29115075f, 0.29621606f, 0.12957321f, -0.04941351f, -0.04870047f, 0.01201412f, -0.01164530f, -0.06291991f, -0.02843410f, -0.01610257f, 0.12109750f, 0.04631030f, 0.03473264f, 0.02901399f, 0.02780893f, -0.00292868f, -0.08319262f, -0.03696434f, -0.07052003f, -0.01910194f, 0.00125427f, 0.07362900f, 0.01621709f, -0.05505499f, -0.05746995f, -0.08102766f, -0.05450313f, -0.00804614f, 0.02127556f, 0.05470977f, 0.06892800f, 0.06354294f, 0.06729061f, 0.00720600f, -0.02632980f, 0.00932793f, -0.02056070f, 0.08735841f, 0.06143756f, 0.04898791f, 0.06836458f, 0.04427061f, 0.01690682f, -0.01186558f, -0.00846702f, 0.07050262f, 0.10607520f, 0.02717329f, -0.02830156f, -0.01041690f, -0.04602304f, -0.03251315f, -0.04546427f, -0.03408293f, 0.00712380f, 0.05476869f, 0.06992219f, -0.07318057f, -0.04269656f, -0.03513868f, -0.01570996f, -0.06054290f, -0.08939751f, -0.09651177f, -0.00298308f, 0.00241314f, 0.00383805f, 0.06963382f, 0.04174110f, 0.03687547f, -0.01346464f, -0.00540072f, 0.06980318f, 0.14343751f, 0.04158169f, 0.02790872f, 0.01896098f, -0.01361889f, 0.01174622f, -0.03511487f, -0.01637691f, 0.02517369f, 0.16691091f, 0.05469737f, -0.05407617f, -0.02175886f, -0.00219090f, 0.02977610f, -0.01183363f, -0.04554290f, -0.06598578f, 0.00493783f, 0.05727448f, -0.05029985f, -0.02627704f, -0.03895453f, 0.00088526f, -0.04637572f, -0.06445765f, -0.05937397f, -0.03974638f, -0.03247331f, 0.08054627f, 0.05132806f, 0.03924033f, 0.02032054f, -0.02387696f, -0.02688822f, 0.01981987f, 0.11452399f, 0.13948572f, 0.05550212f, 0.04976687f, 0.00221827f, 0.03054305f, 0.00398379f, -0.02054215f, 0.03222634f, 0.05969597f, 0.15074978f, 0.00940513f, 0.05665812f, -0.03690442f, 0.01710226f, -0.03467103f, -0.04547962f, -0.02574404f, -0.00995656f, -0.07441129f, 0.04549617f, 0.05103092f, -0.02467454f, -0.01810496f, -0.03644529f, -0.09585622f, -0.07879090f, -0.01724366f, -0.05386656f, 0.12007583f, 0.11593883f, -0.00108973f, 0.06331847f, 0.01483515f, 0.00451878f, 0.01377495f, 0.05806591f, 0.12538862f, 0.11623885f, 0.09289539f, 0.05097044f, 0.06774928f, 0.03855121f, -0.02080855f, -0.06821406f, 0.01697278f, 0.06857783f, 0.06661061f, 0.06923361f, -0.05979605f, 0.00377394f, -0.02806776f, -0.06358518f, -0.08243922f, 0.01563826f, -0.00984804f, 0.06256996f, 0.07169235f, -0.08222328f, -0.02896287f, 0.01538426f, -0.02692310f, -0.04377355f, 0.05103557f, 0.04364534f, 0.13145330f, 0.08245789f, -0.01590376f, 0.02563823f, 0.04328925f, -0.03831622f, -0.05657394f, 0.00318892f, 0.00437099f, 0.06859436f, 0.01449919f, -0.09820340f, -0.03559065f, 0.00493211f, -0.03552414f, -0.09336953f, -0.00091086f, 0.02923482f, 0.04489774f, 0.06887023f, -0.04845246f, -0.00746722f, 0.00141716f, -0.00646829f, 0.01808559f, 0.14371538f, 0.07217328f, -0.01361160f, 0.00556389f, -0.01801031f, -0.01086600f, -0.03105601f, -0.05184779f, -0.03021807f, 0.12271431f, -0.00181778f, 0.04887300f, -0.04234366f, -0.09153778f, -0.02617231f, 0.01713646f, -0.01478739f, -0.04321050f, 0.02912005f, 0.01325879f, -0.01987462f, -0.02996652f, -0.07688789f, -0.02976402f, -0.03632253f, -0.06232147f, -0.04756397f, 0.01862723f, -0.03647855f, 0.04767905f, 0.05104293f, -0.00800814f, -0.00121305f, -0.01993883f, -0.01607671f, -0.04351053f, 0.05002163f, 0.03215578f, 0.05351920f, 0.06451320f, -0.04023814f, -0.03602701f, -0.03037966f, -0.00912842f, -0.03311651f, 0.01706739f, 0.02357509f, 0.01756228f, -0.03193479f, -0.07198922f, -0.02792396f, -0.02827433f, -0.04522877f, -0.08325717f, 0.02296899f, 0.01415474f, -0.00589296f, -0.07635492f, -0.09233603f, -0.05708796f, -0.00733712f, -0.06157023f, -0.07755379f, -0.00886959f, -0.01915531f, 0.09597610f, 0.11547233f, -0.00117541f, -0.02678376f, -0.01678947f, 0.02603399f, -0.00905471f, 0.06443106f, 0.07343042f, 0.02791316f, 0.10418534f, 0.02619791f, -0.01273704f, 0.03018652f, 0.05706981f, 0.08502818f, 0.10690930f, 0.05445301f, -0.02091751f, -0.03302636f, -0.06870660f, -0.04162627f, -0.02169462f, -0.03989058f, -0.03941961f, -0.01179037f, -0.04620381f, 0.00033928f, 0.00511216f, -0.08074241f, -0.09587951f, -0.05421209f, -0.06318947f, -0.02312299f, 0.00967500f, -0.03451054f, 0.05763408f, 0.18511230f, 0.12477451f, 0.16714466f, 0.11437043f, 0.13684174f, 0.17166358f, 0.19563850f, 0.12549030f, 0.02097778f, 0.03502142f, 0.02029730f, 0.09439357f, 0.10507995f, -0.05077747f, 0.00498007f, 0.03822214f, 0.01388552f, 0.02075358f, 0.10385404f, 0.06137262f, 0.04708865f, -0.02733714f, 0.00793731f, 0.02321913f, 0.07716232f, 0.06091208f, -0.05706971f, -0.03948791f, -0.04271427f, -0.03517904f, -0.04758576f, -0.08988640f, -0.05482712f, -0.07229480f, -0.07365687f, 0.04173532f, -0.01189127f, 0.04457097f, 0.19052899f, 0.06822842f, 0.02415028f, 0.01004056f, -0.03388381f, 0.00984858f, -0.01559888f, -0.05946589f, -0.02291987f, 0.04766615f, -0.08530377f, -0.00586304f, -0.00944876f, -0.10487568f, -0.06150673f, 0.04823397f, 0.03052609f, 0.05436728f, 0.14490046f, 0.02523789f, 0.02398118f, 0.01779756f, -0.04894702f, 0.02155512f, 0.00844772f, -0.04484100f, -0.02029081f, 0.05158258f, -0.06525607f, 0.03135741f, -0.00066580f, -0.07032901f, 0.02243728f, 0.07674856f, -0.05227525f, -0.05889293f, 0.02205534f, 0.01686337f, 0.03017922f, -0.00484920f, -0.01172374f, 0.08201273f, 0.05114947f, -0.02445091f, 0.00575338f, 0.07527946f, 0.00326127f, -0.00376301f, 0.00837297f, -0.00912501f, 0.07453136f, 0.08611866f, -0.07457649f, -0.09211471f, 0.04600267f, 0.06200267f, 0.06069857f, -0.04124541f, -0.05625218f, 0.07210362f, 0.09153432f, 0.00258992f, -0.03244420f, 0.06909473f, -0.01301208f, 0.04247276f, -0.01195426f, -0.00310029f, 0.06990365f, 0.04827955f, -0.06103435f, -0.00885628f, 0.03581566f, 0.06849755f, 0.01423548f, -0.01458802f, -0.04234752f, 0.02318813f, 0.03994895f, -0.03319721f, 0.00137956f, 0.04551387f, 0.04772971f, 0.02057125f, -0.07002313f, -0.05042756f, 0.00902410f, 0.08252606f, -0.03221759f, -0.03292146f, 0.04478717f, 0.05955368f, 0.03122935f, -0.03405582f, -0.05572539f, 0.06060901f, 0.06588193f, 0.02233029f, -0.00662791f, 0.07011611f, 0.04730594f, 0.05009932f, -0.04105226f, -0.03124980f, 0.03795244f, 0.04922187f, 0.02457097f, -0.00158194f, 0.03016409f, 0.04604079f, 0.01225655f, 0.00768139f, -0.02086293f, -0.00642677f, 0.03937373f, -0.01988622f, -0.01759670f, 0.03014201f, 0.00526252f, 0.02391432f, 0.06521229f, 0.02446797f, -0.01662892f, 0.03800649f, 0.05449071f, -0.00900585f, 0.02948467f, 0.03399858f, -0.03196317f, -0.03003429f, -0.02005314f, -0.01004214f, 0.02173329f, 0.02314603f, -0.00899497f, 0.04594270f, 0.01146284f, -0.03988936f, -0.03529846f, -0.04203388f, -0.00364545f, -0.03990285f, -0.01681246f, 0.00730125f, 0.00651391f, 0.01306287f, -0.02902056f, 0.06507879f, 0.18274443f, 0.05493136f, -0.07769020f, -0.05945407f, -0.03419486f, -0.00301465f, -0.01015667f, -0.05859650f, -0.06581258f, 0.12914647f, 0.00855314f, -0.02999214f, 0.01870564f, 0.01636983f, 0.00594101f, 0.09158290f, -0.01992963f, -0.01439548f, 0.03148111f, 0.01745277f, -0.04684882f, 0.01672750f, -0.00008909f, 0.00395870f, 0.07990142f, -0.04886562f, -0.03347239f, 0.04238323f, 0.03645968f, -0.03737678f, -0.02685757f, 0.00676550f, 0.01079637f, 0.02164989f, 0.00477513f, -0.02776546f, 0.14522868f, 0.09419426f, -0.06457892f, -0.03947229f, 0.00852369f, 0.01658891f, -0.00120213f, -0.02136028f, 0.03650957f, 0.08211399f, 0.04113915f, 0.02842554f, 0.01558206f, -0.00391282f, 0.03667829f, 0.11830539f, -0.02953592f, -0.03840422f, 0.09616995f, 0.06149084f, 0.00840583f, 0.00193418f, -0.01846520f, 0.01816136f, 0.04909121f, -0.02784627f, -0.02958180f, 0.05261611f, -0.00366010f, -0.04943417f, 0.01787256f, 0.04266484f, 0.04936877f, 0.01663504f, 0.01918873f, 0.01811861f, 0.05427383f, 0.02698422f, -0.03559382f, -0.00615228f, -0.00310941f, 0.00801493f, 0.01006382f, -0.03470408f, -0.04310779f, 0.04137935f, -0.03009445f, 0.06824666f, 0.09271848f, -0.00634434f, 0.00791791f, 0.04936559f, -0.00087824f, -0.03730512f, 0.03447651f, 0.05986234f, 0.01454374f, 0.05844504f, -0.03401435f, -0.01088313f, 0.05046283f, -0.03814898f, -0.07983704f, 0.00310541f, 0.00343942f, -0.01146103f, 0.05558245f, 0.04443827f, 0.04133765f, 0.05042951f, -0.01331093f, -0.00546845f, 0.09302687f, 0.06905136f, -0.00335239f, -0.01396125f, -0.00265998f, 0.00885855f, -0.01532695f, -0.02225095f, 0.00465921f, 0.05504971f, 0.07093435f, 0.05815101f, 0.07383817f, -0.02263498f, 0.03237242f, 0.08934339f, 0.01669418f, -0.00245146f, 0.05532628f, 0.01440430f, 0.00903079f, 0.00999576f, -0.02363705f, -0.03990779f, 0.01175401f, -0.01886074f, -0.03928206f, -0.02686220f, -0.00578218f, 0.05856084f, 0.06536100f, 0.07953058f, 0.12277063f, 0.06579981f, 0.04669595f, 0.02615001f, 0.07795757f, 0.08860185f, 0.00894653f, -0.00095026f, 0.05556465f, 0.09124576f, 0.07953962f, 0.04879292f, -0.03274483f, 0.02219813f, -0.01095338f, 0.04570318f, 0.04971752f, 0.04153429f, 0.00813604f, 0.01544392f, 0.01436589f, -0.02617892f, -0.02909352f, 0.03323800f, -0.02523698f, -0.00542603f, -0.04327979f, -0.00883037f, -0.01592903f, -0.04900195f, -0.07470231f, -0.03331223f, 0.01508832f, 0.03193058f, -0.01235855f, 0.03888994f, 0.12690319f, 0.10259648f, 0.08085199f, 0.01383060f, 0.05938118f, 0.02375473f, -0.02110355f, -0.06727097f, -0.05353086f, 0.04581920f, 0.05902756f, 0.02369021f, -0.01088583f, 0.04770785f, -0.03019699f, 0.01489520f, -0.02308588f, -0.03301457f, 0.03577179f, 0.01534300f, 0.03176180f, -0.02570661f, 0.03082427f, 0.01702997f, -0.06417046f, -0.00743611f, -0.07683954f, -0.02416229f, -0.02716891f, 0.00648389f, -0.01913810f, -0.05292414f, -0.04493919f, -0.03205090f, -0.03512553f, -0.02067491f, 0.07810253f, 0.06956147f, 0.07608060f, -0.00439459f, 0.06422699f, 0.04618165f, 0.00073652f, -0.00065605f, -0.01351979f, 0.05947396f, 0.08363741f, 0.03709802f, -0.00890000f, 0.08036263f, 0.02515828f, -0.01526812f, 0.02674250f, -0.01620544f, 0.02942497f, -0.02800028f, 0.02515996f, -0.03516104f, -0.05285145f, -0.02623024f, -0.07686347f, -0.01465029f, -0.05995917f, -0.02811397f, 0.00407168f, 0.02590344f, -0.02709220f, -0.01482974f, -0.03581450f, 0.02306355f, 0.00125612f, -0.00746959f, 0.11636010f, 0.06353668f, 0.05856116f, -0.00510922f, 0.10661128f, 0.06956509f, -0.00509406f, -0.07645416f, -0.04797424f, 0.09195671f, 0.06642561f, 0.01856304f, -0.01840665f, 0.07166628f, 0.05389682f, -0.02966385f, -0.00412842f, -0.06313597f, 0.02032228f, 0.01803891f, 0.03856259f, -0.00636893f, 0.00583526f, 0.00333887f, -0.07075556f, -0.02598729f, -0.04262249f, 0.02747966f, 0.04269435f, 0.04036256f, -0.05993697f, -0.01309204f, -0.00616782f, -0.01100572f, 0.02741581f, 0.00678166f, 0.08646400f, 0.06094679f, 0.08226234f, 0.02216218f, 0.05885366f, 0.03172926f, 0.00432838f, 0.07154184f, -0.00016575f, 0.02468891f, 0.08248253f, 0.06417041f, 0.03836306f, 0.03213010f, 0.00334243f, -0.00700404f, 0.02201848f, -0.00667748f, -0.00374756f, -0.00417861f, 0.06362017f, -0.03027024f, 0.02560591f, -0.00797921f, -0.05395662f, 0.00976152f, -0.03326687f, -0.09777773f, -0.02438785f, 0.02784561f, -0.01839195f, -0.00662658f, -0.06515074f, 0.03205412f, 0.16445254f, 0.11896603f, 0.11915756f, 0.10712766f, 0.12403933f, 0.07145707f, 0.08147925f, 0.05313059f, -0.00463895f, 0.05130856f, 0.03937747f, 0.11215770f, 0.05129913f, 0.07595516f, 0.04988167f, 0.00387684f, 0.01303993f, 0.01003944f, 0.06360882f, 0.06987923f, 0.04032590f, 0.03986067f, 0.01867262f, -0.00641625f, 0.01586917f, 0.00829057f, 0.02717276f, -0.02545939f, -0.00865146f, -0.01549467f, -0.05586487f, -0.03216166f, -0.03435429f, -0.02628871f, 0.00819020f, 0.04948847f, 0.04158808f, 0.01903186f, 0.09328521f, -0.02443861f, 0.03512367f, 0.02545465f, -0.04979515f, 0.02396157f, 0.03697643f, -0.00363333f, -0.00771977f, 0.03739197f, -0.03720551f, 0.05574818f, 0.01588334f, -0.09487705f, 0.03341061f, 0.11053745f, 0.00748562f, -0.03168483f, 0.03220961f, -0.04060721f, 0.06359432f, 0.01968495f, 0.01757930f, 0.08122966f, 0.08566149f, -0.02383279f, -0.04548879f, 0.02258308f, 0.02139437f, 0.11535354f, 0.02789094f, -0.05609750f, 0.03475765f, -6.858349267f };
	return std::vector<float>(detector, detector + sizeof(detector) / sizeof(detector[0]));
}

std::vector<float> HOGDescriptor_Mod::HOG_Optimal_48_96()
{
	static const float detector[] = { 0.09704294f, -0.09404296f, -0.09264451f, 0.00233257f, 0.08523120f, 0.01162400f, -0.06430730f, -0.09698108f, 0.08591208f, 0.12046674f, 0.00382261f, 0.02530063f, 0.05699618f, 0.07680603f, 0.06031238f, 0.01570685f, -0.04026570f, 0.10637201f, 0.08030518f, -0.08865493f, -0.05830098f, 0.01712700f, 0.01524952f, 0.03273709f, -0.00791331f, -0.05222553f, 0.11249964f, 0.09472853f, -0.03177176f, -0.02112908f, 0.02049019f, -0.01261057f, 0.05120576f, 0.01474442f, 0.03315216f, 0.10261246f, 0.05698076f, 0.00264874f, 0.02421737f, 0.03315810f, 0.09224025f, 0.02950718f, 0.05435059f, 0.00359644f, 0.05891895f, 0.02177277f, 0.05348719f, 0.06197016f, 0.01810634f, 0.06430286f, 0.05276782f, 0.07071312f, 0.03128457f, 0.02746354f, 0.07191923f, -0.06187612f, -0.04629605f, 0.00977473f, 0.05452585f, 0.02519837f, -0.01832478f, -0.03484390f, 0.05446516f, -0.02502962f, -0.10025666f, -0.08603612f, 0.00087145f, 0.04544096f, 0.01957439f, 0.01055788f, 0.01600967f, 0.00956620f, 0.02651227f, 0.07969916f, 0.05006885f, -0.01361039f, 0.04389553f, 0.00801019f, 0.06485510f, 0.04850648f, 0.01067884f, -0.00620165f, -0.01665498f, -0.03911096f, -0.01352284f, 0.05072292f, -0.02710555f, 0.02337306f, 0.00885723f, 0.01063383f, 0.03636149f, 0.05253469f, 0.02569619f, -0.02688036f, 0.03507653f, 0.02806451f, 0.06632368f, 0.06917192f, 0.05894904f, 0.01078385f, 0.06343300f, -0.07114184f, -0.04740561f, 0.00109703f, -0.00536848f, 0.00857385f, -0.00436310f, 0.04077695f, -0.01927282f, -0.01270930f, -0.04067280f, 0.02233046f, 0.13096419f, 0.00033311f, 0.04461118f, 0.06645032f, 0.02517688f, -0.05154896f, -0.03233535f, -0.12132152f, -0.01056414f, 0.09838103f, -0.05524288f, -0.05580739f, 0.05740855f, -0.03677814f, 0.11269077f, 0.13883445f, -0.02635584f, 0.01127549f, 0.09496318f, 0.06361653f, 0.02858781f, 0.01004665f, 0.01317015f, -0.01808924f, 0.03608288f, -0.07963456f, -0.00786453f, 0.07748957f, 0.00221248f, -0.02483998f, 0.01081864f, -0.06242160f, 0.01631614f, 0.02684950f, -0.06015692f, 0.03281515f, 0.09732607f, -0.02371435f, -0.04931925f, 0.05181242f, 0.03784150f, -0.05710549f, -0.07853282f, -0.07657256f, 0.01842508f, 0.02733587f, -0.05615168f, -0.08242791f, 0.01024042f, 0.02756822f, 0.04249174f, 0.11106752f, -0.02918227f, 0.04153901f, 0.13420011f, 0.03057074f, 0.04637970f, 0.08345338f, 0.04599189f, 0.01371022f, -0.01522912f, -0.06379850f, 0.02679155f, 0.08867664f, -0.02072670f, 0.01250695f, 0.00785863f, -0.01120053f, -0.00777525f, -0.01860584f, -0.03400801f, 0.06017798f, 0.07832230f, -0.02714868f, -0.04187518f, 0.06625708f, 0.12033571f, -0.05486478f, -0.03765761f, -0.04119350f, -0.05952018f, -0.02852880f, -0.11708650f, -0.08276714f, 0.01920479f, -0.02170934f, 0.10271463f, 0.09015078f, 0.05387986f, 0.11753530f, 0.13029659f, 0.08920325f, 0.07389570f, 0.07610526f, 0.06575281f, 0.01394680f, 0.01506709f, -0.00009179f, 0.04202854f, 0.06386163f, 0.07376345f, 0.04082262f, -0.00084543f, 0.00215832f, -0.01389685f, 0.00078224f, 0.02376509f, 0.03280905f, 0.01062430f, -0.02123280f, -0.01905642f, 0.04503723f, -0.00452306f, -0.00788767f, 0.00820679f, -0.03901858f, -0.03953451f, 0.00727499f, -0.04188185f, -0.06842983f, 0.03612176f, -0.01794283f, 0.03201301f, 0.02307078f, 0.01843997f, 0.08207232f, 0.13163853f, 0.12950414f, 0.04653864f, -0.01038076f, -0.00453918f, -0.04781098f, 0.00202030f, -0.01228559f, 0.02508506f, 0.08082474f, 0.03113791f, -0.03547370f, -0.02293313f, -0.03117179f, -0.00046340f, 0.02831757f, -0.03332982f, -0.00729251f, -0.00167500f, 0.01244612f, -0.04241767f, 0.05295489f, 0.02608278f, -0.03040576f, -0.01626657f, -0.03354100f, -0.03225352f, -0.02612318f, -0.04719190f, -0.05916613f, 0.01859113f, -0.00302025f, 0.00441520f, 0.01351802f, -0.03774631f, 0.04949731f, 0.06251439f, 0.09677348f, 0.00752594f, 0.02272055f, -0.01439065f, -0.02872270f, 0.00540527f, -0.02514138f, 0.02300836f, 0.06991167f, 0.08601391f, 0.00638830f, -0.02697001f, -0.08916937f, 0.02773917f, 0.01715038f, 0.00569107f, 0.01953852f, -0.00751216f, 0.00158556f, -0.02009859f, 0.05042040f, 0.03184902f, 0.01802920f, 0.01800022f, -0.01417489f, -0.00730228f, -0.01352368f, -0.06001259f, -0.03393819f, 0.00154519f, -0.01661884f, -0.01695938f, 0.04804196f, 0.03804207f, 0.05044955f, 0.10449062f, 0.10196092f, 0.03899664f, -0.01781017f, -0.09197667f, -0.01975720f, 0.04182478f, 0.00062294f, 0.06015245f, 0.06912434f, 0.01632216f, 0.02559911f, -0.00393933f, -0.02773323f, -0.00501814f, 0.02863713f, 0.01129037f, 0.02133319f, 0.03161496f, -0.04377764f, 0.01206643f, 0.03043320f, 0.00504221f, -0.04156394f, -0.01203870f, 0.00371134f, 0.01624169f, -0.06690533f, -0.03045550f, -0.04583092f, 0.02697520f, -0.05347309f, 0.04001070f, 0.05763417f, 0.04737428f, 0.14803921f, 0.07481420f, 0.07838101f, 0.09387757f, 0.07402184f, 0.05848897f, 0.01440240f, -0.01178969f, 0.02675746f, 0.05477036f, 0.01839877f, 0.11121179f, 0.04158660f, 0.01280445f, -0.00945190f, 0.03432130f, 0.03950326f, 0.05294622f, 0.10910512f, -0.03212875f, 0.05411688f, 0.00202279f, 0.01868015f, 0.04380630f, 0.10609076f, -0.03872228f, 0.00292740f, 0.07631289f, -0.03925285f, 0.05888532f, -0.02544458f, -0.03808445f, 0.11376998f, 0.04218186f, -0.00117646f, 0.02560653f, 0.11917710f, -0.07276708f, 0.10059494f, -0.00044834f, -0.02457037f, 0.04743939f, 0.08198231f, -0.06368829f, -0.00787638f, 0.05811253f, -0.00948007f, 0.05101834f, -0.05628267f, -0.08375954f, 0.09649587f, 0.06177526f, -0.11170032f, -0.03652520f, 0.05143711f, 0.12932742f, 0.03056854f, -0.03212971f, -0.08572961f, 0.07581145f, 0.10703526f, 0.01032917f, 0.01193648f, 0.03999643f, 0.07586719f, 0.05228392f, 0.00091948f, -0.04042387f, 0.08199324f, 0.05302255f, -0.13145086f, -0.07796956f, 0.01872553f, 0.09825356f, -0.00904728f, -0.12662218f, -0.07194603f, 0.07058406f, 0.03568995f, 0.07655273f, 0.15478945f, 0.11628055f, 0.06113598f, -0.00230418f, -0.05593939f, -0.13130660f, 0.01177477f, 0.05288411f, -0.03850977f, -0.07537075f, -0.06894960f, 0.03453264f, -0.03557390f, -0.04226612f, -0.09892403f, 0.04096111f, 0.01039296f, -0.10423355f, -0.03266004f, 0.00750238f, 0.05426798f, 0.01513538f, 0.05707094f, 0.06282192f, 0.10318081f, 0.14401772f, 0.19051928f, 0.13866281f, 0.04510098f, 0.02743149f, -0.07103036f, -0.01242548f, -0.05968178f, 0.09186213f, 0.12243801f, 0.06935975f, 0.07160067f, 0.06232606f, 0.03314865f, -0.02329963f, 0.17276798f, 0.25531581f, 0.23917242f, -0.02145913f, 0.11564192f, 0.21746804f, 0.08103895f, -0.02444904f, -0.08042882f, 0.02667634f, 0.02284745f, -0.00546188f, 0.08027013f, 0.26428207f, 0.19874309f, 0.01603200f, -0.03677072f, -0.07374286f, -0.03381312f, -0.07913262f, -0.02937266f, -0.05538964f, 0.15140660f, 0.39412808f, 0.26534001f, 0.02998121f, -0.05425143f, 0.09359057f, 0.13220711f, 0.00802695f, -0.11015196f, 0.06324341f, 0.22839594f, 0.06845898f, -0.06314254f, -0.07405142f, -0.00047651f, -0.03810750f, -0.13437118f, 0.22797864f, 0.32310507f, 0.07350347f, -0.00685282f, 0.03375394f, 0.03014295f, 0.05385944f, 0.02848375f, 0.08245826f, 0.07588128f, 0.22636469f, 0.02422134f, -0.03988256f, -0.02464110f, 0.00179036f, 0.03780741f, 0.04964998f, 0.02276328f, 0.00322772f, 0.04327676f, -0.09547464f, -0.09645042f, -0.12122501f, -0.03608746f, 0.01433413f, 0.03619516f, 0.00130383f, -0.01255401f, 0.01381109f, -0.07830126f, -0.08327183f, -0.10247006f, -0.06570303f, -0.06167982f, -0.00952881f, 0.04872391f, 0.09598012f, 0.20663760f, 0.03556940f, 0.01076099f, 0.01664544f, 0.01247335f, 0.06963738f, 0.05267685f, 0.02050333f, 0.10823374f, 0.18584773f, 0.02247138f, -0.03393320f, -0.01905116f, -0.02255006f, 0.07770268f, 0.05706461f, 0.01117615f, 0.02771975f, 0.09514167f, -0.01784209f, -0.03492560f, -0.04131358f, -0.00959385f, -0.01910868f, -0.02101364f, -0.00408871f, 0.06271537f, 0.09842602f, -0.07615816f, -0.11873367f, -0.06944142f, -0.01029374f, -0.00979321f, -0.04572506f, -0.01217948f, 0.19258931f, 0.19994701f, 0.05681718f, -0.01861973f, 0.00509739f, 0.01104104f, 0.06666094f, 0.13279101f, 0.10779116f, 0.11281941f, 0.11126272f, -0.00578148f, 0.00781118f, 0.02448956f, 0.07424548f, 0.05632261f, 0.13659693f, 0.11291632f, 0.11053381f, 0.05509882f, -0.02747819f, -0.03606524f, -0.01748575f, 0.01993543f, 0.00842445f, 0.02705224f, 0.03405889f, 0.06544329f, -0.05474868f, -0.08956165f, -0.07465373f, -0.00080848f, -0.01918860f, -0.05185906f, 0.00938704f, 0.09135036f, 0.05757755f, 0.06806729f, 0.00283954f, 0.01864786f, 0.07418914f, 0.08160040f, 0.05429894f, 0.14961427f, 0.15570128f, -0.02675257f, 0.02904450f, -0.06224375f, -0.03386937f, 0.01904556f, -0.03740256f, -0.08401444f, 0.01667739f, 0.06392481f, 0.04506516f, -0.00924719f, -0.06662599f, -0.03672435f, 0.05890842f, 0.03671397f, -0.04456398f, 0.17088223f, 0.12330943f, 0.03493865f, 0.00865561f, -0.08017132f, -0.06006601f, -0.00464997f, -0.03040995f, -0.15978887f, 0.13669670f, 0.02161643f, 0.04781520f, 0.03021352f, -0.11462383f, -0.04584262f, -0.00256311f, -0.00154010f, -0.03895279f, -0.03436361f, 0.00332259f, 0.00907708f, 0.00473249f, -0.07832059f, -0.07007512f, -0.01228121f, -0.03317922f, -0.05382863f, -0.05982078f, -0.03523856f, 0.10277468f, 0.15953230f, -0.03742193f, -0.04678626f, 0.00758036f, -0.01692663f, -0.07018211f, 0.09657356f, 0.00686801f, 0.01837343f, 0.11713295f, -0.07585940f, -0.05498678f, -0.00787710f, -0.05713310f, -0.08110489f, 0.03182187f, -0.01020153f, -0.02162001f, 0.00946652f, -0.01912971f, -0.07696014f, -0.00798237f, -0.05247193f, -0.08147296f, -0.08648086f, -0.04199878f, -0.07028710f, -0.03754479f, -0.06412251f, -0.10180710f, -0.05016395f, -0.07905228f, -0.09782258f, -0.08479765f, -0.06054419f, 0.02642760f, 0.03718761f, -0.08564058f, -0.03448189f, 0.02710844f, -0.05710458f, -0.10088112f, 0.08115349f, 0.06895507f, 0.06604196f, 0.11247200f, 0.00892231f, 0.02509727f, 0.04639540f, -0.04072173f, -0.03970481f, 0.10060739f, 0.07574029f, -0.02233495f, -0.00449771f, -0.01572371f, -0.01285498f, -0.03353433f, -0.02263672f, 0.00560429f, 0.02632452f, -0.00932612f, -0.13953975f, -0.15752239f, -0.11049603f, -0.13708625f, -0.11145936f, -0.09593115f, -0.06216750f, -0.07891934f, -0.12822527f, 0.13792524f, 0.29431277f, 0.20943047f, 0.21167549f, 0.12127799f, 0.08846678f, 0.11096800f, 0.23022122f, 0.07845184f, 0.04843962f, 0.04748055f, 0.01290187f, 0.01395453f, 0.16030650f, 0.12535583f, 0.06508936f, 0.06124035f, 0.01163863f, 0.04521741f, -0.04952662f, 0.00584155f, 0.01140037f, 0.00086083f, 0.14056815f, 0.05251986f, 0.01784434f, 0.06243640f, 0.05193138f, -0.08045775f, -0.00855909f, -0.00651408f, -0.05013932f, 0.02198664f, -0.03233365f, -0.06951960f, 0.06237828f, 0.02313609f, -0.02554867f, 0.02846740f, 0.08469251f, 0.03092733f, 0.23524124f, 0.08589984f, 0.02518689f, 0.01258796f, -0.00565202f, -0.11566716f, -0.02463213f, 0.00177689f, -0.04655072f, 0.05212771f, -0.03558776f, -0.09146556f, 0.03040665f, 0.10821085f, -0.14434680f, -0.10785992f, -0.07940133f, 0.03478696f, -0.06975466f, -0.11527152f, -0.08529695f, 0.12330097f, 0.00845232f, 0.05515201f, 0.31942985f, 0.26933606f, 0.15353311f, 0.14234544f, 0.08879642f, -0.05849259f, -0.01388438f, 0.09350616f, -0.09137692f, -0.10115396f, -0.05141247f, 0.00287531f, -0.07712841f, -0.11145342f, -0.12916673f, 0.10355907f, -0.03893811f, -0.08622213f, 0.07928380f, 0.13921500f, 0.10976811f, 0.23119585f, 0.31331902f, 0.11787453f, 0.05162506f, 0.02594153f, 0.20412695f, 0.25540379f, 0.17382808f, 0.05688196f, 0.04099021f, 0.09217469f, 0.02796401f, -0.03292966f, 0.09257399f, 0.09642021f, 0.07874663f, 0.01563810f, -0.07651377f, -0.01691227f, 0.09518386f, 0.20798305f, 0.16897484f, -0.02059806f, 0.00687187f, 0.08932656f, 0.06886199f, 0.05909703f, 0.13988276f, 0.24771574f, 0.21316873f, 0.08968877f, 0.16738481f, 0.24063699f, 0.11503545f, -0.02058646f, -0.08953737f, -0.00063627f, 0.04877280f, 0.06899959f, 0.11358556f, -0.03071199f, 0.10045680f, 0.28509298f, 0.25069641f, 0.10812544f, 0.13667551f, 0.19408209f, 0.17075883f, 0.02418404f, -0.19478424f, -0.11677786f, 0.03147544f, 0.01262789f, -0.01528905f, -0.02547599f, 0.03972013f, -0.02763180f, -0.15385120f, 0.04506200f, 0.21332422f, 0.19936098f, 0.13112795f, 0.08573775f, 0.19882136f, 0.27541285f, 0.10569424f, -0.01506712f, -0.17703938f, -0.06508087f, 0.00215249f, -0.03035714f, -0.04778150f, -0.00757145f, 0.03710911f, -0.12620819f, -0.23081972f, -0.02533979f, 0.00435396f, -0.01458921f, -0.00605496f, -0.01182098f, 0.05168531f, 0.09270270f, 0.10516848f, 0.03701244f, -0.04519693f, -0.00385408f, 0.00100347f, -0.04083143f, -0.04409402f, -0.01848429f, -0.03907403f, -0.01443707f, -0.01593390f, 0.02879968f, 0.10822436f, 0.08911141f, 0.02660569f, -0.06278834f, -0.04727991f, -0.02634648f, -0.03102747f, -0.10733170f, 0.06331927f, -0.01305726f, -0.02816222f, -0.04772131f, -0.08543380f, -0.04690955f, -0.02982952f, -0.04407402f, -0.07573844f, -0.01536096f, 0.06341108f, 0.06406488f, 0.02769218f, 0.00136459f, 0.03592863f, 0.01453845f, 0.01130140f, -0.04851278f, -0.07393061f, 0.03124757f, 0.00868070f, -0.01528984f, 0.02656518f, 0.04379634f, 0.02700371f, -0.04026794f, -0.13670813f, 0.02331815f, -0.00513495f, 0.04017119f, 0.03260131f, -0.03809156f, 0.01933558f, 0.03327834f, 0.05148747f, 0.01671995f, -0.07434426f, 0.00040815f, -0.01182095f, 0.01785795f, -0.01550115f, -0.01075832f, -0.01955898f, -0.01957993f, -0.03428998f, -0.11510496f, 0.00901739f, 0.02548478f, 0.11556044f, 0.14252895f, 0.12061938f, 0.07272642f, 0.00186276f, -0.14582554f, -0.08684350f, -0.01167271f, -0.02625902f, 0.07650384f, 0.13110152f, 0.06569376f, 0.00942557f, 0.01035368f, -0.05538436f, -0.11437777f, 0.01058124f, 0.04227795f, 0.11626786f, 0.09703674f, 0.07649324f, 0.03935752f, -0.00414108f, -0.08796321f, -0.04189392f, 0.05519395f, 0.01256344f, 0.08891113f, 0.10806020f, 0.02537994f, -0.01263298f, -0.02389242f, -0.06870301f, -0.08463242f, 0.04782957f, -0.04602372f, 0.01730513f, 0.02144818f, 0.01629821f, -0.06335574f, 0.01070874f, -0.03663630f, 0.05178293f, 0.14202815f, 0.00903900f, 0.01098284f, 0.01514647f, 0.02373366f, -0.07681863f, 0.13505585f, 0.05740543f, 0.03685604f, 0.06126055f, -0.05916950f, 0.03168268f, 0.00370282f, -0.01733277f, -0.05590023f, -0.00064493f, -0.06932547f, 0.04630286f, 0.10074161f, -0.06776281f, -0.00591499f, -0.03683351f, -0.03264463f, -0.01630473f, 0.11118993f, 0.05249326f, 0.01859027f, 0.16716541f, -0.00241472f, -0.04309258f, -0.03769898f, -0.03376414f, -0.04652074f, 0.09897690f, 0.03965620f, 0.01581024f, 0.13286325f, -0.03214633f, -0.02498388f, -0.02913953f, -0.05628576f, -0.06845016f, 0.07658130f, 0.01465582f, 0.02320348f, 0.04508044f, -0.06325508f, -0.05508907f, -0.05695785f, -0.07049284f, -0.01813822f, 0.17532453f, 0.01948552f, -0.01399545f, 0.00656219f, -0.06356832f, -0.03064174f, -0.03332108f, -0.06732583f, -0.07101192f, 0.15528328f, -0.03675918f, 0.01138998f, 0.04304910f, -0.10519457f, -0.06679758f, -0.03240108f, -0.09980231f, -0.15034964f, 0.00969214f, 0.04635271f, 0.08812509f, 0.09690702f, -0.02393519f, -0.02450863f, -0.05752964f, -0.05706702f, -0.05933841f, 0.07637024f, 0.07193869f, 0.04471044f, 0.00301423f, -0.13040231f, -0.06762078f, -0.05109986f, -0.10586298f, -0.11737404f, 0.06935249f, -0.02274745f, 0.09391536f, 0.07696083f, -0.06216522f, -0.09566147f, -0.03089081f, -0.01121098f, -0.02106923f, 0.14213343f, 0.06176082f, 0.12673491f, 0.19389804f, 0.11606169f, 0.08879655f, -0.01058433f, 0.01370086f, 0.05366760f, 0.15122272f, 0.07383071f, -0.00121432f, 0.02932640f, 0.00167150f, -0.02731539f, 0.13191163f, -0.03080079f, -0.02238644f, 0.00484529f, -0.03383114f, 0.06710867f, 0.12565575f, 0.08343783f, 0.03354042f, 0.03246001f, 0.09092279f, 0.11099475f, 0.23847370f, 0.07365508f, -0.01669316f, -0.02397140f, -0.01063865f, -0.00189723f, 0.13745503f, -0.05701892f, -0.06685667f, 0.05740433f, -0.01038145f, 0.04073412f, 0.02733854f, 0.02807580f, 0.11124484f, 0.10677699f, 0.17168502f, 0.05010542f, 0.02330237f, 0.02997737f, -0.02447022f, -0.10076450f, -0.03835526f, -0.03466142f, -0.10071066f, -0.04029906f, -0.05908245f, -0.09732382f, 0.00550562f, 0.04810321f, 0.04635149f, 0.05728725f, 0.22300632f, 0.15077484f, 0.12061301f, 0.02941196f, 0.04962211f, 0.05334687f, 0.02432496f, -0.06947445f, -0.03697752f, 0.02245884f, -0.05557854f, -0.02820860f, -0.02275222f, -0.09149342f, -0.01244063f, 0.07084723f, -0.09058618f, -0.06572435f, 0.05786590f, 0.11238069f, 0.01216453f, -0.07584180f, -0.11065924f, 0.07602353f, 0.00174215f, -0.12226716f, -0.02116575f, -0.01864633f, 0.02116176f, 0.08258545f, 0.14487675f, 0.08691575f, 0.08098910f, 0.06885769f, -0.11429214f, -0.02579430f, 0.06625062f, 0.09788693f, 0.07044452f, -0.01786581f, -0.06924179f, 0.09454148f, 0.06877645f, -0.05546652f, 0.02120747f, 0.06644467f, 0.01614945f, -0.00859189f, -0.00549757f, 0.00025202f, 0.06692201f, 0.09250043f, -0.06346050f, -0.01031727f, -0.07399848f, 0.01721347f, 0.03607085f, 0.13101971f, 0.20358120f, 0.20070550f, 0.24087071f, 0.25927697f, 0.18642768f, -0.02111868f, 0.07219672f, 0.06503284f, 0.04711649f, 0.03229653f, 0.15892513f, 0.03773850f, -0.07641200f, -0.00926974f, -0.03753882f, 0.02248266f, -0.05437388f, -0.04476756f, -0.04433881f, 0.05913529f, 0.06503341f, 0.02168182f, 0.05645561f, 0.02247681f, 0.07451941f, -0.01111755f, -0.01938552f, -0.11140801f, 0.01543946f, 0.00459881f, 0.11099736f, 0.07504577f, -0.02293111f, 0.02551336f, 0.26612931f, 0.41033190f, 0.13975285f, -0.05802781f, -0.16583446f, -0.04821488f, -0.03609959f, -0.06468494f, -0.10183378f, 0.05789493f, 0.26758954f, 0.15301402f, -0.07990419f, -0.02202566f, -0.01196533f, 0.00680303f, -0.02362903f, 0.01435306f, 0.04306518f, 0.20147913f, 0.09041198f, -0.03414414f, -0.04288399f, -0.08007756f, -0.02916089f, -0.02729518f, -0.02856850f, 0.03540546f, 0.17873810f, 0.23873295f, 0.06695722f, -0.03241103f, 0.01101829f, -0.01295992f, -0.08032235f, -0.14641283f, -0.12533390f, -0.08483819f, 0.07593332f, 0.01567627f, 0.06596721f, -0.02060243f, -0.05516015f, -0.06243456f, -0.09131067f, -0.09004716f, -0.08013159f, 0.01612216f, -0.01284854f, 0.05043842f, 0.00134803f, 0.02894139f, 0.00549133f, -0.01556172f, -0.01657744f, 0.09203662f, 0.31330524f, 0.19634719f, 0.06113440f, 0.03809267f, 0.02792735f, 0.00972992f, -0.01078066f, -0.00219240f, 0.04099902f, 0.19724121f, 0.07953353f, -0.01474491f, -0.02475189f, -0.00175782f, 0.00772974f, -0.07483244f, -0.05302805f, -0.01325038f, 0.09171741f, 0.07030170f, -0.02005100f, -0.05180698f, -0.02737640f, -0.03119270f, -0.11730955f, -0.12999033f, -0.05609156f, 0.08651311f, 0.09467099f, 0.04733558f, 0.06745798f, 0.07529736f, 0.03583945f, 0.01340303f, 0.02876514f, 0.04255432f, 0.17634513f, 0.08032851f, 0.02205579f, 0.04638791f, 0.06089829f, -0.01013133f, -0.04300636f, -0.03126074f, 0.02240635f, 0.13021459f, 0.07382039f, 0.01382063f, 0.01971046f, 0.01516333f, 0.01099670f, -0.04779792f, -0.06458773f, -0.01561756f, 0.09036125f, 0.12421368f, 0.05922261f, 0.02450201f, -0.03569031f, -0.01004884f, -0.02595982f, -0.11301780f, -0.07314343f, -0.01659894f, 0.08706049f, 0.07271367f, 0.10065530f, 0.06971091f, 0.03769251f, -0.03435165f, -0.04505845f, 0.02883032f, 0.16415918f, 0.14894522f, 0.08834071f, 0.08326996f, 0.06317408f, 0.08046032f, 0.03380962f, -0.03514586f, -0.02357674f, 0.07053911f, 0.11453456f, 0.19680059f, 0.15169058f, -0.02655952f, 0.03760101f, 0.03404234f, -0.06210195f, -0.07319282f, -0.01204692f, 0.08590830f, 0.06636255f, 0.07992220f, -0.13233555f, -0.04495757f, -0.02271509f, -0.08700457f, -0.10103115f, 0.02239459f, 0.05079730f, 0.14780812f, 0.08936942f, 0.02417919f, 0.07165867f, 0.08869460f, 0.01224150f, -0.00201891f, 0.03683778f, 0.06237298f, 0.05905903f, -0.04650912f, -0.11496477f, -0.04627328f, 0.04029761f, -0.03622017f, -0.07192448f, 0.01123386f, -0.04970246f, 0.04003742f, 0.07566982f, -0.04386486f, -0.03875264f, 0.00219915f, -0.05609296f, -0.03531975f, 0.16679987f, 0.11906829f, 0.00187898f, 0.01303031f, -0.03081489f, -0.02550991f, 0.01178175f, -0.04963666f, -0.07013262f, 0.11409438f, 0.00201558f, -0.02505196f, -0.05606173f, -0.04653484f, -0.01953631f, 0.01269314f, -0.03033236f, -0.07193584f, 0.06007862f, 0.02574913f, -0.04919250f, -0.12044069f, -0.05595275f, -0.03987198f, 0.02277875f, -0.04858179f, -0.06871121f, 0.03237476f, 0.04687406f, 0.07612100f, 0.05876088f, -0.05979176f, -0.03782249f, 0.05256662f, -0.02810596f, -0.07412630f, 0.05449711f, 0.05378477f, 0.08491730f, 0.09283547f, -0.02625268f, -0.03622905f, 0.03617854f, -0.00180482f, -0.00140366f, 0.08438043f, 0.06185229f, -0.01989243f, -0.13081157f, -0.07889940f, -0.04496945f, 0.05362119f, -0.02829479f, -0.02842441f, -0.01147963f, 0.04573714f, -0.07196586f, -0.08775458f, -0.09727339f, -0.06411818f, -0.02729066f, -0.09266587f, -0.07088945f, -0.03004998f, -0.02544358f, 0.11181971f, 0.20729888f, 0.15703810f, 0.15115232f, 0.16108008f, 0.19429715f, 0.21407219f, 0.29699633f, 0.12767384f, 0.02780656f, 0.05611593f, 0.07670163f, 0.15832298f, 0.17579922f, 0.05100116f, 0.01224768f, 0.08070841f, 0.03465386f, -0.00514099f, 0.02356194f, 0.01485237f, 0.03287734f, 0.01764238f, -0.01239792f, 0.00129261f, 0.03188046f, -0.01240933f, -0.10373565f, -0.09503219f, -0.06748286f, -0.06755011f, -0.08161656f, -0.11374162f, -0.11032314f, -0.13528521f, -0.10929562f, -0.00129230f, 0.00181152f, 0.06726545f, 0.22484158f, 0.06353094f, 0.07181604f, 0.01091729f, 0.01044138f, -0.01399836f, 0.01633802f, -0.08620252f, -0.00164358f, 0.05904337f, -0.06693273f, -0.01430601f, -0.02028281f, -0.10923157f, -0.02298127f, 0.04644550f, 0.01981420f, 0.08687677f, 0.12439938f, 0.02171628f, 0.06094162f, 0.01450324f, 0.00349182f, 0.01785915f, 0.03922000f, -0.09102815f, -0.01552420f, 0.03794057f, -0.00605644f, 0.08889448f, 0.00211633f, -0.08214331f, 0.04772318f, 0.09736003f, -0.08090471f, -0.04318808f, 0.04059591f, 0.05186099f, 0.03286642f, -0.01873678f, -0.06900043f, 0.10412179f, 0.08784681f, -0.00915221f, 0.00769984f, 0.08768112f, 0.01220177f, 0.01989234f, -0.00508515f, -0.03124082f, 0.07537845f, 0.10901033f, -0.07698010f, -0.06695887f, 0.10171302f, 0.08196348f, 0.04550446f, -0.02453868f, -0.06374582f, 0.09811466f, 0.13033564f, 0.02265015f, 0.00782516f, 0.06940789f, -0.01020320f, 0.08192433f, 0.04089277f, -0.01719946f, 0.09826740f, 0.01794209f, -0.03115428f, -0.00121978f, 0.01683556f, 0.06092347f, 0.01156501f, 0.00310840f, -0.05735964f, 0.04303766f, -0.01243530f, 0.01251837f, 0.05535906f, 0.04232094f, 0.05505496f, -0.00490809f, -0.04755514f, -0.09569632f, -0.00895425f, 0.05853438f, 0.01689393f, 0.00530152f, 0.02433566f, 0.04237611f, 0.04203849f, 0.01051859f, -0.01012976f, 0.09364594f, 0.06593708f, 0.03187219f, 0.03188266f, 0.01977709f, 0.01749906f, 0.05289435f, 0.02029400f, 0.04453890f, 0.10389563f, 0.03478450f, 0.08266787f, 0.07079163f, 0.05935082f, 0.06820928f, -0.04711834f, 0.01070640f, -0.00638519f, -0.00600724f, -0.02537994f, 0.02832212f, 0.04866460f, 0.03968958f, 0.04907361f, -0.03383192f, -0.03905835f, 0.05176930f, -0.03280682f, 0.05635149f, 0.03723919f, 0.05319552f, 0.01605161f, 0.03294128f, -0.00121157f, 0.00076042f, 0.03833876f, 0.05624065f, -0.03702525f, -0.00347923f, 0.01699932f, 0.00053627f, -0.00654527f, -0.04319109f, -0.04588608f, -0.03064336f, -0.03008819f, -0.02485351f, 0.05914138f, 0.04489624f, 0.05452225f, 0.12457912f, -0.00058330f, -0.02167312f, 0.14463240f, 0.08760407f, -0.04406744f, -0.00808849f, -0.01413483f, 0.00459099f, 0.06803548f, -0.00638441f, -0.05153438f, 0.07695373f, 0.02150755f, 0.00072168f, 0.05892925f, 0.02512629f, 0.00932163f, 0.07436098f, -0.02199609f, -0.03623645f, -0.01802865f, 0.01050595f, -0.03777776f, 0.04348689f, -0.03035584f, -0.01700055f, 0.05019889f, -0.02912215f, -0.10911976f, -0.05940097f, -0.05066778f, 0.03487982f, 0.06586347f, 0.03999907f, 0.04844084f, 0.13367932f, 0.03109732f, 0.00404561f, 0.10503622f, 0.07671331f, -0.02992229f, 0.05512905f, -0.01875147f, -0.00852659f, 0.05392998f, 0.01158351f, -0.02554222f, 0.00858335f, 0.00350628f, 0.01516915f, 0.04952117f, -0.01741584f, 0.01642536f, 0.07525610f, 0.03355637f, -0.05060860f, -0.01665226f, -0.00884436f, -0.02225338f, 0.02599719f, -0.07183378f, -0.03284854f, 0.03837427f, 0.01094718f, -0.07389339f, -0.05218883f, -0.01724777f, 0.06342028f, 0.04134783f, 0.02545394f, 0.07108847f, 0.07294786f, 0.08344634f, 0.04042597f, 0.06488745f, 0.08596324f, -0.00961856f, -0.00226815f, 0.01200549f, 0.05573589f, 0.03554142f, 0.03586485f, 0.01781382f, 0.01528289f, 0.01836091f, 0.03391675f, 0.05324844f, -0.03720093f, 0.01504477f, 0.05952165f, 0.07287916f, -0.02238517f, 0.02175206f, -0.00724353f, -0.04385042f, 0.02896211f, -0.08177772f, -0.03364721f, -0.07496103f, -0.00400536f, -0.05017489f, -0.04011143f, -0.04885572f, -0.00210436f, -0.00451476f, -0.00266716f, 0.10276304f, 0.08771757f, 0.08811594f, 0.05431274f, 0.05126261f, 0.05482224f, -0.04799624f, -0.06162736f, -0.05021211f, -0.01347766f, 0.04800192f, 0.02853259f, 0.02159122f, 0.03028378f, -0.03585166f, -0.00772628f, 0.02417052f, -0.05592563f, -0.00958208f, -0.02110092f, 0.06549526f, 0.00179113f, 0.01321263f, 0.00394119f, -0.00297596f, 0.01405040f, -0.10311706f, -0.06387822f, -0.01017533f, -0.01889544f, -0.05025611f, 0.00338155f, -0.04375202f, -0.02523173f, 0.02004614f, 0.03953922f, 0.07898285f, 0.00344118f, 0.06185938f, 0.01857499f, 0.07381004f, 0.02716019f, -0.04619727f, -0.00937351f, 0.00597590f, 0.06570271f, 0.04512402f, 0.03046790f, 0.00374561f, 0.03426301f, -0.04934379f, 0.05696843f, 0.07108632f, -0.03085425f, 0.00827079f, -0.03439316f, 0.01509693f, -0.04257619f, 0.01591038f, -0.00820132f, -0.06634856f, 0.00985586f, -0.05719750f, -0.04128314f, -0.02535608f, -0.01446960f, -0.05906772f, -0.04931477f, -0.08221318f, -0.03915311f, -0.01601552f, 0.01640273f, 0.08630029f, 0.10634460f, 0.09872271f, 0.04742125f, 0.03056765f, 0.00408714f, -0.05219893f, 0.01898822f, -0.00097093f, 0.01787088f, 0.01850866f, 0.08598141f, 0.00710775f, 0.03806509f, -0.01721473f, -0.00168339f, 0.04045156f, -0.03048587f, 0.01154933f, -0.00700005f, 0.03405448f, 0.00863478f, 0.04511243f, -0.02488127f, -0.03416001f, 0.01781376f, -0.02258551f, -0.01876023f, -0.03341861f, 0.01161659f, -0.02736058f, -0.00508410f, -0.06083047f, 0.01400521f, 0.11427422f, 0.07918878f, 0.12130253f, 0.10955977f, 0.13819106f, 0.06877220f, 0.06188086f, 0.02928701f, 0.01930980f, 0.04404383f, 0.03463696f, 0.07062077f, 0.00540550f, 0.02610247f, 0.00590315f, -0.01577813f, -0.01995169f, -0.00076754f, 0.07143968f, 0.01953625f, 0.05275492f, 0.01072872f, 0.07237665f, 0.03214786f, 0.04419057f, -0.04929715f, -0.04405805f, -0.00377078f, -0.07279358f, -0.00651045f, -0.04007916f, -0.00465116f, -0.01413242f, 0.00356408f, -0.04751017f, 0.08166426f, 0.00763876f, 0.02219428f, 0.05486224f, -0.05782322f, 0.05976040f, 0.03713590f, -0.00572788f, 0.01317427f, 0.09654777f, -0.05017064f, -0.02679111f, 0.01214870f, -0.03324833f, 0.09242442f, 0.01888214f, -0.05948557f, 0.09849026f, 0.02471538f, -0.02074307f, -0.03000445f, 0.06661420f, -0.01546718f, 0.09014191f, 0.05581454f, 0.03794374f, 0.01670722f, 0.08308319f, -0.05664399f, -0.01919703f, 0.06205443f, -0.03550056f, 0.13205627f, 0.06595257f, -0.03317502f, 0.05507434f, -7.15898f };

	return std::vector<float>(detector, detector + sizeof(detector) / sizeof(detector[0]));
}

std::vector<float> HOGDescriptor_Mod::HOG_Optimal_32_64()
{
	static const float detector[] = {0.12894585f, -0.13370498f, -0.12836904f, -0.00298548f, 0.11705760f, 0.05017139f, -0.08695133f, -0.06372724f, 0.15756414f, 0.11224895f, -0.05474196f, -0.03049062f, 0.03059517f, 0.16135692f, 0.15967069f, 0.11201413f, 0.09058776f, 0.17114211f, 0.12493896f, -0.03796765f, 0.00899631f, 0.08093542f, 0.09405639f, -0.03316300f, -0.18823609f, -0.20982268f, 0.08140266f, 0.24059861f, 0.09555875f, 0.13205918f, 0.06751004f, -0.00801431f, 0.04279287f, 0.06143641f, 0.12118561f, 0.25703750f, -0.04891258f, -0.01480657f, -0.02723828f, -0.06710277f, 0.09291933f, 0.02388697f, 0.06160094f, 0.04382448f, -0.01325442f, 0.13693780f, 0.21113624f, 0.01488789f, -0.06685515f, 0.10961065f, 0.03208507f, 0.05474491f, 0.05544661f, 0.03218453f, 0.02926576f, 0.23562029f, 0.41503235f, 0.17575923f, -0.01138443f, -0.04759770f, 0.03835360f, 0.08020230f, 0.03978895f, 0.09635123f, 0.34408931f, 0.38902363f, 0.06895414f, -0.03217845f, -0.09289941f, -0.06749404f, -0.09796774f, -0.03341203f, 0.18514362f, 0.20508783f, -0.05095470f, -0.00759454f, 0.16409453f, 0.05878464f, 0.05557542f, 0.09870477f, 0.06411455f, 0.10137698f, 0.10907063f, -0.05615081f, 0.05007936f, 0.10250659f, -0.02899476f, 0.01226257f, 0.07528311f, 0.04915636f, 0.21843983f, 0.27657870f, -0.03519693f, -0.16345819f, -0.14990675f, -0.08430281f, 0.02381669f, 0.02442686f, 0.04358394f, 0.15944198f, 0.14219304f, -0.04039334f, -0.09442776f, -0.15752760f, -0.09180774f, 0.04633692f, 0.06230943f, 0.05764179f, 0.14814075f, 0.11446327f, -0.00772635f, 0.05074353f, 0.08954830f, -0.00587021f, 0.01879326f, 0.12784925f, 0.13203390f, 0.01781093f, 0.02596584f, -0.05506870f, 0.02468239f, 0.07598807f, -0.00054164f, -0.02704913f, -0.00530741f, 0.00994232f, 0.19867103f, 0.20480051f, 0.04321557f, -0.03446068f, -0.04674964f, 0.03415174f, 0.15230189f, 0.19786865f, 0.10929322f, 0.11448580f, 0.01471795f, -0.13239962f, -0.05776739f, 0.02462651f, 0.01164128f, 0.11447236f, 0.21759051f, 0.12787729f, 0.03075300f, 0.02472044f, -0.05093864f, 0.04616588f, 0.18366560f, 0.05191755f, -0.06450538f, -0.04435196f, -0.02540611f, -0.14044365f, -0.02717482f, -0.11765455f, -0.01453278f, 0.15542853f, 0.03889475f, -0.11929038f, -0.12308952f, -0.19652047f, 0.22610798f, 0.12024472f, -0.08790319f, -0.00927309f, 0.10215207f, 0.01491322f, 0.01007446f, 0.28680928f, 0.22626908f, 0.05454608f, 0.12062005f, -0.11973171f, -0.09148014f, 0.04434018f, -0.05939488f, -0.12779472f, 0.07907449f, 0.06486546f, -0.07987915f, 0.03502274f, 0.01349961f, 0.07588979f, 0.18687017f, 0.04538053f, -0.03843862f, -0.06431571f, -0.13478265f, -0.21096554f, -0.09562742f, -0.09455998f, -0.04944860f, -0.00730340f, -0.11926300f, -0.06719358f, -0.09799459f, -0.21219903f, 0.15144404f, 0.15434729f, -0.03555737f, -0.04728743f, 0.09058720f, -0.01622953f, -0.04627473f, 0.10772297f, 0.20866718f, 0.13964316f, 0.14318393f, 0.02606120f, -0.01329439f, 0.02190431f, -0.03190881f, -0.04094209f, 0.10715544f, 0.10254680f, -0.06851100f, -0.02102214f, -0.00908732f, 0.00405626f, 0.06188881f, -0.00314207f, 0.05702314f, 0.03314749f, 0.00204613f, -0.03172231f, -0.07141891f, -0.04686181f, -0.08816220f, -0.19449704f, -0.09851901f, -0.05724229f, -0.08590471f, 0.04349303f, 0.22901641f, 0.21020104f, 0.19968805f, 0.18722273f, 0.17946551f, 0.30460834f, 0.22667358f, 0.24913307f, 0.18099031f, -0.06326475f, -0.17489991f, -0.10574027f, -0.04146709f, 0.01844105f, 0.12165348f, 0.04093351f, -0.11439343f, -0.06083938f, -0.00624532f, -0.07102198f, 0.11162334f, 0.18589658f, 0.18477818f, 0.02675974f, -0.06926657f, -0.21637755f, -0.08885158f, 0.20609990f, 0.27901206f, 0.36148506f, 0.22048686f, -0.00339441f, 0.07239064f, 0.21924520f, 0.22893867f, 0.17561984f, -0.08117594f, -0.26403353f, -0.11071107f, 0.05582651f, 0.16492742f, 0.20740809f, 0.17260408f, -0.00444470f, -0.03009572f, 0.22015106f, 0.17989858f, 0.15953382f, 0.08945950f, 0.02190766f, 0.18320739f, 0.36974291f, 0.30037756f, 0.17326369f, 0.06053552f, 0.30629745f, 0.51396306f, 0.35651598f, 0.09041767f, 0.17410491f, 0.28919405f, 0.24457261f, 0.05231103f, -0.29722145f, -0.00858001f, 0.20176841f, 0.07201379f, -0.00888572f, -0.02890043f, -0.08217842f, -0.23008409f, -0.41235184f, 0.11121297f, 0.23903015f, 0.25554826f, 0.16724595f, 0.08328586f, 0.33152244f, 0.53461428f, 0.30627401f, 0.05908586f, -0.36025426f, -0.20296232f, -0.08448214f, -0.08658518f, -0.02221800f, 0.14193541f, 0.27395044f, -0.00193060f, -0.24685403f, 0.12650490f, 0.15128110f, 0.02432520f, -0.02940502f, -0.05786868f, 0.00581568f, 0.01764999f, 0.03699572f, 0.01960695f, 0.07198520f, 0.06617642f, -0.01206945f, 0.02327935f, 0.02009924f, 0.04988352f, 0.03230651f, -0.00796560f, -0.00043744f, 0.05793970f, 0.05799602f, 0.03261250f, -0.01567642f, -0.05916625f, -0.05811241f, 0.00077421f, 0.16607084f, 0.13187736f, 0.02077745f, -0.00453578f, 0.01178770f, 0.05371428f, 0.01071995f, -0.03845902f, -0.03361013f, 0.05943915f, 0.03840321f, -0.11635178f, 0.01302144f, -0.00081595f, 0.15366470f, 0.21765525f, 0.18745833f, 0.09505152f, 0.00202956f, -0.14959666f, -0.03756436f, -0.08213170f, -0.10747672f, 0.09704818f, 0.21643508f, 0.16042759f, 0.09817909f, 0.15742134f, 0.02923665f, -0.06486861f, 0.00761103f, 0.07173188f, 0.20852031f, 0.21312940f, 0.09475795f, 0.01340503f, 0.06180945f, -0.04277145f, 0.02521012f, 0.18125921f, 0.10100851f, 0.19727726f, 0.23816700f, 0.05986642f, -0.08767899f, -0.04278916f, 0.06050332f, 0.04620085f, 0.00256551f, -0.14110323f, -0.06870544f, -0.03751270f, -0.05048855f, -0.09209834f, 0.13184251f, 0.13760268f, 0.11571147f, 0.22597099f, -0.06558477f, -0.12064328f, -0.06911096f, -0.12326863f, -0.08339368f, 0.21015050f, 0.16898585f, 0.14903720f, 0.18053735f, -0.06945252f, -0.05810344f, -0.04380760f, -0.09927080f, -0.15015157f, 0.00048135f, 0.10676240f, 0.10043955f, 0.20171912f, -0.05615288f, -0.14032860f, -0.07229423f, -0.14322302f, -0.07919409f, 0.22088346f, 0.05359741f, 0.11316498f, 0.13431920f, -0.11541351f, -0.19837082f, -0.19095101f, -0.18360510f, -0.15337754f, 0.07552248f, 0.17034924f, 0.19247800f, 0.20709818f, 0.03788904f, -0.11066652f, -0.12771870f, -0.12275497f, -0.02162630f, 0.15466303f, 0.15923950f, 0.10913715f, 0.04727313f, -0.14108435f, -0.20856354f, -0.19376887f, -0.17637368f, -0.13216312f, 0.10448082f, 0.05735636f, 0.13189635f, 0.16813161f, -0.01566518f, -0.12772974f, -0.08535947f, -0.09034676f, 0.01718101f, 0.19779176f, 0.13510024f, 0.20258057f, 0.20003466f, 0.16749129f, 0.17140419f, 0.27719951f, 0.22821799f, 0.19738170f, 0.22103412f, 0.18507534f, -0.20877795f, -0.24331190f, -0.20153680f, -0.10819995f, 0.11435711f, -0.01313106f, -0.06685130f, -0.20158173f, -0.19221264f, 0.20721281f, 0.25093550f, 0.21894025f, 0.26838988f, 0.22153383f, 0.12176040f, 0.17944683f, 0.25423782f, 0.22691593f, -0.21705298f, -0.19261998f, -0.09205906f, 0.02740143f, 0.11725876f, -0.15472680f, -0.21834262f, -0.26708084f, -0.20906886f, 0.07690592f, -0.20564023f, -0.16533164f, -0.04942225f, 0.06197254f, 0.01896226f, -0.01893829f, -0.02925711f, 0.10347308f, 0.26004343f, 0.10610000f, 0.05022434f, 0.01660705f, -0.04066994f, 0.03044161f, 0.06271234f, 0.03264578f, 0.16717487f, 0.20399700f, -0.04644022f, -0.04305985f, 0.10101468f, 0.12380756f, 0.01445636f, -0.10727015f, -0.09402632f, 0.16512634f, 0.19462738f, 0.10555939f, 0.13880197f, 0.18093231f, 0.13742259f, 0.09407714f, -0.03217260f, -0.06829398f, 0.11998099f, 0.03692400f, 0.08162452f, 0.06449282f, -0.06964641f, -0.02268953f, 0.12325884f, 0.37112092f, 0.18284512f, -0.02103538f, -0.04939413f, -0.03408766f, -0.00359159f, -0.10881089f, -0.07824212f, 0.01478395f, 0.38349979f, 0.36809955f, 0.14622198f, 0.04672673f, 0.03828134f, 0.10956928f, 0.03375399f, 0.08938259f, -0.01555302f, -0.00509316f, -0.02841177f, -0.03263884f, 0.02027876f, 0.04329844f, 0.06241629f, 0.03492914f, 0.06946055f, -0.04881541f, 0.03715324f, 0.16938998f, 0.09899293f, 0.04355277f, 0.02697374f, -0.00502609f, -0.08538337f, -0.14201144f, -0.19583916f, -0.04130796f, 0.32117708f, 0.24112015f, 0.04516666f, 0.04500584f, -0.01593412f, -0.09223982f, -0.11106034f, -0.16001338f, -0.06553175f, 0.18073719f, 0.11970690f, 0.06600726f, 0.08984960f, 0.03555800f, 0.06272764f, 0.16239328f, 0.02981734f, -0.03429912f, 0.18733464f, 0.15119686f, 0.07512144f, 0.07709434f, 0.01156111f, 0.04562415f, 0.13256378f, 0.05326757f, -0.04676104f, 0.07802777f, 0.05757949f, 0.13162984f, 0.15538001f, 0.09208372f, 0.02937477f, -0.00503333f, -0.09540808f, 0.01197033f, 0.25751893f, 0.23415298f, 0.15568677f, 0.20471290f, 0.10724316f, 0.02861861f, 0.01428135f, -0.09777993f, -0.12721296f, 0.04963869f, 0.16353283f, 0.13442316f, 0.09257193f, 0.00320904f, 0.03264154f, 0.13857973f, 0.05022560f, -0.02509907f, 0.05771366f, 0.12022047f, 0.00858492f, -0.02335403f, -0.03984610f, -0.01304522f, 0.02884178f, -0.03717620f, -0.10297049f, -0.01529948f, 0.00884758f, 0.22138184f, 0.26138835f, 0.05125682f, 0.04089680f, 0.08109533f, -0.01422475f, -0.05706362f, 0.13915002f, 0.20502530f, 0.02436984f, 0.03235583f, -0.11080711f, -0.02918696f, 0.08267855f, -0.02726478f, -0.05944939f, 0.13788587f, -0.00026718f, -0.04067007f, -0.03952638f, -0.02275045f, 0.06943454f, 0.14071782f, 0.04436264f, -0.06760966f, 0.03115653f, 0.01321854f, -0.21900587f, -0.14544032f, -0.12400691f, 0.03334311f, 0.10490047f, 0.03659610f, -0.09175741f, -0.02771081f, -0.13744265f, 0.15704691f, 0.08293962f, -0.03531798f, 0.04091253f, 0.14489622f, 0.05652955f, 0.00180826f, 0.16860828f, 0.11432795f, 0.07916520f, 0.10197580f, -0.01831297f, -0.03520693f, 0.02614939f, 0.01980905f, -0.00531623f, 0.11770048f, 0.10393029f, -0.14001341f, -0.08328097f, -0.07645778f, 0.07730555f, 0.12172365f, 0.10527507f, 0.00059017f, 0.01547307f, -0.07598742f, -0.21285276f, -0.12150876f, -0.11706585f, -0.12243214f, -0.06600648f, -0.03212898f, -0.09528874f, -0.09160268f, -0.17967942f, 0.20024316f, 0.24723662f, 0.21460562f, 0.29680646f, 0.16640589f, 0.12226175f, 0.15615457f, 0.23583682f, 0.22048506f, -0.08053712f, -0.09909153f, 0.01170506f, 0.12024205f, 0.02408723f, -0.07600577f, -0.10328296f, -0.12815773f, -0.08587341f, 0.01506935f, 0.01143858f, -0.01717932f, -0.00755263f, 0.03935153f, 0.06566999f, 0.03881857f, 0.00890241f, -0.01230535f, 0.01115247f, -0.09828417f, -0.10833244f, -0.07551555f, -0.11087710f, -0.04889930f, -0.04228393f, -0.10576238f, -0.02107343f, -6.278353f };

	return std::vector<float>(detector, detector + sizeof(detector) / sizeof(detector[0]));
}

class HOGConfInvoker :
    public ParallelLoopBody
{
public:
    HOGConfInvoker( const HOGDescriptor_Mod* _hog, const Mat& _img,
        double _hitThreshold, const Size& _padding,
        std::vector<DetectionROI>* locs,
        std::vector<Rect>* _vec, Mutex* _mtx )
    {
        hog = _hog;
        img = _img;
        hitThreshold = _hitThreshold;
        padding = _padding;
        locations = locs;
        vec = _vec;
        mtx = _mtx;
    }

    void operator()( const Range& range ) const
    {
        int i, i1 = range.start, i2 = range.end;

        Size maxSz(cvCeil(img.cols/(*locations)[0].scale), cvCeil(img.rows/(*locations)[0].scale));
        Mat smallerImgBuf(maxSz, img.type());
        std::vector<Point> dets;

        for( i = i1; i < i2; i++ )
        {
            double scale = (*locations)[i].scale;

            Size sz(cvRound(img.cols / scale), cvRound(img.rows / scale));
            Mat smallerImg(sz, img.type(), smallerImgBuf.ptr());

            if( sz == img.size() )
                smallerImg = Mat(sz, img.type(), img.data, img.step);
            else
                resize(img, smallerImg, sz);

            hog->detectROI(smallerImg, (*locations)[i].locations, dets, (*locations)[i].confidences, hitThreshold, Size(), padding);
            Size scaledWinSize = Size(cvRound(hog->winSize.width*scale), cvRound(hog->winSize.height*scale));
            mtx->lock();
            for( size_t j = 0; j < dets.size(); j++ )
                vec->push_back(Rect(cvRound(dets[j].x*scale),
                                    cvRound(dets[j].y*scale),
                                    scaledWinSize.width, scaledWinSize.height));
            mtx->unlock();
        }
    }

    const HOGDescriptor_Mod* hog;
    Mat img;
    double hitThreshold;
    std::vector<DetectionROI>* locations;
    Size padding;
    std::vector<Rect>* vec;
    Mutex* mtx;
};

void HOGDescriptor_Mod::detectROI(const cv::Mat& img, const std::vector<cv::Point> &locations,
    CV_OUT std::vector<cv::Point>& foundLocations, CV_OUT std::vector<double>& confidences,
    double hitThreshold, cv::Size winStride, cv::Size padding) const
{
    foundLocations.clear();
    confidences.clear();

    if( svmDetector.empty() || locations.empty())
        return;

    if( winStride == Size() )
        winStride = cellSize;
    Size cacheStride(gcd(winStride.width, blockStride.width),
                     gcd(winStride.height, blockStride.height));

    size_t nwindows = locations.size();
    padding.width = (int)alignSize(std::max(padding.width, 0), cacheStride.width);
    padding.height = (int)alignSize(std::max(padding.height, 0), cacheStride.height);
    Size paddedImgSize(img.cols + padding.width*2, img.rows + padding.height*2);

    // HOGCache cache(this, img, padding, padding, nwindows == 0, cacheStride);
    HOGCache cache(this, img, padding, padding, true, cacheStride);
    if( !nwindows )
        nwindows = cache.windowsInImage(paddedImgSize, winStride).area();

    const HOGCache::BlockData* blockData = &cache.blockData[0];

    int nblocks = cache.nblocks.area();
    int blockHistogramSize = cache.blockHistogramSize;
    size_t dsize = getDescriptorSize();

    double rho = svmDetector.size() > dsize ? svmDetector[dsize] : 0;
    std::vector<float> blockHist(blockHistogramSize);

#ifdef UseSSE
    float partSum[4];
#endif

    for( size_t i = 0; i < nwindows; i++ )
    {
        Point pt0;
        pt0 = locations[i];
        if( pt0.x < -padding.width || pt0.x > img.cols + padding.width - winSize.width ||
                pt0.y < -padding.height || pt0.y > img.rows + padding.height - winSize.height )
        {
            // out of image
            confidences.push_back(-10.0);
            continue;
        }

        double s = rho;
        const float* svmVec = &svmDetector[0];
        int j, k;

        for( j = 0; j < nblocks; j++, svmVec += blockHistogramSize )
        {
            const HOGCache::BlockData& bj = blockData[j];
            Point pt = pt0 + bj.imgOffset;

            // need to devide this into 4 parts!
            const float* vec = cache.getBlock(pt, &blockHist[0]);
#ifdef UseSSE
            __m128 _vec = _mm_loadu_ps(vec);
            __m128 _svmVec = _mm_loadu_ps(svmVec);
            __m128 sum = _mm_mul_ps(_svmVec, _vec);

            for( k = 4; k <= blockHistogramSize - 4; k += 4 )
            {
                _vec = _mm_loadu_ps(vec + k);
                _svmVec = _mm_loadu_ps(svmVec + k);

                sum = _mm_add_ps(sum, _mm_mul_ps(_vec, _svmVec));
            }

            _mm_storeu_ps(partSum, sum);
            double t0 = partSum[0] + partSum[1];
            double t1 = partSum[2] + partSum[3];
            s += t0 + t1;
#else
            for( k = 0; k <= blockHistogramSize - 4; k += 4 )
                s += vec[k]*svmVec[k] + vec[k+1]*svmVec[k+1] +
                        vec[k+2]*svmVec[k+2] + vec[k+3]*svmVec[k+3];
#endif
            for( ; k < blockHistogramSize; k++ )
                s += vec[k]*svmVec[k];
        }
        confidences.push_back(s);

        if( s >= hitThreshold )
            foundLocations.push_back(pt0);
    }
}

void HOGDescriptor_Mod::detectMultiScaleROI(const cv::Mat& img,
    CV_OUT std::vector<cv::Rect>& foundLocations, std::vector<DetectionROI>& locations,
    double hitThreshold, int groupThreshold) const
{
    std::vector<Rect> allCandidates;
    Mutex mtx;

    parallel_for_(Range(0, (int)locations.size()),
                  HOGConfInvoker(this, img, hitThreshold, Size(8, 8),
                                 &locations, &allCandidates, &mtx));

    foundLocations.resize(allCandidates.size());
    std::copy(allCandidates.begin(), allCandidates.end(), foundLocations.begin());
    cv::groupRectangles(foundLocations, groupThreshold, 0.2);
}

void HOGDescriptor_Mod::readALTModel(String modelfile)
{
    // read model from SVMlight format..
    FILE *modelfl;
    if ((modelfl = fopen(modelfile.c_str(), "rb")) == NULL)
    {
        String eerr("file not exist");
        String efile(__FILE__);
        String efunc(__FUNCTION__);
        throw Exception(Error::StsError, eerr, efile, efunc, __LINE__);
    }
    char version_buffer[10];
    if (!fread (&version_buffer,sizeof(char),10,modelfl))
    {
        String eerr("version?");
        String efile(__FILE__);
        String efunc(__FUNCTION__);
        throw Exception(Error::StsError, eerr, efile, efunc, __LINE__);
    }
    if(strcmp(version_buffer,"V6.01")) {
        String eerr("version doesnot match");
        String efile(__FILE__);
        String efunc(__FUNCTION__);
        throw Exception(Error::StsError, eerr, efile, efunc, __LINE__);
    }
    /* read version number */
    int version = 0;
    if (!fread (&version,sizeof(int),1,modelfl))
    { throw Exception(); }
    if (version < 200)
    {
        String eerr("version doesnot match");
        String efile(__FILE__);
        String efunc(__FUNCTION__);
        throw Exception();
    }
    int kernel_type;
    size_t nread;
    nread=fread(&(kernel_type),sizeof(int),1,modelfl);

    {// ignore these
        int poly_degree;
        nread=fread(&(poly_degree),sizeof(int),1,modelfl);

        double rbf_gamma;
        nread=fread(&(rbf_gamma),sizeof(double), 1, modelfl);
        double coef_lin;
        nread=fread(&(coef_lin),sizeof(double),1,modelfl);
        double coef_const;
        nread=fread(&(coef_const),sizeof(double),1,modelfl);
        int l;
        nread=fread(&l,sizeof(int),1,modelfl);
        char* custom = new char[l];
        nread=fread(custom,sizeof(char),l,modelfl);
        delete[] custom;
    }
    int totwords;
    nread=fread(&(totwords),sizeof(int),1,modelfl);
    {// ignore these
        int totdoc;
        nread=fread(&(totdoc),sizeof(int),1,modelfl);
        int sv_num;
        nread=fread(&(sv_num), sizeof(int),1,modelfl);
    }

    double linearbias;
    nread=fread(&linearbias, sizeof(double), 1, modelfl);

    std::vector<float> detector;
    detector.clear();
    if(kernel_type == 0) { /* linear kernel */
        /* save linear wts also */
        double *linearwt = new double[totwords+1];
        int length = totwords;
        nread = fread(linearwt, sizeof(double), totwords + 1, modelfl);
        if(nread != static_cast<size_t>(length) + 1) {
            delete [] linearwt;
            throw Exception();
        }

        for(int i = 0; i < length; i++)
            detector.push_back((float)linearwt[i]);

        detector.push_back((float)-linearbias);
        setSVMDetector(detector);
        delete [] linearwt;
    } else {
        throw Exception();
    }
    fclose(modelfl);
}

void HOGDescriptor_Mod::groupRectangles(std::vector<cv::Rect>& rectList, std::vector<double>& weights, int groupThreshold, double eps) const
{
    if( groupThreshold <= 0 || rectList.empty() )
    {
        return;
    }

    CV_Assert(rectList.size() == weights.size());

    std::vector<int> labels;
    int nclasses = partition(rectList, labels, SimilarRects(eps));

    std::vector<cv::Rect_<double> > rrects(nclasses);
    std::vector<int> numInClass(nclasses, 0);
    std::vector<double> foundWeights(nclasses, -std::numeric_limits<double>::max());
    int i, j, nlabels = (int)labels.size();

    for( i = 0; i < nlabels; i++ )
    {
        int cls = labels[i];
        rrects[cls].x += rectList[i].x;
        rrects[cls].y += rectList[i].y;
        rrects[cls].width += rectList[i].width;
        rrects[cls].height += rectList[i].height;
        foundWeights[cls] = max(foundWeights[cls], weights[i]);
        numInClass[cls]++;
    }

    for( i = 0; i < nclasses; i++ )
    {
        // find the average of all ROI in the cluster
        cv::Rect_<double> r = rrects[i];
        double s = 1.0/numInClass[i];
        rrects[i] = cv::Rect_<double>(cv::saturate_cast<double>(r.x*s),
            cv::saturate_cast<double>(r.y*s),
            cv::saturate_cast<double>(r.width*s),
            cv::saturate_cast<double>(r.height*s));
    }

    rectList.clear();
    weights.clear();

    for( i = 0; i < nclasses; i++ )
    {
        cv::Rect r1 = rrects[i];
        int n1 = numInClass[i];
        double w1 = foundWeights[i];
        if( n1 <= groupThreshold )
            continue;
        // filter out small rectangles inside large rectangles
        for( j = 0; j < nclasses; j++ )
        {
            int n2 = numInClass[j];

            if( j == i || n2 <= groupThreshold )
                continue;

            cv::Rect r2 = rrects[j];

            int dx = cv::saturate_cast<int>( r2.width * eps );
            int dy = cv::saturate_cast<int>( r2.height * eps );

            if( r1.x >= r2.x - dx &&
                r1.y >= r2.y - dy &&
                r1.x + r1.width <= r2.x + r2.width + dx &&
                r1.y + r1.height <= r2.y + r2.height + dy &&
                (n2 > std::max(3, n1) || n1 < 3) )
                break;
        }

        if( j == nclasses )
        {
            rectList.push_back(r1);
            weights.push_back(w1);
        }
    }
}

void hog_mod_test(void) {
	Mat img;
	HOGDescriptor_Mod hog_mod;
	hog_mod.winSize = Size(48, 96);
//	hog_mod.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
//	hog_mod.setSVMDetector(HOGDescriptor_Mod::HOG_Optimal_64_128());
//	hog_mod.setSVMDetector(HOGDescriptor_Mod::HOG_Optimal_48_120());
	hog_mod.setSVMDetector(HOGDescriptor_Mod::HOG_Optimal_48_96());
	hog_mod.SVM_Eval_Method = hog_mod.SVM_Dot_Product;

	vector<Rect> found;
	vector<double> scores;
	char sc[10];
	VideoCapture capture;
	capture.open(0);//"E:\\RnD\\Current_Projects\\Musawwir\\Frameworks\\SW\\Dataset\\Person\\others\\HALLWAY_A.mpg");// 
	float fps = 9.5, proc_t = 105;
	char str[50];
//	setNumThreads(0);

	while (capture.read(img)) {
		//resize(img, img, Size(320, 240));
		ftime(&t_start);
		hog_mod.detectMultiScale(img, found, scores, 0, Size(8, 8), Size(32, 32), 1.05, 2);
		ftime(&t_end);
		float msec = int((t_end.time - t_start.time) * 1000 + (t_end.millitm - t_start.millitm));
		proc_t = 0.99*proc_t + 0.01*msec;
		printf("\nfps = %.1f\ttime = %.1f", 1000 / msec, proc_t);
		fps = 1000 / proc_t;
		sprintf(str, "%.1f", fps);
		cv::putText(img, str, Point(20, 20), CV_FONT_HERSHEY_COMPLEX, 1, Scalar(255, 0, 0), 1, 8);

		size_t i;
		for (i = 0; i < found.size(); i++)
		{
			Rect r = found[i];
			rectangle(img, Point(found[i].x, found[i].y), Point(found[i].x + found[i].width, found[i].y + found[i].height), cv::Scalar(0, 255, 100), 3);
			sprintf(sc, "%.02f", scores[i]);
			cv::putText(img, sc, Point(found[i].x, found[i].y), CV_FONT_HERSHEY_COMPLEX, 1, Scalar::all(255), 1, 8);
		}
		imshow("HOG_Mod", img);
		waitKey(1);
	}
}

