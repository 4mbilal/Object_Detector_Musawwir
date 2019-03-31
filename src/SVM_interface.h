#include "math.h"
#include "Defines.h"
//#include "Musawwir.h"
//
#include "Image_Utilities.h"

#ifdef __cplusplus
extern "C" {
#endif

# include "..\svm_light\svm_common.h"
# include "..\svm_light\svm_learn.h"

#ifdef __cplusplus
}
#endif

void SVM_Training(Musawwir_Obj_Detector &MOD, string TrainPosDirPath, string TrainNegDirPath, string SVM_Model_FilePath);
double svm_dist_orig(double *Data, MODEL *model, DOC **docs, mWORD *words, int feature_size);
double svm_dist_simple_1(double *Data, MODEL *model, int feature_size);
double Kernel_Dist(float *Data);
int svm_dist_i(int *Data);
float svm_dist_f(float *Data);
float svm_dist_f(double *Data);

int dat2words(int row, double *data, mWORD *words, int feature_size);
int computeOffset(int numRows, int numCols, int i, int j);
void support_vec_stats();
void save_SVM(MODEL *model, const char* filename);// , int win_R, int win_C, char grayscale);
void load_SVM(const char* filename, int &feature_size, int &sv_num, double**& supvec, double*& alpha, double &b, double*& ASV);// , int &win_R, int &win_C, char &grayscale);
void ConvToDOC(double** examples, DOC ***docs, long int totwords, long int totdoc);
void set_input_parameters(LEARN_PARM *learn_parm,KERNEL_PARM *kernel_parm);
void SVM_Train(const char* svm_file_name, double** examples, double* labels, long totwords, long totdoc);// , int win_R, int win_C, char grayscale);
float Kernel_LUT_Dist(float* ex);
void Build_Kernel_LUT(void);
void filter_LP_LUT(void);