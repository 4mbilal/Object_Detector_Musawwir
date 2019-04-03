#ifndef _DatasetTrain_H__
#define _DatasetTrain_H__

#include "Defines.h"
#include "SVM_interface.h"

void Delete_Files_In_A_Dir(string DirPath, int step);
void Create_Neg_Training_Images(string Neg_Full_Img_Src_Dir, string Neg_Train_Img_Dst_Dir, Size Img_Size, int examples_cnt);
void Create_Pos_Training_Images(string Pos_Full_Img_Src_Dir, string Pos_Train_Img_Dst_Dir, float Scale, Rect crop);
int Hard_Negative_Mining(Musawwir_Obj_Detector &MOD, float Init_Scale, float Scale_Stride, Size Spatial_Stride, float Score_Threshold, string Neg_Full_Img_Src_Dir, string Neg_Train_Img_Dst_Dir);
void Purge_Examples(Musawwir_Obj_Detector &MOD, string Neg_Train_Img_Dst_Dir, int count, bool type);
void SVM_Classification();
void SVM_Classification2();
float Pos_Window_Classification(Mat& img);
float Neg_Window_Classification(Mat& img, float** DET, int& cn);
void Training_Dir_Prep();
void Training_Master(Musawwir_Obj_Detector &MOD);
void LUT_CSV(void);
void Detect_Caltech_format(Musawwir_Obj_Detector &MOD);
void Caltech_res_files_generation(Mat& img, Musawwir_Obj_Detector &MOD, char* res_path, char* ann_path);
void read_annotations(vector<Rect>& Ann_Rects, char* ann_file_path, int min_height, float min_overlap);
void Grid_Search_SVM_Parameters(Musawwir_Obj_Detector &MOD);

#endif