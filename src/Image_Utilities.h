#include "Defines.h"

void DrawRectangles(Mat& Frame, vector<Rect>& BB_Rects, vector<double>& BB_Scores, float threshold);
void write_rgb(char* input_file_name, char* output_file_name);
void read_rgb(char* input_file_name, char* output_file_name);
void plot(float* data);
void RGB2Gray(unsigned char* img, int row, int col);
void EdgeDetection(unsigned char* img_in, unsigned char* img_out, int row, int col);
void img_resize_fast(Mat& frame, double scale);