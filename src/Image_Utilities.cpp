#include "Image_Utilities.h"
using namespace std;
using namespace cv;

void DrawRectangles(Mat& Frame, vector<Rect>& BB_Rects, vector<double>& BB_Scores, float threshold) {
	for (int x = 0; x < BB_Scores.size(); x++) {
		if (BB_Scores[x] < threshold) continue;
		Rect detected_bb;
		char str[50];
		sprintf(str, "%.02f", BB_Scores[x]);
		detected_bb = BB_Rects[x];
		rectangle(Frame, detected_bb, Scalar(255, 0, 0), 4);
	}
}

float FastInvSqrt(float x) {
	float xhalf = 0.5f * x;
	int i = *(int*)&x;
	i = 0x5f3759df - (i >> 1);
	x = *(float*)&i;
	x = x*(1.5f - (xhalf*x*x));
	return x;
}


void write_rgb(char* input_file_name, char* output_file_name){
	ofstream file;
	Mat image = imread(input_file_name);
//	FitSize(image,image);
	char hex_value[9]="";
	//imshow("out",image);
	//waitKey(1000);
	file.open (output_file_name,ios::binary);
	unsigned char *im = (unsigned char*)(image.data);
	int rows = image.rows;
	int cols = image.cols;
	unsigned char r,g,b;

	for(int j = 0;j < rows;j++){
        for(int i = 0;i < cols;i++){
			b = im[j*cols*3+i*3];
			g = im[j*cols*3+i*3+1];
			r = im[j*cols*3+i*3+2];
			sprintf(hex_value,"%02x\n%02x\n%02x\n",r,g,b);
			file.write(reinterpret_cast<char*>( hex_value ), sizeof hex_value);
			//file.write(reinterpret_cast<char*>( &r ), sizeof r);
			//file.write(reinterpret_cast<char*>( &g ), sizeof g);
			//file.write(reinterpret_cast<char*>( &b ), sizeof b);
		}
	}
	file.close();
}

void read_rgb(char* input_file_name, char* output_file_name){
	ifstream file;
	Mat image;
	image.create(240,320, CV_8UC3);
	file.open (input_file_name,ios::binary);
	unsigned char *im = (unsigned char*)(image.data);
	int rows = image.rows;
	int cols = image.cols;
	unsigned char r,g,b;

	for(int j = 0;j < rows;j++){
        for(int i = 0;i < cols;i++){
			file.read(reinterpret_cast<char*>( &r ), sizeof r);
			file.read(reinterpret_cast<char*>( &g ), sizeof g);
			file.read(reinterpret_cast<char*>( &b ), sizeof b);
			im[j*cols*3+i*3] = b;
			im[j*cols*3+i*3+1] = g;
			im[j*cols*3+i*3+2] = r;
		}
	}
	file.close();
	imwrite(output_file_name,image);
}


void plot(float* data){
	Mat fig;
	const int height = 100;
	const int bar_width = 2;
	const int x_limit = 1000;
	const int width = x_limit*bar_width;

	fig.create(height, width, CV_8UC3);
	unsigned char* fig_ptr = (unsigned char*)(fig.data);
	for(int i=0;i<width*height*3;i++)
		fig_ptr[i] = 255;


	for(int i=0;i<width;i=i+bar_width){
		for(int j=0;j<height;j++){
			if(data[int(i/bar_width)]*height>j){
				for(int k=0;k<bar_width;k++){
					fig_ptr[(height-j-1)*width*3 + i*3 + k*3] = 0;
					fig_ptr[(height-j-1)*width*3 + i*3 + k*3 + 1] = 0;
				}
			}
		}
	}
	do{
	imshow("Plot",fig);
	}while(waitKey(1)!='n');
	
}

void RGB2Gray(unsigned char* img, int row, int col)
{
	int i, j;
	unsigned char r, g, b;
	unsigned short gray;

	for (j = 0; j < row; j++){
		for (i = 0; i < col; i++){
			b = img[j*col * 3 + i * 3];
			g = img[j*col * 3 + i * 3 + 1];
			r = img[j*col * 3 + i * 3 + 2];
			gray = (r >> 2) + (r >> 4) + (g >> 1) + (g >> 4) + (b >> 3);
			img[j*col * 3 + i * 3] = gray;
			img[j*col * 3 + i * 3 + 1] = gray;
			img[j*col * 3 + i * 3 + 2] = gray;
		}
	}
}

void EdgeDetection(unsigned char* img_in, unsigned char* img_out, int row, int col){
	int i, j;
	short dx, dy;
	unsigned short mag;
	unsigned char edge;

	RGB2Gray(img_in, row, col);

	for (j = 1; j < row - 1; j++){			//leave 1-pixel wide boundary
		for (i = 1; i < col - 1; i++){
			dx = img_in[j*col * 3 + (i + 1) * 3] - img_in[j*col * 3 + (i - 1) * 3];
			dy = img_in[(j + 1)*col * 3 + i * 3] - img_in[(j - 1)*col * 3 + i * 3];
			mag = dx*dx + dy*dy;
			if (mag>2000)
				edge = 255;
			else
				edge = 0;
			//mag = mag>>8;
			img_out[j*col * 3 + i * 3] = edge;
			img_out[j*col * 3 + i * 3 + 1] = edge;
			img_out[j*col * 3 + i * 3 + 2] = edge;
		}
	}
}

void img_resize_fast(Mat& frame, double scale){
	resize(frame, frame, Size(0, 0), scale, scale);// , CV_INTER_AREA);
}