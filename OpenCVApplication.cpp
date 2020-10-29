// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <math.h>
#include <queue>
#include <time.h>
#include <vector>
#include <algorithm>
#include <opencv2/imgproc/imgproc.hpp>


void testOpenImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image",src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName)==0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName,"bmp");
	while(fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(),src);
		if (waitKey()==27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", IMREAD_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, WINDOW_AUTOSIZE);
	moveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, WINDOW_AUTOSIZE);
	moveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, COLOR_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	printf("Press any key to continue ...\n");
	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]
		
		Mat src = imread(fname,IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height,width,CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				uchar val = src.at<uchar>(i,j);
				uchar neg = 255 - val;
				dst.at<uchar>(i,j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void L1_changeGrayLevels()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				uchar other;
				if (val >= 205) {
					other = 255;
				}
				else {
					other = val + 50;
				}
				dst.at<uchar>(i, j) = other;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("different grayscale", dst);
		waitKey();
	}
}

void L1_fourSquareColors()
{
	int height = 700;
	int width = 500;
	Mat_<Vec3b> result(height,width);
	for (int i = 0; i < height; i++) {
		if (i < height/2) {
			for (int j = 0; j < width; j++) {
				if (j < width / 2) {
					result(i, j) = Vec3b(255, 255, 255);
				}
				else {
					result(i, j) = Vec3b(0, 0, 255);
				}
			}
		}
		else {
			for (int j = 0; j < width; j++) {
				if (j < width / 2) {
					result(i, j) = Vec3b(0, 255, 0);
				}
				else {
					result(i, j) = Vec3b(0, 255, 255);
				}
			}
		}
	}
	imshow("Four Colors", result);
	waitKey();
}

void L1_inverseMatrix() {
	float vals[9] = { 0,2,4,1,6,0,1,2,3};
	Mat M(3, 3, CV_32FC1, vals); //9 parameter constructor
	std::cout << M.inv() << std::endl;
	getchar();
	getchar();
}

void L2_splitChannels() {

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{ 
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, IMREAD_COLOR);
		int height = src.rows;
		int width = src.cols;
		Mat red = Mat(height, width, CV_8UC3);
		Mat green = Mat(height, width, CV_8UC3);
		Mat blue = Mat(height, width, CV_8UC3);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b p = src.at<Vec3b>(i, j);
				red.at<Vec3b>(i, j) = Vec3b(0, 0, p[2]);
				green.at<Vec3b>(i, j) = Vec3b(0, p[1], 0);
				blue.at<Vec3b>(i, j) = Vec3b(p[0], 0, 0);
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("red image", red);
		imshow("green image", green);
		imshow("blue image", blue);
		waitKey();
	}
}


void L2_colorToGrayscale() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, IMREAD_COLOR);
		int height = src.rows;
		int width = src.cols;
		Mat gray = Mat(height, width, CV_8UC1);
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b p = src.at<Vec3b>(i, j);
				uchar g = (p[0]*0.1 + p[1]*0.7 + p[2]*0.2) / 3;
				gray.at<uchar>(i, j) = g;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("grayscale image", gray);
		waitKey();
	}
}

void L2_grayToBw() {
	char fname[MAX_PATH];
	uchar th = 100;
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat bw = Mat(height, width, CV_8UC1);
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar p = src.at<uchar>(i, j);
				if (p > th) {
					bw.at<uchar>(i, j) = 255;
				}
				else {
					bw.at<uchar>(i, j) = 0;
				}
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("bw image", bw);
		waitKey();
	}
}

void L2_rgbToHsv() {
	char fname[MAX_PATH];
	float r, g, b;
	float V, m, C;
	float H, S;
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]
		Mat src = imread(fname, IMREAD_COLOR);
		int height = src.rows;
		int width = src.cols;
		Mat h = Mat(height, width, CV_8UC1);
		Mat s = Mat(height, width, CV_8UC1);
		Mat v = Mat(height, width, CV_8UC1);
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b p = src.at<Vec3b>(i, j);
				r = (float)p[2] / 255;
				g = (float)p[1] / 255;
				b = (float)p[0] / 255;
				V = max(r, max(g, b));
				m = min(r, min(g, b));
				C = V - m;
				if (V != 0) {
					S = C / V;
				}
				else {
					S = 0;
				}

				if (C != 0) {
					if(V == r) H = 60 * (g - b) / C; 
					if (V == g) H = 120 + 60 * (b - r) / C; 
					if (V == b) H = 240 + 60 * (r - g) / C;
				}
				else {
					H = 0;
				}

				if (H < 0) {
					H = H + 360;
				}
				//setting H,S,V
				float H_norm = H * 255 / 360;
				float S_norm = S * 255;
				float V_norm = V * 255;
				h.at<uchar>(i, j) = H_norm;
				s.at<uchar>(i, j) = S_norm;
				v.at<uchar>(i, j) = V_norm;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("hue image", h);
		imshow("saturation image", s);
		imshow("value image", v);
		waitKey();
	}
}

/* Histogram display function - display a histogram using bars (simlilar to L3 / PI)
Input:
name - destination (output) window name
hist - pointer to the vector containing the histogram values
hist_cols - no. of bins (elements) in the histogram = histogram image width
hist_height - height of the histogram image
Call example:
showHistogram ("MyHist", hist_dir, 255, 200);
*/
void showHistogram(const std::string& name, int* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i < hist_cols; i++)
		if (hist[i] > max_hist)
			max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}

void L3_histogram() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		long int M = height * width;
		Mat bw = Mat(height, width, CV_8UC1);
		int hist[256] = {};
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar pixel = src.at<uchar>(i, j);
				hist[pixel]++;
			}
		}

		imshow("input image", src);
		showHistogram ("Histogram", hist, width, height);
		waitKey();
	}
}

void L3_histogramAccumulators() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		long int M = height * width;
		int WH = 16;
		Mat bw = Mat(height, width, CV_8UC1);
		int hist[256] = {};
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar pixel = src.at<uchar>(i, j);
				hist[pixel]++;
			}
		}

		float p[256];
		for (int i = 0; i < 256; i++) {
			p[i] = (float)hist[i] / M;
		}

		int accHist[256] = {};
		for (int i = 0; i < 256; i++) {
			accHist[i] = hist[i] / (256 / WH);
		}

		imshow("input image", src);
		showHistogram("Histogram", accHist, WH, height);
		waitKey();
	}
}

void L3_multilevelThr() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		long int M = height * width;
		int WH = 5;
		int windowWidth = 2*WH + 1;
		float TH = 0.0003;
		int hist[256] = {};
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar pixel = src.at<uchar>(i, j);
				hist[pixel]++;
			}
		}

		float p[256];
		for (int i = 0; i < 256; i++) {
			p[i] = (float)hist[i] / M;
		}

		int multi[256] = {};
		for (int i = WH; i < 255 - WH; i++) {
			float v = 0.0;
			float max = 0.0;
			for (int k = i - WH; k <= i + WH; k++) {
				v += p[k];
				if (max <= p[k]) {
					max = p[k];
				}
			}
			v = v / (2 * WH + 1);
			if (p[i] > v + TH && p[i] >= max) {
				multi[i] = hist[i];
			}
		}
		multi[0] = hist[0];
		multi[255] = hist[255];

		imshow("input image", src);
		showHistogram("Histogram", multi, width, height);
		waitKey();
	}
}

void L3_disth() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		int M = height * width;
		int WH = 5;
		int windowWidth = 2 * WH + 1;
		float TH = 0.0003;
		int hist[256] = {};
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar pixel = src.at<uchar>(i, j);
				hist[pixel]++;
			}
		}

		float p[256];
		for (int i = 0; i < 256; i++) {
			p[i] = (float)hist[i] / M;
		}

		int multi[256] = {};
		for (int i = WH; i < 255 - WH; i++) {
			float v = 0.0f;
			float max = 0.0f;
			for (int k = i - WH; k <= i + WH; k++) {
				v += p[k];
				if (max <= p[k]) {
					max = p[k];
				}
			}
			v = v / (2 * WH + 1);
			if (p[i] > v + TH && p[i] >= max) {
				multi[i] = i;
			}
		}
		multi[0] = hist[0];
		multi[255] = hist[255];

		imshow("input image", src);
		showHistogram("Histogram", multi, width, height);
		waitKey();
	}
}

void L4_drawElongation(float phi, int gr,int gc, int cmin, int cmax, Vec3b color) {
	int ra = gr + tan(phi) * (cmin - gc);
	int rb = gr + tan(phi) * (cmax - gc);
	Point A(cmin, ra);
	Point B(cmax, rb);
	Mat elongation = Mat(400, 400, CV_8UC3);
	for (int i = 0; i < 400; i++) {
		for (int j = 0; j < 400; j++) {

		}
	}
}

void callBackDClick(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == EVENT_LBUTTONDBLCLK)
	{
		Vec3b pix = (Vec3b)(*src).at<Vec3b>(y, x);
		Vec3b back = (Vec3b)(*src).at<Vec3b>(0, 0);
		int area = 0;
		int r = 0; //current row
		int c = 0; //current column
		int X = 0; //X from the tan
		int Y = 0; //Y from the tan
		int P = 0; // perimeter
		float T = 0.0; // thinness ratio
		int cmin = (int)(*src).cols - 1; // to calculate minimum column value of the object
		int rmin = (int)(*src).rows - 1; //to calculate minimum row value of the object
		int cmax = 0;
		int rmax = 0;
		float R = 0.0; //aspect ratio
		Mat elong = Mat((int)(*src).rows, (int)(*src).cols, CV_8UC3);  //separate window for drawing the object and its elongation line
		Mat horizontal = Mat((int)(*src).rows, (int)(*src).cols, CV_8UC3); //window for horizontal projection
		Mat vertical = Mat((int)(*src).rows, (int)(*src).cols, CV_8UC3); //window for vertical projection
		int xh = 0;			//incrementer for the x value on the horizontal projection
		int yv = 0;
		for (int i = 0; i < (int)(*src).rows; i++) {
			for (int j = 0; j < (int)(*src).cols; j++) {
				if ((Vec3b)(*src).at<Vec3b>(i, j) == pix) {
					elong.at<Vec3b>(i, j) = (Vec3b)(*src).at<Vec3b>(i, j);
					horizontal.at<Vec3b>(i, xh) = pix;
					xh++;
					area++;
					r += i;
					c += j;
					if ((Vec3b)(*src).at<Vec3b>(i - 1, j - 1) == back ||
						(Vec3b)(*src).at<Vec3b>(i - 1, j) == back ||
						(Vec3b)(*src).at<Vec3b>(i - 1, j + 1) == back ||
						(Vec3b)(*src).at<Vec3b>(i, j - 1) == back ||
						(Vec3b)(*src).at<Vec3b>(i, j + 1) == back ||
						(Vec3b)(*src).at<Vec3b>(i + 1, j - 1) == back ||
						(Vec3b)(*src).at<Vec3b>(i + 1, j) == back ||
						(Vec3b)(*src).at<Vec3b>(i + 1, j + 1) == back) {
						P++;
					}
					if (i < rmin)
						rmin = i;
					if (j < cmin)
						cmin = j;
					if (i > rmax)
						rmax = i;
					if (j > cmax)
						cmax = j;
				}
				else {
					elong.at<Vec3b>(i, j) = back;
				}
			}
			xh = 0;
		}
		for (int j = 0; j < (int)(*src).cols; j++) {
			for (int i = 0; i < (int)(*src).rows; i++) {
				if ((Vec3b)(*src).at<Vec3b>(i, j) == pix) {
					vertical.at<Vec3b>(yv, j) = pix;
					yv++;
				}
			}
			yv = 0;
		}
		float gr =(float) r / area;
		float gc =(float) c / area;
		for (int i = 0; i < (int)(*src).rows; i++) {
			for (int j = 0; j < (int)(*src).cols; j++) {
				if ((Vec3b)(*src).at<Vec3b>(i, j) == pix) {
					Y += (i - gr)*(j-gc);
					X += (j - gc)*(j - gc) - (i - gr) * (i - gr);
				}
			}
		}
		Y = 2 * Y;
		float p = atan2(Y,X)/2;
		if (p < 0) {
			p += CV_PI;
		}
		int phi = p * (180 / CV_PI);
		P = P * (CV_PI / 4);
		T = 4 * CV_PI * area / (P * P);
		R = (float) (cmax - cmin + 1) / (rmax - rmin + 1);
		int ra = gr + tan(p) * (cmin - gc);
		int rb = gr + tan(p) * (cmax - gc);
		Point A(cmin, ra);
		Point B(cmax, rb);
		line(elong, A, B, Scalar(0, 0, 0), 2);
		imshow("Elongation", elong);
		imshow("Horizontal projection", horizontal);
		imshow("Vertical projection", vertical);
		printf("The area is: %d \n", area);
		printf("The center of mass has the coordinates: %lf, and %lf \n", gr,gc);
		printf("The angle of the elongation axis %d \n", phi);
		printf("The perimeter is %d\n", P);
		printf("The thinness ratio is %lf\n", T);
		printf("The aspect ratio is %lf\n", R);
	}
}

void L4_GeometricalFeatures() {
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", callBackDClick, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

void L5_BFS() 
{
	int label = 0;
	char fname[MAX_PATH];
	while (openFileDlg(fname)) 
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat labels = Mat::zeros(height, width, CV_32SC1);
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				if (src.at<uchar>(i, j) == 0 && labels.at<int>(i, j) == 0) {
					label++;
					std::queue<Point> Q;
					labels.at<int>(i, j) = label;
					Q.push(Point(j, i));
					while (!Q.empty()) {
						Point q = Q.front();
						Q.pop();
						int dy[8] = { -1,-1,-1,0,0,1,1,1 };
						int dx[8] = { -1,0,1,-1,1,-1,0,1 };
						for (int k = 0; k < 8; k++) {
							if (labels.at<int>(q.y + dy[k], q.x + dx[k]) == 0 && src.at<uchar>(q.y + dy[k], q.x + dx[k]) == 0) {
								labels.at<int>(q.y + dy[k], q.x + dx[k]) = label;
								Q.push(Point(q.x + dx[k], q.y + dy[k]));
							}
						}
					}
				}
			}
		}

		srand(time(NULL));
		Mat finalimg = Mat(height, width, CV_8UC3);
		for (int k = 1; k < label; k++) {
			Vec3b color = Vec3b(rand() % 256, rand() % 256, rand() % 256);
			for (int i = 0; i < height; i++) {
				for (int j = 0; j < width; j++) {
					if (labels.at<int>(i, j) == k) {
						finalimg.at<Vec3b>(i, j) = color;
					}
				}
			}
		}
		imshow("src", src);
		imshow("Colored Image", finalimg);
		waitKey(0);
	}
}

void L5_twoPass()
{
	int label = 0;
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat labels = Mat::zeros(height, width, CV_32SC1);
		std::vector<std::vector<int>> edges(1000);
		for (int i = 0; i < height-1; i++) {
			for (int j = 0; j < width-1; j++) {
				if (src.at<uchar>(i, j) == 0 && labels.at<int>(i, j) == 0) {
					std::vector<int> L;
					int dy[4] = { -1,-1,-1,0};
					int dx[4] = { -1,0,1,-1};
					for (int k = 0; k < 4; k++) {
						if (labels.at<int>(i + dy[k], j + dx[k]) > 0) {
							L.push_back(labels.at<int>(i + dy[k], j + dx[k]));
						}
					}
					if (L.size() == 0) {
						label++;
						edges.resize(label + 1);
						labels.at<int>(i, j) = label;
					}
					else {
						int x = 10000;
						for (int y = 0; y < L.size(); y++) {
							if (x >= L[y])
								x = L[y];
						}
						labels.at<int>(i, j) = x;
						for (int k = 0; k < L.size(); k++) {
							if (L.at(k) != x) {
								edges[x].push_back(L.at(k));
								edges[L.at(k)].push_back(x);
							}
						}
					}
				}
			}
		}

		int newlabel = 0;
		int* newlabels = new int[label + 1];
		for (int i = 0; i < (label + 1) ; i++) {
			newlabels[i] = 0;
		}
		for (int i = 1; i <= label; i++) {
			if (newlabels[i] == 0) {
				newlabel++;
				std::queue<int> Q;
				newlabels[i] = newlabel;
				Q.push(i);
				while (!Q.empty()) {
					int x = Q.front();
					Q.pop();
					for (int k = 0; k < edges[x].size(); k++) {
						if (newlabels[edges[x][k]] == 0) {
							newlabels[edges[x][k]] = newlabel;
							Q.push(edges[x][k]);
						}
					}
				}
			}
		}

		for (int i = 0; i < height - 1; i++) {
			for (int j = 0; j < width - 1; j++) {
				labels.at<int>(i, j) = newlabels[labels.at<int>(i, j)];
			}
		}

		srand(time(NULL));
		Mat finalimg = Mat(height, width, CV_8UC3);
		for (int k = 1; k <= newlabel; k++) {
			Vec3b color = Vec3b(rand() % 256, rand() % 256, rand() % 256);
			for (int i = 0; i < height; i++) {
				for (int j = 0; j < width; j++) {
					if (labels.at<int>(i, j) == k) {
						finalimg.at<Vec3b>(i, j) = color;
					}
				}
			}
		}
		imshow("src", src);
		imshow("Colored Image", finalimg);
		waitKey(0);
	}
}

void L6_contourAlg() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		int dir = 7;
		int n = 0;
		int ac[1000];
		int k = 0;
		Point p0, p1, pn1, pn;
		int dy[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };
		int dx[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };
		std::vector<Point> contour;
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				if (src.at<uchar>(i, j) == 0) {
					p0 = Point(j, i);
					contour.push_back(p0);
					do {
						
						if (dir % 2 == 0) {
							dir = (dir + 7) % 8;
						}
						else {
							dir = (dir + 6) % 8;
						}
						while (src.at<uchar>(contour.at(n).y + dy[dir], contour.at(n).x + dx[dir]) != 0) {
							dir = (dir + 1) % 8;
						}
						if (n == 0) {
							p1 = Point(contour.at(n).x + dx[dir], contour.at(n).y + dy[dir]);
							contour.push_back(p1);;
						}
						else {
							contour.push_back(Point(contour.at(n).x + dx[dir], contour.at(n).y + dy[dir]));
						}
						ac[k++] = dir;
						n++;
					} while (n < 2 || contour.at(n) != p1 || contour.at(n-1) != p0);
					goto label;
				}
				
			}
		}
		label:
		Mat drawContour = Mat(height, width, CV_8UC1);
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				drawContour.at<uchar>(i, j) = 255;
			}
		}

		for (int i = 0; i < contour.size()-2; i++) {
			drawContour.at<uchar>(contour.at(i).y, contour.at(i).x) = 0;
		}
		for (int i = 0; i < k; i++) {
			printf(" %d ", ac[i]);
		}
		printf("\n");
		int dc[1000];
		for (int i = 0; i < k - 1; i++) {
			dc[i] = (ac[i + 1] - ac[i] + 8) % 8;
			printf("%d ", dc[i]);
		}
		imshow("Original", src);
		imshow("Contour", drawContour);
		waitKey(0);
	}
}

void L6_drawContour() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname,IMREAD_GRAYSCALE);
		FILE *input = fopen("D:\\facultate\\IP\\lab1\\reconstruct.txt","r");
		Point startPoint = Point(0, 0);
		int pixelNo = 0;
		int dy[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };
		int dx[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };
		fscanf(input, "%d %d\n", &startPoint.y, &startPoint.x);
		fscanf(input, "%d\n", &pixelNo);

		for (int i = 0; i < pixelNo-1; i++) {
			int dir;
			fscanf(input, "%d", &dir);
			src.at<uchar>(startPoint.y, startPoint.x) = 0;
			startPoint.y = startPoint.y + dy[dir];
			startPoint.x = startPoint.x + dx[dir];
			fscanf(input, " ");
		}
		
		imshow("All Contour", src);
		waitKey(0);
	}
}

bool IsInside(Mat img, int i, int j) {
	if (i >= 0 && i < img.rows && j >= 0 && j < img.cols) return true;
	else return false;
}

void L7_dilation() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		Mat myStruct(3, 3, CV_8UC1);
		int middle = 3 / 2;
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
				myStruct.at<uchar>(i, j) = 255;
		for (int i = 0; i < 3; i++)
			myStruct.at<uchar>(i, middle) = 0;
		for (int i = 0; i < 3; i++)
			myStruct.at<uchar>(middle, i) = 0;
		Mat myStruct2;
		resize(myStruct, myStruct2, Size(3 * 100, 3 * 100), 0, 0, 0);

		Mat res(src.rows, src.cols, CV_8UC1);
		res = src.clone();
		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++)
				if (src.at<uchar>(i, j) < 128) {
					for (int i2 = 0; i2 < myStruct.rows; i2++) {
						for (int j2 = 0; j2 < myStruct.rows; j2++) {
							int ipixel = i + i2 - myStruct.rows / 2;
							int jpixel = j + j2 - myStruct.rows / 2;
							if (IsInside(src, ipixel, jpixel) && myStruct.at<uchar>(i2, j2) < 128)
								res.at<uchar>(ipixel, jpixel) = 0;
						}
					}
				}

		}
				
		imshow("input", src);
		imshow("structure", myStruct2);
		imshow("dilate", res);
		waitKey();
	}
}

void L7_erode() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);

		Mat myStruct(3, 3, CV_8UC1);
		int middle = 3 / 2;
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
				myStruct.at<uchar>(i, j) = 255;
		for (int i = 0; i < 3; i++)
			myStruct.at<uchar>(i, middle) = 0;
		for (int i = 0; i < 3; i++)
			myStruct.at<uchar>(middle, i) = 0;
		Mat myStruct2;

		resize(myStruct, myStruct2, Size(3 * 100, 3 * 100), 0, 0, 0);

		Mat res(src.rows, src.cols, CV_8UC1);
		res = src.clone();
		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				if (src.at<uchar>(i, j) < 128) {
					bool anyOutside = false;
					for (int i2 = 0; i2 < myStruct.rows; i2++) {
						for (int j2 = 0; j2 < myStruct.rows; j2++) {
							int ipixel = i + i2 - myStruct.rows / 2;
							int jpixel = j + j2 - myStruct.rows / 2;
							if (IsInside(src, ipixel, jpixel) && myStruct.at<uchar>(i2, j2) < 128 && src.at<uchar>(ipixel, jpixel) > 128)
								anyOutside = true;
						}
					}
						
					if (anyOutside)
						res.at<uchar>(i, j) = 255;
				}
			}
		}
			

		imshow("input", src);
		imshow("structure", myStruct2);
		imshow("erode", res);
		waitKey();
	}
}

void L7_open() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
			Mat src = imread(fname, IMREAD_GRAYSCALE);

			Mat myStruct(3, 3, CV_8UC1);
			int middle = 3 / 2;
			for (int i = 0; i < 3; i++)
				for (int j = 0; j < 3; j++)
					myStruct.at<uchar>(i, j) = 255;
			for (int i = 0; i < 3; i++)
				myStruct.at<uchar>(i, middle) = 0;
			for (int i = 0; i < 3; i++)
				myStruct.at<uchar>(middle, i) = 0;
			Mat myStruct2;

			resize(myStruct, myStruct2, Size(3 * 100, 3 * 100), 0, 0, 0);

			Mat res(src.rows, src.cols, CV_8UC1);
			res = src.clone();
			for (int i = 0; i < src.rows; i++) {
				for (int j = 0; j < src.cols; j++) {
					if (src.at<uchar>(i, j) < 128) {
						bool anyOutside = false;
						for (int i2 = 0; i2 < myStruct.rows; i2++) {
							for (int j2 = 0; j2 < myStruct.rows; j2++) {
								int ipixel = i + i2 - myStruct.rows / 2;
								int jpixel = j + j2 - myStruct.rows / 2;
								if (IsInside(src, ipixel, jpixel) && myStruct.at<uchar>(i2, j2) < 128 && src.at<uchar>(ipixel, jpixel) > 128)
									anyOutside = true;
							}
						}

						if (anyOutside)
							res.at<uchar>(i, j) = 255;
					}
				}
			}

			Mat finalRes(src.rows, src.cols, CV_8UC1);
			finalRes = res.clone();
			for (int i = 0; i < src.rows; i++) {
				for (int j = 0; j < src.cols; j++)
					if (res.at<uchar>(i, j) < 128) {
						for (int i2 = 0; i2 < myStruct.rows; i2++) {
							for (int j2 = 0; j2 < myStruct.rows; j2++) {
								int ipixel = i + i2 - myStruct.rows / 2;
								int jpixel = j + j2 - myStruct.rows / 2;
								if (IsInside(res, ipixel, jpixel) && myStruct.at<uchar>(i2, j2) < 128)
									finalRes.at<uchar>(ipixel, jpixel) = 0;
							}
						}
					}

			}

			imshow("input", src);
			imshow("structure", myStruct2);
			imshow("Open", finalRes);
			waitKey();
	}
}

void L7_close() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);


		Mat myStruct(3, 3, CV_8UC1);
		int middle = 3 / 2;
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
				myStruct.at<uchar>(i, j) = 255;
		for (int i = 0; i < 3; i++)
			myStruct.at<uchar>(i, middle) = 0;
		for (int i = 0; i < 3; i++)
			myStruct.at<uchar>(middle, i) = 0;
		Mat myStruct2;

		resize(myStruct, myStruct2, Size(3 * 100, 3 * 100), 0, 0, 0);


		Mat res(src.rows, src.cols, CV_8UC1);
		res = src.clone();
		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++)
				if (src.at<uchar>(i, j) < 128) {
					for (int i2 = 0; i2 < myStruct.rows; i2++) {
						for (int j2 = 0; j2 < myStruct.rows; j2++) {
							int ipixel = i + i2 - myStruct.rows / 2;
							int jpixel = j + j2 - myStruct.rows / 2;
							if (IsInside(src, ipixel, jpixel) && myStruct.at<uchar>(i2, j2) < 128)
								res.at<uchar>(ipixel, jpixel) = 0;
						}
					}
				}

		}

		Mat result(src.rows, src.cols, CV_8UC1);
		result = res.clone();
		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				if (res.at<uchar>(i, j) < 128) {
					bool anyOutside = false;
					for (int i2 = 0; i2 < myStruct.rows; i2++) {
						for (int j2 = 0; j2 < myStruct.rows; j2++) {
							int ipixel = i + i2 - myStruct.rows / 2;
							int jpixel = j + j2 - myStruct.rows / 2;
							if (IsInside(res, ipixel, jpixel) && myStruct.at<uchar>(i2, j2) < 128 && res.at<uchar>(ipixel, jpixel) > 128)
								anyOutside = true;
						}
					}

					if (anyOutside)
						result.at<uchar>(i, j) = 255;
				}
			}
		}

		imshow("input", src);
		imshow("structure", myStruct2);
		imshow("Close", result);
		waitKey();
	}
}

void L7_extractBoundary() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);


		Mat myStruct(3, 3, CV_8UC1);
		int middle = 3 / 2;
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
				myStruct.at<uchar>(i, j) = 0;
		Mat myStruct2;

		resize(myStruct, myStruct2, Size(3 * 100, 3 * 100), 0, 0, 0);

		Mat erodedImg(src.rows, src.cols, CV_8UC1);
		erodedImg = src.clone();
		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				if (src.at<uchar>(i, j) < 128) {
					bool anyOutside = false;
					for (int i2 = 0; i2 < myStruct.rows; i2++) {
						for (int j2 = 0; j2 < myStruct.rows; j2++) {
							int ipixel = i + i2 - myStruct.rows / 2;
							int jpixel = j + j2 - myStruct.rows / 2;
							if (IsInside(src, ipixel, jpixel) && myStruct.at<uchar>(i2, j2) < 128 && src.at<uchar>(ipixel, jpixel) > 128)
								anyOutside = true;
						}
					}

					if (anyOutside)
						erodedImg.at<uchar>(i, j) = 255;
				}
			}
		}

		Mat res(src.rows, src.cols, CV_8UC1);
		for (int i = 0; i < src.rows; i++)
			for (int j = 0; j < src.cols; j++)
				if (src.at<uchar>(i, j) < 128) {
					if (erodedImg.at<uchar>(i, j) < 128)
						res.at<uchar>(i, j) = 255;
					else
						res.at<uchar>(i, j) = 0;
				}
				else {
					if (erodedImg.at<uchar>(i, j) > 128)
						res.at<uchar>(i, j) = 255;
					else
						res.at<uchar>(i, j) = 255;
				}

		imshow("input", src);
		imshow("structure", myStruct2);
		imshow("Extracted Boundary", res);
		waitKey();
	}
}

void L8_meanAndDeviation() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int width = src.cols;
		int height = src.rows;
		int M = width * height;
		int sum = 0;
		int sqsum = 0;
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				sum += src.at<uchar>(i, j);
			}
		}
		double meanValue = sum / M;

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				sqsum += (src.at<uchar>(i, j) - meanValue)*(src.at<uchar>(i, j) - meanValue);
			}
		}

		double deviation = sqrt(sqsum/M);
		printf("The mean value for the selected image is: %lf\n", meanValue);
		printf("The standard deviation for the selected image is: %lf\n", deviation);
	}
}

void L8_thresholdAlg() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int width = src.cols;
		int height = src.rows;
		int imin = 255, imax = 0;
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				if (src.at<uchar>(i, j) < imin) {
					imin = src.at<uchar>(i, j);
				}
				if (src.at<uchar>(i, j) > imax) {
					imax = src.at<uchar>(i, j);
				}
			}
		}
		double t0 = (imax + imin)/2;
		double t1 = 255;
		double epsilon = 0.1;
		//computing histogram
		int hist[256] = {};
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar pixel = src.at<uchar>(i, j);
				hist[pixel]++;
			}
		}
		do {
			double mg1 = 0;
			double mg2 = 0;
			int sum1 = 0, sum2 = 0;
			int n1 = 0, n2 = 0;
			if (t1 != 255) {
				t0 = t1;
			}
			for (int i = imin; i <= t0; i++) {
				sum1 += i*hist[i];
				n1 += hist[i];
			}
			mg1 = sum1 / n1;

			for (int i = t0 + 1 ; i < imax; i++) {
				sum2 += i * hist[i];
				n2 += hist[i];
			}
			mg2 = sum2 / n2;

			t1 = (mg1 + mg2) / 2;
		} while (abs(t1 - t0) >= epsilon);
		printf("The threshold for the given image is: %lf\n", t1);

		Mat newImage = Mat(height, width, CV_8UC1);
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				if (src.at<uchar>(i, j) < t1) {
					newImage.at<uchar>(i, j) = 0;
				}
				if (src.at<uchar>(i, j) > t1) {
					newImage.at<uchar>(i, j) = 255;
				}
			}
		}
		imshow("Threshold image", newImage);
		waitKey(0);
	}
}

void L8_changeBrightness() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int width = src.cols;
		int height = src.rows;
		Mat dst = Mat(height, width, CV_8UC1);
		int offset;
		printf("Please enter offset value: ");
		scanf("%d", &offset);
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				if (src.at<uchar>(i, j) + offset <= 255) {
					dst.at<uchar>(i, j) = src.at<uchar>(i, j) + offset;
				}
				else {
					dst.at<uchar>(i, j) = 255;
				}
			}
		}
		imshow("Modified Brightness", dst);
		waitKey(0);
	}
}

void L8_contrastChange() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int width = src.cols;
		int height = src.rows;
		Mat dst = Mat(height, width, CV_8UC1);
		int imin = 255, imax = 0;
		int min,max;
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				if (src.at<uchar>(i, j) < imin) {
					imin = src.at<uchar>(i, j);
				}
				if (src.at<uchar>(i, j) > imax) {
					imax = src.at<uchar>(i, j);
				}
			}
		}
		printf("Please enter new min and new max: ");
		scanf("%d %d", &min,&max);
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				dst.at<uchar>(i, j) = min + (src.at<uchar>(i, j) - imin) * ((max - min) / (imax - imin));
			}
		}
		imshow("Modified Contrast", dst);
		waitKey(0);
	}
}

void L8_gammaCorrection() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int width = src.cols;
		int height = src.rows;
		Mat dst = Mat(height, width, CV_8UC1);
		double gamma;
		printf("Please enter gamma value: ");
		scanf("%lf", &gamma);
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				double gin = (double)src.at<uchar>(i, j) / 255;
				uchar gout = (uchar)(255 * pow(gin, gamma));
				if (gout <= 0) {
					dst.at<uchar>(i, j) = 0;
				}
				else if (gout >= 255) {
					dst.at<uchar>(i, j) = 255;
				}
				else {
					dst.at<uchar>(i, j) = gout;
				}
				
			}
		}
		imshow("Gamma Correction", dst);
		waitKey(0);
	}
}


void L8_histogramEqualization() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int width = src.cols;
		int height = src.rows;
		int M = height * width;
		int hist[256] = {};
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar pixel = src.at<uchar>(i, j);
				hist[pixel]++;
			}
		}

		float p[256]; //normalized histogram
		for (int i = 0; i < 256; i++) {
			p[i] = (float)hist[i] / M;
		}
		float pc[256] = {};
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				for (int k = 0; k < src.at<uchar>(i,j); k++) {
					pc[src.at<uchar>(i, j)] += p[k];
				}
			}
		}

		int gout[256];
		/*Mat dst = Mat(height, width, CV_8UC1);
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				dst.at<uchar>(i, j) = 255 * pc[src.at<uchar>(i, j)];
			}
		}*/
		for (int i = 0; i < 256; i++) {
			gout[i] = 255 * pc[i];
		}
		showHistogram("Equalized Histogram", gout, width, height);
		waitKey(0);
	}
}

void L9_meanFilter() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int dim;
		printf("\n Please input the dimension for the kernel: ");
		scanf("%d", &dim);
		int width = src.cols;
		int height = src.rows;
		Mat dst = Mat(height, width, CV_8UC1);
		for (int i = dim; i < height - dim; i++) {
			for (int j = dim; j < width - dim; j++) {
				int sum = 0;
				for (int u = -(dim/2); u < (dim/2 + 1); u++) {
					for (int v = -(dim / 2); v < (dim / 2 + 1); v++) {
						sum += src.at<uchar>(i + u, j + v);
					}
				}
				sum = sum / (dim*dim);
				if (sum >= 255) {
					dst.at<uchar>(i, j) = 255;
				}
				else {
					dst.at<uchar>(i, j) = (uchar) sum;
				}
			}
		}
		imshow("Original Image", src);
		imshow("Mean Filter", dst);
		waitKey(0);
	}
}

void L9_gaussianFilter() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int H[3][3] = {
			{1,2,1},
			{2,4,2},
			{1,2,1}
		};
		int width = src.cols;
		int height = src.rows;
		Mat dst = Mat(height, width, CV_8UC1);
		for (int i = 3; i < height - 3; i++) {
			for (int j = 3; j < width - 3; j++) {
				int sum = 0;
				for (int u = -1; u < 2; u++) {
					for (int v = -1; v < 2; v++) {
						sum += src.at<uchar>(i + u, j + v)*H[u+1][v+1];
					}
				}
				sum = sum / 16;
				if (sum >= 255) {
					dst.at<uchar>(i, j) = 255;
				}
				else {
					dst.at<uchar>(i, j) = (uchar) sum;
				}
			}
		}
		imshow("Original Image", src);
		imshow("Gaussian Filter", dst);
		waitKey(0);
	}
}

void L9_laplaceFilter() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int H[3][3] = {
			{-1,-1,-1},
			{-1,8,-1},
			{-1,-1,-1}
		};
		int width = src.cols;
		int height = src.rows;
		Mat dst = Mat(height, width, CV_8UC1);
		for (int i = 3; i < height - 3; i++) {
			for (int j = 3; j < width - 3; j++) {
				int sum = 0;
				for (int u = -1; u < 2; u++) {
					for (int v = -1; v < 2; v++) {
						sum += src.at<uchar>(i + u, j + v) * H[u + 1][v + 1];
					}
				}
				
				if (sum >= 255) {
					dst.at<uchar>(i, j) = 255;
				}
				else if (sum <= 0) {
					dst.at<uchar>(i, j) = 0;
				} 
				else{
					dst.at<uchar>(i, j) = (uchar)sum;
				}
			}
		}
		imshow("Original Image", src);
		imshow("Laplace Filter", dst);
		waitKey(0);
	}
}

void L9_highPassFilter() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int H[3][3] = {
			{-1,-1,-1},
			{-1,9,-1},
			{-1,-1,-1}
		};
		int width = src.cols;
		int height = src.rows;
		Mat dst = Mat(height, width, CV_8UC1);
		for (int i = 3; i < height - 3; i++) {
			for (int j = 3; j < width - 3; j++) {
				int sum = 0;
				for (int u = -1; u < 2; u++) {
					for (int v = -1; v < 2; v++) {
						sum += src.at<uchar>(i + u, j + v) * H[u + 1][v + 1];
					}
				}

				if (sum >= 255) {
					dst.at<uchar>(i, j) = 255;
				}
				else if (sum <= 0) {
					dst.at<uchar>(i, j) = 0;
				}
				else {
					dst.at<uchar>(i, j) = (uchar)sum;
				}
			}
		}
		imshow("Original Image", src);
		imshow("High Pass Filter", dst);
		waitKey(0);
	}
}

void centering_transform(Mat img) {
	//expectsfloatingpointimage
	for(int i=0; i<img.rows;i++){
		for(int j=0;j<img.cols;j++){
			img.at<float>(i,j)=((i+j)&1)? -(img.at<float>(i,j)) : img.at<float>(i,j);
		}
	}
}

void L9_idealCutLPF() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int width = src.cols;
		int height = src.rows;

		Mat srcf;
		src.convertTo(srcf, CV_32FC1);
		
		centering_transform(srcf);

		Mat fourier;
		dft(srcf, fourier, DFT_COMPLEX_OUTPUT);

		Mat channels[] = { Mat::zeros(src.size(),CV_32F),Mat::zeros(src.size(),CV_32F) };
		split(fourier, channels);

		Mat mag;
		magnitude(channels[0], channels[1], mag);
		float maxlog = 0;
		for (int i = 0; i < mag.rows; i++) {
			for (int j = 0; j < mag.cols; j++) {
				float x = mag.at<float>(i, j);
				float newVal = log(x + 1);
				if (newVal > maxlog)
					maxlog = newVal;
			}
		}
		Mat finalMag;

		for (int i = 0; i < mag.rows; i++) {
			for (int j = 0; j < mag.cols; j++) {
				float x = mag.at<float>(i, j);
				float newVal = (log(x + 1) / maxlog )*255;
				mag.at<float>(i, j) = newVal;
			}
		}
		normalize(mag, finalMag, 0, 255, NORM_MINMAX, CV_8UC1);

		printf("\nInput a value for R:");
		int R;
		scanf("%d", &R);

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				int x = (height / 2 - i) * (height / 2 - i) + (width / 2 - j) * (width / 2 - j);
				if (x >= R * R) {
					channels[0].at<float>(i, j) = 0;
					channels[1].at<float>(i, j) = 0;
				}
			}
		}

		Mat dst,dstf;
		merge(channels, 2, fourier);
		dft(fourier, dstf, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE);

		centering_transform(dstf);

		normalize(dstf, dst, 0, 255, NORM_MINMAX, CV_8UC1);

		imshow("Original Image", src);
		imshow("Magnitude", finalMag);
		imshow("Ideal Cut LPF", dst);
		waitKey(0);
	}
}

void L9_idealCutHPF() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int width = src.cols;
		int height = src.rows;

		Mat srcf;
		src.convertTo(srcf, CV_32FC1);

		centering_transform(srcf);

		Mat fourier;
		dft(srcf, fourier, DFT_COMPLEX_OUTPUT);

		Mat channels[] = { Mat::zeros(src.size(),CV_32F),Mat::zeros(src.size(),CV_32F) };
		split(fourier, channels);

		Mat mag, phi;
		magnitude(channels[0], channels[1], mag);
		float maxlog = 0;
		for (int i = 0; i < mag.rows; i++) {
			for (int j = 0; j < mag.cols; j++) {
				float x = mag.at<float>(i, j);
				float newVal = log(x + 1);
				if (newVal > maxlog)
					maxlog = newVal;
			}
		}
		Mat finalMag;

		for (int i = 0; i < mag.rows; i++) {
			for (int j = 0; j < mag.cols; j++) {
				float x = mag.at<float>(i, j);
				float newVal = (log(x + 1) / maxlog) * 255;
				mag.at<float>(i, j) = newVal;
			}
		}
		normalize(mag, finalMag, 0, 255, NORM_MINMAX, CV_8UC1);


		printf("\nInput a value for R:");
		int R;
		scanf("%d", &R);

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				int x = (height / 2 - i) * (height / 2 - i) + (width / 2 - j) * (width / 2 - j);
				if (x < R * R) {
					channels[0].at<float>(i, j) = 0;
					channels[1].at<float>(i, j) = 0;
				}
			}
		}

		Mat dst, dstf;
		merge(channels, 2, fourier);
		dft(fourier, dstf, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE);

		centering_transform(dstf);

		normalize(dstf, dst, 0, 255, NORM_MINMAX, CV_8UC1);

		imshow("Original Image", src);
		imshow("Magnitude", finalMag);
		imshow("Ideal Cut HPF", dst);
		waitKey(0);
	}
}

void L9_gaussianCutLPF() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int width = src.cols;
		int height = src.rows;

		Mat srcf;
		src.convertTo(srcf, CV_32FC1);

		centering_transform(srcf);

		Mat fourier;
		dft(srcf, fourier, DFT_COMPLEX_OUTPUT);

		Mat channels[] = { Mat::zeros(src.size(),CV_32F),Mat::zeros(src.size(),CV_32F) };
		split(fourier, channels);

		Mat mag;
		magnitude(channels[0], channels[1], mag);
		float maxlog = 0;
		for (int i = 0; i < mag.rows; i++) {
			for (int j = 0; j < mag.cols; j++) {
				float x = mag.at<float>(i, j);
				float newVal = log(x + 1);
				if (newVal > maxlog)
					maxlog = newVal;
			}
		}
		Mat finalMag;

		for (int i = 0; i < mag.rows; i++) {
			for (int j = 0; j < mag.cols; j++) {
				float x = mag.at<float>(i, j);
				float newVal = (log(x + 1) / maxlog) * 255;
				mag.at<float>(i, j) = newVal;
			}
		}
		normalize(mag, finalMag, 0, 255, NORM_MINMAX, CV_8UC1);


		printf("\nInput a value for A:");
		int A;
		scanf("%d", &A);

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				int x = exp(- (((height / 2 - i) * (height / 2 - i) + (width / 2 - j) * (width / 2 - j))/(A*A)));
				channels[0].at<float>(i, j) *= x;
				channels[1].at<float>(i, j) *= x;
			}
		}

		Mat dst, dstf;
		merge(channels, 2, fourier);
		dft(fourier, dstf, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE);

		centering_transform(dstf);

		normalize(dstf, dst, 0, 255, NORM_MINMAX, CV_8UC1);

		imshow("Original Image", src);
		imshow("Magnitude", finalMag);
		imshow("Gaussian LPF", dst);
		waitKey(0);
	}
}

void L9_gaussianCutHPF() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int width = src.cols;
		int height = src.rows;

		Mat srcf;
		src.convertTo(srcf, CV_32FC1);

		centering_transform(srcf);

		Mat fourier;
		dft(srcf, fourier, DFT_COMPLEX_OUTPUT);

		Mat channels[] = { Mat::zeros(src.size(),CV_32F),Mat::zeros(src.size(),CV_32F) };
		split(fourier, channels);

		Mat mag;
		magnitude(channels[0], channels[1], mag);
		float maxlog = 0;
		for (int i = 0; i < mag.rows; i++) {
			for (int j = 0; j < mag.cols; j++) {
				float x = mag.at<float>(i, j);
				float newVal = log(x + 1);
				if (newVal > maxlog)
					maxlog = newVal;
			}
		}
		Mat finalMag;

		for (int i = 0; i < mag.rows; i++) {
			for (int j = 0; j < mag.cols; j++) {
				float x = mag.at<float>(i, j);
				float newVal = (log(x + 1) / maxlog) * 255;
				mag.at<float>(i, j) = newVal;
			}
		}
		normalize(mag, finalMag, 0, 255, NORM_MINMAX, CV_8UC1);


		printf("\nInput a value for A:");
		int A;
		scanf("%d", &A);

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				int x = exp(-(((height / 2 - i) * (height / 2 - i) + (width / 2 - j) * (width / 2 - j)) / (A * A)));
				channels[0].at<float>(i, j) *= (1-x);
				channels[1].at<float>(i, j) *= (1-x);
			}
		}

		Mat dst, dstf;
		merge(channels, 2, fourier);
		dft(fourier, dstf, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE);

		centering_transform(dstf);

		normalize(dstf, dst, 0, 255, NORM_MINMAX, CV_8UC1);

		imshow("Original Image", src);
		imshow("Magnitude", finalMag);
		imshow("Gaussian HPF", dst);
		waitKey(0);
	}
}

void L10_medianFilter() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int dim;
		printf("\n Please input the dimension for the kernel: ");
		scanf("%d", &dim);
		int width = src.cols;
		int height = src.rows;
		Mat dst = Mat(height, width, CV_8UC1);
		for (int i = dim; i < height - dim; i++) {
			for (int j = dim; j < width - dim; j++) {
				std::vector<int> a;
				for (int u = -(dim / 2); u < (dim / 2 + 1); u++) {
					for (int v = -(dim / 2); v < (dim / 2 + 1); v++) {
						a.push_back(src.at<uchar>(i + u, j + v));

					}
				}
				std::sort(a.begin(), a.end());
				if (a.front() >= 255) {
					dst.at<uchar>(i, j) = 255;
				}
				else {
					dst.at<uchar>(i, j) = (uchar) a.at(a.size()/2);
					
				}
			}
		}
		imshow("Original Image", src);
		imshow("Median Filter", dst);
		waitKey(0);
	}
}

void L10_gaussian1x2D() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int dim;
		printf("\n Please input the dimension for the kernel: ");
		scanf("%d", &dim);
		int width = src.cols;
		int height = src.rows;

		float sigma = dim / 6.0f;
		Mat f(dim, dim, CV_32FC1);

		for (int i = 0; i < dim; ++i) {
			for (int j = 0; j < dim; ++j) {
				f.at<float>(i, j) = std::exp(-((i - dim / 2) * (i - dim / 2) + (j - dim / 2) * (j - dim / 2)) /(2 * sigma * sigma)) / (2 * PI * sigma * sigma);
			}
		}

		Mat dst = Mat(height, width, CV_8UC1);
		for (int i = dim; i < height - dim; i++) {
			for (int j = dim; j < width - dim; j++) {
				std::vector<int> a;
				float sum = 0.0f;
				for (int u = -(dim / 2); u < (dim / 2 + 1); u++) {
					for (int v = -(dim / 2); v < (dim / 2 + 1); v++) {
						sum += src.at<uchar>(i + u, j + v) * f.at<float>(dim / 2 + u, dim / 2 + v);
					}
				}

				dst.at<uchar>(i, j) = (uchar) sum;
				
			}
		}
		imshow("Original Image", src);
		imshow("Gaussian 1", dst);
		waitKey(0);
	}
}

void L10_gaussian2x1D()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int dim;
		printf("\n Please input the dimension for the kernel: ");
		scanf("%d", &dim);
		int width = src.cols;
		int height = src.rows;

		float sigma = dim / 6.0f;

		float* gx = (float*)malloc(sizeof(float) * dim);
		float* gy = (float*)malloc(sizeof(float) * dim);

		for (int i = 0; i < dim; ++i) {
			for (int j = 0; j < dim; ++j) {
				gx[i] = std::exp(-((i - dim / 2) * (i - dim / 2)) / (2 * sigma * sigma)) / (std::sqrt(2 * PI) * sigma);
				gy[i] = std::exp(-((i - dim / 2) * (i - dim / 2)) / (2 * sigma * sigma)) / (std::sqrt(2 * PI) * sigma);
			}
		}

		Mat dst = Mat(height, width, CV_8UC1);
		for (int i = dim; i < height - dim; i++) {
			for (int j = dim; j < width - dim; j++) {
				std::vector<int> a;
				float sum = 0.0f;
				for (int u = -(dim / 2); u < (dim / 2 + 1); u++) {
					sum += (int)src.at<uchar>(i + u, j) * gy[dim / 2 + u];
				}

				dst.at<uchar>(i, j) = (uchar)sum;

				sum = 0.0f;
				for (int v = -(dim / 2); v < (dim / 2 + 1); v++) {
					sum += (int)src.at<uchar>(i + v, j) * gx[dim / 2 + v];
				}

				dst.at<uchar>(i, j) = (uchar)sum;

			}
		}
		imshow("Original Image", src);
		imshow("Gaussian 2", dst);
		waitKey(0);
	}
}

void L11_gaussianFiltering()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int H[3][3] = {
			{1,2,1},
			{2,4,2},
			{1,2,1}
		};
		int width = src.cols;
		int height = src.rows;
		Mat dst = Mat(height, width, CV_8UC1);
		for (int i = 3; i < height - 3; i++) {
			for (int j = 3; j < width - 3; j++) {
				int sum = 0;
				for (int u = -1; u < 2; u++) {
					for (int v = -1; v < 2; v++) {
						sum += src.at<uchar>(i + u, j + v) * H[u + 1][v + 1];
					}
				}
				sum = sum / 16;
				if (sum >= 255) {
					dst.at<uchar>(i, j) = 255;
				}
				else {
					dst.at<uchar>(i, j) = (uchar)sum;
				}
			}
		}
		imshow("Original Image", src);
		imshow("Gaussian Filter", dst);
		waitKey(0);
	}
}

void L11_gradientMagnitude()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int Sx[3][3] = {
			{-1,0,1},
			{-2,0,2},
			{-1,0,1}
		};
		int Sy[3][3] = {
			{1,2,1},
			{0,0,0},
			{-1,-2,-1}
		};
		int width = src.cols;
		int height = src.rows;
		Mat dst = Mat(height, width, CV_8UC1);
		Mat G = Mat(height, width, CV_32FC1);
		Mat phi = Mat(height, width, CV_32FC1);
		for (int i = 3; i < height - 3; i++) {
			for (int j = 3; j < width - 3; j++) {
				float gx = 0;
				float gy = 0;
				for (int u = -1; u < 2; u++) {
					for (int v = -1; v < 2; v++) {
						gx += src.at<uchar>(i + u, j + v) * Sx[u + 1][v + 1];
						gy += src.at<uchar>(i + u, j + v) * Sy[u + 1][v + 1];
					}
				}

				G.at<float>(i, j) = sqrt(gx * gx + gy * gy);
				phi.at<float>(i, j) = atan2(gy, gx);

				dst.at<uchar>(i, j) = (uchar)G.at<float>(i, j) / (4 * sqrt(2));
			}
		}

		imshow("Original Image", src);
		imshow("G normalized", dst);
		waitKey(0);
	}
}

void L11_nonMaximaSupression()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int Sx[3][3] = {
			{-1,0,1},
			{-2,0,2},
			{-1,0,1}
		};
		int Sy[3][3] = {
			{1,2,1},
			{0,0,0},
			{-1,-2,-1}
		};
		int width = src.cols;
		int height = src.rows;
		Mat dst = Mat(height, width, CV_8UC1);
		Mat G = Mat(height, width, CV_32FC1);
		Mat phi = Mat(height, width, CV_32FC1);
		for (int i = 3; i < height - 3; i++) {
			for (int j = 3; j < width - 3; j++) {
				float gx = 0;
				float gy = 0;
				for (int u = -1; u < 2; u++) {
					for (int v = -1; v < 2; v++) {
						gx += src.at<uchar>(i + u, j + v) * Sx[u + 1][v + 1];
						gy += src.at<uchar>(i + u, j + v) * Sy[u + 1][v + 1];
					}
				}

				G.at<float>(i, j) = sqrt(gx * gx + gy * gy) / (4 * sqrt(2));
				phi.at<float>(i, j) = atan2(gy, gx);

			}
		}

		for (int i = 1; i < height -1; i++) {
			for(int j = 1 ; j < width -1; j++) {
				if ( ( phi.at<float>(i, j) > (-CV_PI / 8) && phi.at<float>(i, j) < CV_PI / 8 ) || (phi.at<float>(i, j) < (-7 * CV_PI / 8) || phi.at<float>(i, j) > (7 * CV_PI / 8)) ) {
					if (G.at<float>(i, j) >= G.at<float>(i, j + 1) && G.at<float>(i, j) >= G.at<float>(i, j - 1)) {
						dst.at<uchar>(i, j) = (uchar) G.at<float>(i, j) ;
					}
					else {
						dst.at<uchar>(i, j) = 0;
					}
				}
				else if( (phi.at<float>(i, j) > (CV_PI / 8) && phi.at<float>(i, j) < (3 * CV_PI / 8)) || ((phi.at<float>(i, j) > (-7 * CV_PI / 8)) && (phi.at<float>(i, j) < (-5 * CV_PI / 8)))) {
					if (G.at<float>(i, j) >= G.at<float>(i+1, j - 1) && G.at<float>(i, j) >= G.at<float>(i -1, j + 1)) {
						dst.at<uchar>(i, j) = (uchar)G.at<float>(i, j) ;
					}
					else {
						dst.at<uchar>(i, j) = 0;
					}
				}
				else if ( (phi.at<float>(i, j) > (3 * CV_PI / 8) && phi.at<float>(i, j) < (5 * CV_PI / 8)) || ((phi.at<float>(i, j) > (-5 * CV_PI / 8)) && (phi.at<float>(i, j) < (-3 * CV_PI / 8)))) {
					if (G.at<float>(i, j) >= G.at<float>(i + 1, j) && G.at<float>(i, j) >= G.at<float>(i -1, j)) {
						dst.at<uchar>(i, j) = (uchar)G.at<float>(i, j) ;
					}
					else {
						dst.at<uchar>(i, j) = 0;
					}
				}
				else if ( (phi.at<float>(i, j) > (5 * CV_PI / 8) && phi.at<float>(i, j) < (7 * CV_PI / 8)) || ((phi.at<float>(i, j) > (-3 * CV_PI / 8)) && (phi.at<float>(i, j) < (CV_PI / 8)))) {
					if (G.at<float>(i, j) >= G.at<float>(i + 1 , j + 1) && G.at<float>(i, j) >= G.at<float>(i - 1, j - 1)) {
						dst.at<uchar>(i, j) = (uchar) G.at<float>(i, j);
					}
					else {
						dst.at<uchar>(i, j) = 0;
					}
				}
			}
		}

		imshow("Original Image", src);
		imshow("Output Image", dst);
		waitKey(0);
	}
}

void L12_adaptiveHysteresisThresholding() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		Mat maxima = src.clone();
		int Sx[3][3] = {
			{-1,0,1},
			{-2,0,2},
			{-1,0,1}
		};
		int Sy[3][3] = {
			{1,2,1},
			{0,0,0},
			{-1,-2,-1}
		};
		int width = src.cols;
		int height = src.rows;
		Mat dst = Mat(height, width, CV_32FC1);
		Mat G = Mat(height, width, CV_32FC1);
		Mat phi = Mat(height, width, CV_32FC1);
		for (int i = 1; i < height - 1; i++) {
			for (int j = 1; j < width - 1; j++) {
				float gx = 0;
				float gy = 0;
				for (int u = -1; u < 2; u++) {
					for (int v = -1; v < 2; v++) {
						gx += src.at<uchar>(i + u, j + v) * Sx[u + 1][v + 1];
						gy += src.at<uchar>(i + u, j + v) * Sy[u + 1][v + 1];
					}
				}

				G.at<float>(i, j) = sqrt(gx * gx + gy * gy) / (4 * sqrt(2));
				phi.at<float>(i, j) = atan2(gy, gx);

			}
		}

		for (int i = 1; i < height - 1; i++) {
			for (int j = 1; j < width - 1; j++) {
				if ((phi.at<float>(i, j) > (-CV_PI / 8) && phi.at<float>(i, j) < CV_PI / 8) || (phi.at<float>(i, j) < (-7 * CV_PI / 8) || phi.at<float>(i, j) > (7 * CV_PI / 8))) {
					if (G.at<float>(i, j) >= G.at<float>(i, j + 1) && G.at<float>(i, j) >= G.at<float>(i, j - 1)) {
						dst.at<float>(i, j) = G.at<float>(i, j);
					}
					else {
						dst.at<float>(i, j) = 0;
					}
				}
				else if ((phi.at<float>(i, j) > (CV_PI / 8) && phi.at<float>(i, j) < (3 * CV_PI / 8)) || ((phi.at<float>(i, j) > (-7 * CV_PI / 8)) && (phi.at<float>(i, j) < (-5 * CV_PI / 8)))) {
					if (G.at<float>(i, j) >= G.at<float>(i + 1, j - 1) && G.at<float>(i, j) >= G.at<float>(i - 1, j + 1)) {
						dst.at<float>(i, j) = G.at<float>(i, j);
					}
					else {
						dst.at<float>(i, j) = 0;
					}
				}
				else if ((phi.at<float>(i, j) > (3 * CV_PI / 8) && phi.at<float>(i, j) < (5 * CV_PI / 8)) || ((phi.at<float>(i, j) > (-5 * CV_PI / 8)) && (phi.at<float>(i, j) < (-3 * CV_PI / 8)))) {
					if (G.at<float>(i, j) >= G.at<float>(i + 1, j) && G.at<float>(i, j) >= G.at<float>(i - 1, j)) {
						dst.at<float>(i, j) = G.at<float>(i, j);
					}
					else {
						dst.at<float>(i, j) = 0;
					}
				}
				else if ((phi.at<float>(i, j) > (5 * CV_PI / 8) && phi.at<float>(i, j) < (7 * CV_PI / 8)) || ((phi.at<float>(i, j) > (-3 * CV_PI / 8)) && (phi.at<float>(i, j) < (CV_PI / 8)))) {
					if (G.at<float>(i, j) >= G.at<float>(i + 1, j + 1) && G.at<float>(i, j) >= G.at<float>(i - 1, j - 1)) {
						dst.at<float>(i, j) = G.at<float>(i, j);
					}
					else {
						dst.at<float>(i, j) = 0;
					}
				}
			}
		}

		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				maxima.at<uchar>(i, j) = (uchar) dst.at<float>(i, j);
			}
		}

		float p = 0.1;
		int histogram[256] = { 0 };
		for (int i = 1; i < src.rows - 1; i++) {
			for (int j = 1; j < src.cols - 1; j++) {
				histogram[maxima.at<uchar>(i, j)]++;
			}
		}
		int edge = p * ((src.cols - 2) * (src.rows - 2) - histogram[0]);

		int sum = 0;
		int threshold = 0;

		for (int i = 255; i > 0; i--) {
			sum += histogram[i];
			if (sum > edge) {
				threshold = i;
				break;
			}
		}

		float thresholdHigh = (float)threshold;
		float k = 0.4;
		float thresholdLow = k * threshold;

		Mat addaptiveMat = maxima.clone();
		for (int i = 1; i < src.rows - 1; i++) {
			for (int j = 1; j < src.cols - 1; j++) {
				if (maxima.at<uchar>(i, j) < thresholdLow) {
					addaptiveMat.at<uchar>(i, j) = 0;
				}
				else if (maxima.at<uchar>(i, j) > thresholdHigh) {
					addaptiveMat.at<uchar>(i, j) = 255;
				}
				else {
					addaptiveMat.at<uchar>(i, j) = 127;
				}
			}
		}

		imshow("mid", addaptiveMat);
		int di[8] = { -1,-1,-1, 0, 0, 1,1,1 };
		int dj[8] = { -1, 0, 1, -1,1, -1,0,1 };

		for (int i = 1; i < src.rows - 1; i++) {
			for (int j = 1; j < src.cols - 1; j++) {
				if (addaptiveMat.at<uchar>(i, j) == 255) {

					std::queue<Point2i> Q;
					Q.push(Point2i(j, i));

					while (!Q.empty()) {
						Point2i p = Q.front();
						Q.pop();

						for (int k = 0; k < 8; k++) {
							if (addaptiveMat.at<uchar>(p.y + di[k], p.x + dj[k]) == 127) {
								Q.push(Point2i(p.x + dj[k], p.y + di[k]));

								addaptiveMat.at<uchar>(p.y + di[k], p.x + dj[k]) = 255;
							}
						}

					}

				}
			}
		}

		for (int k = 1; k < addaptiveMat.rows - 1; k++) {
			for (int l = 1; l < addaptiveMat.cols - 1; l++) {
				if (addaptiveMat.at<uchar>(k, l) == 127) {
					addaptiveMat.at<uchar>(k, l) = 0;
				}
			}
		}


		imshow("Original Image", src);
		imshow("Final", addaptiveMat);
		waitKey(0);
	}
}

Mat get_dft(Mat I) {
	Mat image;
	I.convertTo(image, CV_32F);
	Mat planes[] = { Mat_<float>(image), Mat::zeros(image.size(), CV_32F) };
	Mat complexI;
	merge(planes, 2, complexI);
	dft(complexI, complexI);
	return complexI;
}

Mat get_spectrum(Mat I) {
	Mat complexI = get_dft(I);
	Mat planes[2];
	split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
	magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
	Mat magI = planes[0];
	multiply(magI, magI, magI);
	return magI;
}



void shift(Mat magI) {

	// crop if it has an odd number of rows or columns
	magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));

	int cx = magI.cols / 2;
	int cy = magI.rows / 2;

	Mat q0(magI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
	Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
	Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
	Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right

	Mat tmp;                            // swap quadrants (Top-Left with Bottom-Right)
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	q1.copyTo(tmp);                     // swap quadrant (Top-Right with Bottom-Left)
	q2.copyTo(q1);
	tmp.copyTo(q2);
}

void wienerFilter()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat ceva = imread(fname, IMREAD_COLOR);
		Mat blur;
		cvtColor(ceva, blur, COLOR_BGR2GRAY);
		Mat src;
		GaussianBlur(blur, src, Size(7, 7), 0);
		Mat raw_sample;
		cvtColor(ceva, raw_sample, COLOR_BGR2GRAY);
		int defaultNoise = 50;
		Mat padded;
		int m = getOptimalDFTSize(src.rows);
		int n = getOptimalDFTSize(src.cols);
		copyMakeBorder(src, padded, 0, m - src.rows,0, n - src.cols, BORDER_CONSTANT, Scalar::all(0));

		Mat noisy = src.clone();
		Mat noise(padded.rows, padded.cols, CV_8U);
		Mat noisef = Mat::zeros(padded.rows, padded.cols, CV_32F);
		randn(noisef, Scalar::all(0), Scalar::all(defaultNoise));
		noisef.convertTo(noise, CV_8U);
		noisy += noise;


		Mat kernel = Mat(5, 5, CV_32FC1, Scalar(0.04));

		int w = padded.size().width - kernel.cols;
		int h = padded.size().height - kernel.rows;

		int r = w / 2;
		int l = padded.size().width - kernel.cols - r;

		int b = h / 2;
		int t = padded.size().height - kernel.rows - b;

		Mat sample(padded.rows, padded.cols, CV_8UC1);
		resize(raw_sample, sample, sample.size());
		Mat spectrum = get_spectrum(sample);


		Mat padded_noise;
		int x = getOptimalDFTSize(noisy.rows);
		int y = getOptimalDFTSize(noisy.cols);
		copyMakeBorder(src, padded_noise, 0, x - src.rows, 0, y - src.cols, BORDER_CONSTANT, Scalar::all(0));
		Mat noise_spectrum = get_spectrum(padded_noise);

		Mat planes[2];
		Mat complexI = get_dft(padded_noise);
		split(complexI, planes);

		Mat factor = (noise_spectrum / spectrum);

		Mat mask;
		copyMakeBorder(kernel, mask, t, b, l, r, BORDER_CONSTANT, Scalar::all(0));
		shift(mask);

		Mat mplanes[] = { Mat_<float>(mask), Mat::zeros(mask.size(), CV_32F) };
		Mat kernelComplex;
		merge(mplanes, 2, kernelComplex);

		dft(kernelComplex, kernelComplex);
		split(kernelComplex, mplanes);
		Mat val = mplanes[0].clone();

		magnitude(mplanes[0], mplanes[1], mplanes[0]);
		Mat magI = mplanes[0];
		multiply(magI, magI, magI);
		factor += magI;
		magI = magI / factor;
		magI = magI / val;
		factor = magI;
		
		multiply(planes[0], factor, planes[0]);	
		multiply(planes[1], factor, planes[1]);

		merge(planes, 2, complexI);
		idft(complexI, complexI);
		split(complexI, planes);
		Scalar enhanced_mean = mean(planes[0]);
		Scalar padded_mean = mean(padded);
		double norm = padded_mean.val[0] / enhanced_mean.val[0];
		multiply(planes[0], norm, planes[0]);
		Mat normalized;
		planes[0].convertTo(normalized, CV_8UC1);

		imshow("original",blur);
		imshow("initial", noisy);
		imshow("result", normalized);
		waitKey();
	}
}

void testParcurgereSimplaDiblookStyle()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// the fastest approach using the “diblook style”
		uchar *lpSrc = src.data;
		uchar *lpDst = dst.data;
		int w = (int) src.step; // no dword alignment is done !!!
		for (int i = 0; i<height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i*w + j];
				lpDst[i*w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height,width,CV_8UC1);

		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i,j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i,j) = (r+g+b)/3;
			}
		}
		
		imshow("input image",src);
		imshow("gray image",dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// Componentele d eculoare ale modelului HSV
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// definire pointeri la matricele (8 biti/pixeli) folosite la afisarea componentelor individuale H,S,V
		uchar* lpH = H.data;
		uchar* lpS = S.data;
		uchar* lpV = V.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, COLOR_BGR2HSV);

		// definire pointer la matricea (24 biti/pixeli) a imaginii HSV
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i<height; i++)
		{
			for (int j = 0; j<width; j++)
			{
				int hi = i*width * 3 + j * 3;
				int gi = i*width + j;

				lpH[gi] = hsvDataPtr[hi] * 510 / 360;		// lpH = 0 .. 255
				lpS[gi] = hsvDataPtr[hi + 1];			// lpS = 0 .. 255
				lpV[gi] = hsvDataPtr[hi + 2];			// lpV = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1,dst2;
		//without interpolation
		resizeImg(src,dst1,320,false);
		//with interpolation
		resizeImg(src,dst2,320,true);
		imshow("input image",src);
		imshow("resized image (without interpolation)",dst1);
		imshow("resized image (with interpolation)",dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src,dst,gauss;
		src = imread(fname,IMREAD_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int) k*pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss,dst,pL,pH,3);
		imshow("input image",src);
		imshow("canny",dst);
		waitKey();
	}
}

void testVideoSequence()
{
	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}
		
	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, COLOR_BGR2GRAY);
		Canny(grayFrame,edges,40,100,3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = waitKey(0);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n"); 
			break;  //ESC pressed
		};
	}
}


void testSnap()
{
	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];
	
	// video resolution
	Size capS = Size((int)cap.get(CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, WINDOW_AUTOSIZE);
	moveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, WINDOW_AUTOSIZE);
	moveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;
		
		imshow(WIN_SRC, frame);

		c = waitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115){ //'s' pressed - snapp the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess) 
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == EVENT_LBUTTONDOWN)
		{
			printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
				x, y,
				(int)(*src).at<Vec3b>(y, x)[2],
				(int)(*src).at<Vec3b>(y, x)[1],
				(int)(*src).at<Vec3b>(y, x)[0]);
		}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}



int main()
{
	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Open image\n");
		printf(" 2 - Open BMP images from folder\n");
		printf(" 3 - Image negative - diblook style\n");
		printf(" 4 - BGR->HSV\n");
		printf(" 5 - Resize image\n");
		printf(" 6 - Canny edge detection\n");
		printf(" 7 - Edges in a video sequence\n");
		printf(" 8 - Snap frame from live video\n");
		printf(" 9 - Mouse callback demo\n");
		printf(" 10 - Change gray scale level\n");
		printf(" 11 - Four Color Squares\n");
		printf(" 12 - Inverse Matrix\n");
		printf(" 13 - Split Channels\n");
		printf(" 14 - Color to Grayscale\n");
		printf(" 15 - Grayscale to BW\n");
		printf(" 16 - RGB to HSV\n");
		printf(" 17 - Histogram\n");
		printf(" 18 - Histogram with Accumulator\n");
		printf(" 19 - Multilevel Thresholding\n");
		printf(" 20 - Geometrical features\n");
		printf(" 21 - Labeling BFS traversal\n");
		printf(" 22 - Labeling two pass traversal\n");
		printf(" 23 - Draw Contour From Image\n");
		printf(" 24 - Draw Contour From Txt File\n");
		printf(" 25 - Dilation\n");
		printf(" 26 - Erosion\n");
		printf(" 27 - Open\n");
		printf(" 28 - Close\n");
		printf(" 29 - Borderline\n");
		printf(" 30 - Mean and Standard Deviation Computation\n");
		printf(" 31 - Basic Thresholding Algorithm\n");
		printf(" 32 - Brightness Change\n");
		printf(" 33 - Contrast Change\n");
		printf(" 34 - Gamma Correction\n");
		printf(" 35 - Histogram Equalization\n");
		printf(" 36 - Mean Filter\n");
		printf(" 37 - Gaussian Filter\n");
		printf(" 38 - Laplace Filter\n");
		printf(" 39 - High Pass Filter\n");
		printf(" 40 -  Ideal Cut Low Pass Filter\n");
		printf(" 41 -  Ideal Cut High Pass Filter\n");
		printf(" 42 -  Gaussian Cut Low Pass Filter\n");
		printf(" 43 -  Gaussian Cut High Pass Filter\n");
		printf(" 45 -  Median Filter for Salt and Pepper Noise\n");
		printf(" 46 -  Gaussian Filter 1x2D\n");
		printf(" 47 -  Gaussian Filter 2x1D\n");
		printf(" 48 -  Gaussian Filter \n");
		printf(" 49 - Gradient Magnitude and Orientation Computation\n");
		printf(" 50 - Non-maxima Suppression \n");
		printf(" 51 - Canny Edge Detection \n");
		printf(" 100 - Image Restoration Project \n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d",&op);
		switch (op)
		{
			case 1:
				testOpenImage();
				break;
			case 2:
				testOpenImagesFld();
				break;
			case 3:
				testParcurgereSimplaDiblookStyle(); //diblook style
				break;
			case 4:
				//testColor2Gray();
				testBGR2HSV();
				break;
			case 5:
				testResize();
				break;
			case 6:
				testCanny();
				break;
			case 7:
				testVideoSequence();
				break;
			case 8:
				testSnap();
				break;
			case 9:
				testMouseClick();
				break;
			case 10:
				L1_changeGrayLevels();
				break;
			case 11:
				L1_fourSquareColors();
				break;
			case 12:
				L1_inverseMatrix();
				break;
			case 13:
				L2_splitChannels();
				break;
			case 14:
				L2_colorToGrayscale();
				break;
			case 15:
				L2_grayToBw();
				break;
			case 16:
				L2_rgbToHsv();
				break;
			case 17:
				L3_histogram();
				break;
			case 18:
				L3_histogramAccumulators();
				break;
			case 19:
				L3_multilevelThr();
				break;
			case 20:
				L4_GeometricalFeatures();
				break;
			case 21:
				L5_BFS();
				break;
			case 22:
				L5_twoPass();
				break;
			case 23:
				L6_contourAlg();
				break;
			case 24:
				L6_drawContour();
				break;
			case 25:
				L7_dilation();
				break;
			case 26:
				L7_erode();
				break;
			case 27:
				L7_open();
				break;
			case 28:
				L7_close();
				break;
			case 29:
				L7_extractBoundary();
				break;
			case 101:
				L7_open();
				break;
			case 30:
				L8_meanAndDeviation();
				break;
			case 31:
				L8_thresholdAlg();
				break;
			case 32:
				L8_changeBrightness();
				break;
			case 33:
				L8_contrastChange();
				break;
			case 34:
				L8_gammaCorrection();
				break;
			case 35:
				L8_histogramEqualization();
				break;
			case 36:
				L9_meanFilter();
				break;
			case 37:
				L9_gaussianFilter();
				break;
			case 38:
				L9_laplaceFilter();
				break;
			case 39:
				L9_highPassFilter();
				break;
			case 40:
				L9_idealCutLPF();
				break;
			case 41:
				L9_idealCutHPF();
				break;
			case 42:
				L9_gaussianCutLPF();
				break;
			case 43:
				L9_gaussianCutHPF();
				break;
			case 45:
				L10_medianFilter();
				break;
			case 46:
				L10_gaussian1x2D();
				break;
			case 47:
				L10_gaussian2x1D();
				break;
			case 48:
				L11_gaussianFiltering();
				break;
			case 49:
				L11_gradientMagnitude();
				break;
			case 50:
				L11_nonMaximaSupression();
				break;
			case 51:
				L12_adaptiveHysteresisThresholding();
				break;
			case 100:
				wienerFilter();
				break;
		}
	}
	while (op!=0);
	return 0;
}