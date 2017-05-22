#include "opencv2\opencv.hpp"
#include "opencv2\highgui.hpp"
#include "opencv2\xfeatures2d.hpp"
#include "opencv2\features2d.hpp"
#include "opencv2\imgproc.hpp"
#include "opencv2\core\mat.hpp"
#include <iostream>
#include <cmath>
#include <string>

using namespace std;
using namespace cv;
using namespace xfeatures2d;

string type2str(int type) { //для отладки
	string r;

	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch (depth) {
	case CV_8U:  r = "8U"; break;
	case CV_8S:  r = "8S"; break;
	case CV_16U: r = "16U"; break;
	case CV_16S: r = "16S"; break;
	case CV_32S: r = "32S"; break;
	case CV_32F: r = "32F"; break;
	case CV_64F: r = "64F"; break;
	default:     r = "User"; break;
	}

	r += "C";
	r += (chans + '0');

	return r;
}

string int2str(const int &binNum) //мб передалать в char
{
	switch (binNum)
	{
	case 0: return "0";
	case 1: return "1";
	case 10: return "2";
	case 11: return "3";
	case 100: return "4";
	case 101: return "5";
	case 110: return "6";
	case 111: return "7";
	case 1000: return "8";
	case 1001: return "9";
	case 1010: return "A";
	case 1011: return "B";
	case 1100: return "C";
	case 1101: return "D";
	case 1110: return "E";
	case 1111: return "F";
	default:
		break;
	}
}



void hashImage(const Mat &img)
{
	Size size(8, 8);
	std::string hashOfImg = "";
	int k = 0;
	int base = 10;
	Mat dstImg;
	cv::resize(img, dstImg, size);
	cvtColor(dstImg, dstImg, CV_BGR2GRAY);
	Mat1b mask(dstImg.rows, dstImg.cols);
	Scalar averageColor_sc = mean(dstImg, mask);
	double averageColor = averageColor_sc.val[0];
	int tmp = 0;
	int l = 0;
	for (int i = 0; i < 8; ++i)
	{
		for (int j = 0; j < 8; ++j)
		{
			Scalar intensivity = dstImg.at<uchar>(Point(i, j));
			if (intensivity.val[0] > averageColor)
			{
				tmp += 1 * pow(10, 3-k);
			}	
			++k;
			if (k == 4)
			{
				hashOfImg += int2str(tmp);
				tmp = 0;
				k = 0;
				l = 0;
			}
		}
	 }
	cout << "Hash of Image = " << hashOfImg << endl;
}

int calcHammingDistance(string &num1, string &num2)
{
	int count = 0;
	for (auto &to1 : num1)
		for (auto &to2 : num2)
			if (to1 != to2)
				++count;
	return count;
}

bool areSameImg(string &img1, string &img2)
{
	if (calcHammingDistance(img1, img2) >= 10)
		return false;
	else
		return true;
}


int main(){
	Mat img = imread("../data/13-0.jpg",1);
	Mat img2;
	Ptr<Feature2D> f2d = SIFT::create();
	vector<KeyPoint> keypoints;
	f2d->detect(img, keypoints);
	if (!keypoints.empty())
	{
		drawKeypoints(img, keypoints, img2,cv::Scalar::all(-1),4);  
		imshow(" ", img2);
		const Scalar blue = Scalar(255, 0, 0);
		const Scalar green = Scalar(0, 255, 0);
		const Scalar red = Scalar(0, 0, 255);
		Mat M, rotImg, cropped;
		for (int i = 0; i < keypoints.size(); ++i)
		{
			if (keypoints[i].size > 60)
			{
				double angle = keypoints[i].angle;
				const double coordX = keypoints[i].pt.x;
				const double coordY = keypoints[i].pt.y;
				const double sizeOfKeypoint = keypoints[i].size;
				RotatedRect rRect = RotatedRect(Point2d(coordX ,
					coordY), Size2d(sizeOfKeypoint, sizeOfKeypoint),angle);
				Point2f vertices[4];
				Size rectSize = rRect.size;
				rRect.points(vertices);
				for (int i = 0; i < 4; i++)
					line(img2, vertices[i], vertices[(i + 1) % 4], blue);
				if (angle < -45.0) // вынести в функцию
				{
					angle += 90.0;
					swap(rectSize.height, rectSize.width);
				}
				M = getRotationMatrix2D(rRect.center, angle, 1.0);
				warpAffine(img2, rotImg, M, img2.size(), INTER_CUBIC);
				getRectSubPix(rotImg, rectSize, rRect.center, cropped);
				hashImage(cropped);
				imshow("cropped image", cropped);
				break;
			}
		}
		namedWindow("Step 2 draw Rectangle", WINDOW_AUTOSIZE);
		imshow("Step 2 draw Rectangle", img2);
		waitKey(0);
	}
	else
		cout << "No keypoints" << endl;


	return 0;
}


