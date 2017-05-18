#include "opencv2\opencv.hpp"
#include "opencv2\highgui.hpp"
#include "opencv2\xfeatures2d.hpp"
#include "opencv2\features2d.hpp"
#include "opencv2\imgproc.hpp"
#include "opencv2\core\mat.hpp"
#include <iostream>
#include <cmath>

using namespace std;
using namespace cv;
using namespace xfeatures2d;

void hashImage(const Mat &img)
{
	Size size(8, 8);
	Mat dstImg;
	cv::resize(img, dstImg, size);
	cvtColor(dstImg, dstImg, CV_BGR2GRAY);
	imwrite("greyimg.jpeg",dstImg);
			//Vec3b intensity = dstImg.at<Vec3b>(i,j); //что здесь вообще происходит?
	 Mat1b mask(dstImg.rows, dstImg.cols);
	 Scalar averageColor = mean(dstImg, mask);
	 
		
}

int main(){
	Mat img = imread("../data/apple-logo_318-40184.jpg",1);
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


