#include "opencv2\opencv.hpp"
#include "opencv2\highgui.hpp"
#include "opencv2\xfeatures2d.hpp"
#include "opencv2\features2d.hpp"
#include "opencv2\imgproc.hpp"
#include "opencv2\core\mat.hpp"
#include <iostream>
#include <cmath>
#include <string>
#include <filesystem>
#include <set>

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



string hashImage(const Mat &img)
{
	Size size(8, 8);
	std::string hashOfImg = "";
	int k = 0;
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
	//cout << "Hash of Image = " << hashOfImg << endl;
	return hashOfImg;
}

int calcHammingDistance(string &num1, string &num2)
{
	int count = 0;
	for(int i=0;i<num1.size();i++)
	{
		if(num1[i]!=num2[i])
			++count;
	}
				
	return count;
}

bool areSameImg(string &img1, string &img2)
{
	if (calcHammingDistance(img1, img2) >= 10)
		return false;
	else
		return true;
}

void writeFileDirs(string &dir)
{
	namespace fs = std::experimental::filesystem;
	for (auto &to : fs::directory_iterator(dir))
		cout << to << endl;
	
}


vector<DMatch> deleteSame(vector<DMatch>& matches, vector<KeyPoint>& keys1)
{
	int i = 0;
	int j = -1;
	int N = matches.size();
	while (i < N)
	{
		while (j < N-1)
		{
			++j;
			if (i == j)
				continue;
			if (keys1[matches[i].queryIdx].pt == keys1[matches[j].queryIdx].pt)
			{
				if (matches[i].distance > matches[j].distance)
					matches.erase(matches.begin() + i);
				else
					matches.erase(matches.begin() + j);
				i = 0;
				j = 0;
				N = matches.size();
			
			}

		}
		i++;
		j = 0;
	}
	return matches;
}

void drawMatchesMine(const Mat& Img1, const Mat& Img2, const vector<KeyPoint>& Keypoints1,
	const vector<KeyPoint>& Keypoints2, const Mat& Descriptors1, const Mat& Descriptors2) {
	Ptr<DescriptorMatcher> descriptorMatcher = DescriptorMatcher::create("BruteForce");
	BFMatcher tmp_matcher;
	vector<DMatch> matches;
	vector<DMatch> goodMatches;
	double min_dist = 100;
	Mat matchImg;
	descriptorMatcher->match(Descriptors1, Descriptors2, matches);
	for (auto& match : matches)
	{
		if (match.distance < 300)
			goodMatches.push_back(match);
	}
	vector<KeyPoint> keys1 = Keypoints1;
	vector<DMatch> finalMatches = goodMatches;
	//finalMatches = deleteSame(goodMatches, keys1);
	drawMatches(Img1, Keypoints1, Img2, Keypoints2, finalMatches, matchImg, Scalar::all(-1), CV_RGB(255, 255, 255), Mat(), 4);
	
	vector<Point2f> obj;
	vector<Point2f> scene;



	for (int i = 0; i < finalMatches.size(); i++)
	{
		obj.push_back(Keypoints1[finalMatches[i].queryIdx].pt);
		scene.push_back(Keypoints2[finalMatches[i].trainIdx].pt);
	}

	Mat H = findHomography(obj,scene,CV_RANSAC);
	
	vector<Point2f> objCorners(4);
	objCorners[0] = Point2f(0, 0);
	objCorners[1] = Point2f(Img1.cols, 0);
	objCorners[2] = Point2f(Img1.cols, Img1.rows);
	objCorners[3] = Point2f(0, Img1.rows);
	vector<Point2f> sceneCorners(4);
	perspectiveTransform(objCorners, sceneCorners, H);

	line(matchImg, sceneCorners[0] + Point2f(Img1.cols, 0), sceneCorners[1] + Point2f(Img1.cols, 0), Scalar(0, 255, 0), 4);
	line(matchImg, sceneCorners[1] + Point2f(Img1.cols, 0), sceneCorners[2] + Point2f(Img1.cols, 0), Scalar(0, 255, 0), 4);
	line(matchImg, sceneCorners[2] + Point2f(Img1.cols, 0), sceneCorners[3] + Point2f(Img1.cols, 0), Scalar(0, 255, 0), 4);
	line(matchImg, sceneCorners[3] + Point2f(Img1.cols, 0), sceneCorners[0] + Point2f(Img1.cols, 0), Scalar(0, 255, 0), 4);

	imshow("match", matchImg);
	waitKey(0);
}

vector<KeyPoint> findKeyPoints(const Mat& img)
{
	vector<KeyPoint> keypoints;
	Ptr<Feature2D> f2d = SIFT::create();
	f2d->detect(img, keypoints);
	int i = 0;
	int amountOfkeypoints = keypoints.size();
	int vsize = 6; // сделать динамическим
	while (i < amountOfkeypoints)
	{
		if (keypoints[i].size < vsize) {
			keypoints.erase(keypoints.begin() + i);
			i = 0;
			amountOfkeypoints = keypoints.size();
		}
		else i++;
	}
	int k = 0;
	return keypoints;
}

Mat calculateDescriptors(const Mat& img,vector<KeyPoint>& keypoints)
{
	Mat descriptors;
	Ptr<Feature2D> f2d = SIFT::create();
	f2d->compute(img, keypoints, descriptors);
	return descriptors;
}

int i = 0;
vector<string> refactorKeyPoints(const vector<KeyPoint>& keypoints,const Mat& img)
{
	vector<string> res;
	string tmp;
	Mat M, rotImg, cropped;
	for (auto& key : keypoints)
	{
		double angle = key.angle;
		const double coordX = key.pt.x;
		const double coordY = key.pt.y;
		const double sizeOfKeypoint = key.size;
		RotatedRect rRect = RotatedRect(Point2d(coordX,coordY), Size2d(sizeOfKeypoint, sizeOfKeypoint),angle);
		Size rectSize = rRect.size;
		if (angle < -45.0) 
		{
			angle += 90.0;
			swap(rectSize.height, rectSize.width);
		}
		M = getRotationMatrix2D(rRect.center, angle, 1.0);
		warpAffine(img, rotImg, M, img.size(), INTER_CUBIC);
		getRectSubPix(rotImg, rectSize, rRect.center, cropped);
		tmp = hashImage(cropped);
		//imshow("cropped image", cropped);
		//if (i < 5)
		//{
			//string nameOfFile = to_string(i);
			//nameOfFile += ".jpeg";
			//imwrite(nameOfFile, cropped);
	//	}
		res.push_back(tmp);
		++i;
	}
	return res;
}


void compareKeypoints(vector<KeyPoint>& keys1, vector<KeyPoint>& keys2, const Mat& img1, const Mat& img2) //first img is logo, second img is source img
{
	vector<string> hexKeys1 = refactorKeyPoints(keys1,img1);
	vector<string> hexKeys2 = refactorKeyPoints(keys2, img2);
	int distance;
	int i = 0;
	int j = 0;
	for (auto& to1 : hexKeys1)
	{
		for (auto &to2 : hexKeys2)
		{
			distance = calcHammingDistance(to1, to2);
			//cout << "distance =" << distance << endl;
			if (distance < 10)
				cout << j << " " << i << endl;
			i++;
		}
		j++;
		i = 0;
	}
	//distance = calcHammingDistance(hexKeys1[0], hexKeys2[39]);
	//cout << "distance between 0 and 39 = " << distance << endl;
}


int main(){
	Mat img1 = imread("../data/Logo_PUMA.jpg",1);
	Mat img2 = imread("../data/pumaПример.jpg", 1);
	Mat img3;
	string dir("../data");
	vector<KeyPoint> keys1 = findKeyPoints(img1);
	//drawKeypoints(img1, keys1,img3, cv::Scalar::all(-1),4);
	//imshow("keypoints", img3);
	//waitKey(0);
	////refactorKeyPoints(keys1, img1);
	vector<KeyPoint> keys2 = findKeyPoints(img2);
	//drawKeypoints(img2, keys2,img3, cv::Scalar::all(-1),4);
	//imshow("keypoints", img3);
	//waitKey(0);
	Mat descriptors1 = calculateDescriptors(img1,keys1);
	Mat descriptors2 = calculateDescriptors(img2, keys2);
	compareKeypoints(keys1, keys2, img1, img2);
	drawMatchesMine(img1, img2, keys1, keys2, descriptors1, descriptors2);
	return 0;
}

