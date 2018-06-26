#include "stdafx.h"
#include <iostream>
#include "start.h"

using namespace cv;
using namespace std;
start::start()
{
	cv::Mat rgbd1 = cv::imread("image1.jpg");
	cv::Mat rgbd2 = cv::imread("image2.jpg");

	Ptr<ORB> orb = ORB::create();
	vector<KeyPoint> Keypoints1, Keypoints2;
	Mat descriptors1, descriptors2;
	orb->detectAndCompute(rgbd1, Mat(), Keypoints1, descriptors1);
	orb->detectAndCompute(rgbd1, Mat(), Keypoints2, descriptors2);

	//Size dsize1 = Size(rgbd1.cols*0.5, rgbd1.rows*0.5);
	//Size dsize2 = Size(rgbd2.cols*0.5, rgbd2.rows*0.5);
	//resize(rgbd1, rgbd1, dsize1);
	//resize(rgbd2, rgbd2, dsize2);


	//可视化，显示关键点
	Mat ShowKeypoints1, ShowKeypoints2;
	drawKeypoints(rgbd1, Keypoints1, ShowKeypoints1);
	drawKeypoints(rgbd2, Keypoints2, ShowKeypoints2);

	imshow("Keypoints1", ShowKeypoints1);
	imshow("Keypoints2", ShowKeypoints2);
	waitKey(0);


	//Matching
	vector<DMatch> matches;
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce");
	matcher->match(descriptors1, descriptors2, matches);
	cout << "find out total " << matches.size() << " matches" << endl;


	//可视化
	Mat ShowMatches;
	drawMatches(rgbd1, Keypoints1, rgbd2, Keypoints2, matches, ShowMatches);
	imshow("matches", ShowMatches);
	waitKey(0);
}