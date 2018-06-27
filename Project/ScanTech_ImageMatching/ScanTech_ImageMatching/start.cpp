#include "stdafx.h"
#include <iostream>
#include "start.h"

using namespace cv;
using namespace std;
start::start()
{
	/*cv::Mat rgbd1 = cv::imread("image1.jpg");
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


	//���ӻ�����ʾ�ؼ���
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


	//���ӻ�
	Mat ShowMatches;
	drawMatches(rgbd1, Keypoints1, rgbd2, Keypoints2, matches, ShowMatches);
	imshow("matches", ShowMatches);
	waitKey(0);*/

	Mat image01 = imread("g2.jpg", 1);
	Mat image02 = imread("g4.jpg", 1);
	imshow("p2", image01);
	imshow("p1", image02);

	//�Ҷ�ͼת��  
	Mat image1, image2;
	cvtColor(image01, image1, CV_RGB2GRAY);
	cvtColor(image02, image2, CV_RGB2GRAY);


	//��ȡ������    
	OrbFeatureDetector OrbDetector(1000);  // ������������ȣ�ֵԽС��Խ�٣�Խ��׼ 
	vector<KeyPoint> keyPoint1, keyPoint2;
	OrbDetector.detect(image1, keyPoint1);
	OrbDetector.detect(image2, keyPoint2);

	//������������Ϊ�±ߵ�������ƥ����׼��    
	OrbDescriptorExtractor OrbDescriptor;
	Mat imageDesc1, imageDesc2;
	OrbDescriptor.compute(image1, keyPoint1, imageDesc1);
	OrbDescriptor.compute(image2, keyPoint2, imageDesc2);

	flann::Index flannIndex(imageDesc1, flann::LshIndexParams(12, 20, 2), cvflann::FLANN_DIST_HAMMING);

	vector<DMatch> GoodMatchePoints;

	Mat macthIndex(imageDesc2.rows, 2, CV_32SC1), matchDistance(imageDesc2.rows, 2, CV_32FC1);
	flannIndex.knnSearch(imageDesc2, macthIndex, matchDistance, 2, flann::SearchParams());

	// Lowe's algorithm,��ȡ����ƥ���
	for (int i = 0; i < matchDistance.rows; i++)
	{
		if (matchDistance.at<float>(i, 0) < 0.6 * matchDistance.at<float>(i, 1))
		{
			DMatch dmatches(i, macthIndex.at<int>(i, 0), matchDistance.at<float>(i, 0));
			GoodMatchePoints.push_back(dmatches);
		}
	}

	Mat first_match;
	drawMatches(image02, keyPoint2, image01, keyPoint1, GoodMatchePoints, first_match);
	imshow("first_match ", first_match);
	imwrite("first_match.jpg", first_match);
	waitKey();
}