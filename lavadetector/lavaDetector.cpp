#include "lavaDetector.h"
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/legacy/legacy.hpp>
using namespace std;
using namespace cv;

//��������������
lavaDetector::lavaDetector():iPeriod(20),dThreshold(30)
{
	iNumOfFrames = 0;
}
lavaDetector::lavaDetector(int period, double thresh) : iPeriod(period), dThreshold(thresh)
{
	iNumOfFrames = 0;
}
lavaDetector::~lavaDetector()
{

}
//public:
int lavaDetector::readCurrentFrame(Mat src, int mode/*�趨��ǰ����ģʽ*/)//��ȡ��ǰ֡
{
	iWorkmode = mode;
	if (iWorkmode != 0 && iWorkmode != 1 && iWorkmode != 2)
	{
		return -1;//����ģʽ����
	}
	if (2 == iWorkmode)//�����ģʽ3������iPeriodΪ���ڣ���֡�����м���
	{
		iNumOfFrames++;
		if (iPeriod <= iNumOfFrames)//���ڽ�����֡������
		{
			iNumOfFrames = 1;
		}
	}
	src.copyTo(curImage_color);//��ȡ��ǰ��ɫ֡
	colorToBinary(curImage_color, curImage_binary, dThreshold);//��ɫ֡תΪ��ֵ֡
	selectContours(curImage_binary);//��������С����ɸѡ���ų�������
	src.copyTo(imageToShow_color);//���Ƶ�����ʾ��֡��
	sizeOfCur = curImage_color.size();//��ȡ�ߴ�
	return 0;
}
int lavaDetector::staticImageDetect()//��ͷ��̬���ݼ�� mode 0
{
	return 0;
}

int lavaDetector::dynamicImageDetect()//��ͷ��̬���ݼ�� mode 1
{
	return 0;
}
int lavaDetector::imageDetect()//���޼�� mode 2:
{
	if (iWorkmode != 2)
	{
		return -1;//ģʽ����
	}

	if (iNumOfFrames >= 1 && iNumOfFrames <= 10)//ǰ10֡�ؼ��㶨λ
	{
		if (1 == iNumOfFrames)
		{
			vDescriptors.clear();
			vKeyPoints.clear();//ÿ֡��ORB�ؼ���*
			vPoints.clear();//ƥ���ÿ֡�ı�ѡ��**/
			pPosOfBase.x = 0;
			pPosOfBase.y = 0;
		}
		getPosOfBase2(pPosOfBase, iNumOfFrames);//ȷ����ǰ���ڵĹؼ���
	}
	else if (iNumOfFrames >= 11 && iNumOfFrames <= 15)//11-15֡ȷ�����޼��㡢���ɼ��㡢������
	{
		

	}
	else//����������
	{
		

	}
	return 0;
}
int lavaDetector::resetAll()//�����ڲ�����,Ԥ���ӿ�
{
	return 0;
}
int lavaDetector::getBinaryImage(Mat& imagetoshow_binary)//����ʱ������ʾ��ǰ��ֵͼ
{
	curImage_binary.copyTo(imagetoshow_binary);
	return 0;
}
int lavaDetector::getImageToShow(Mat& imagetoshow)//���Ʋ���ȡ����ʾ�ļ����,ע�ⲻͬģʽ����ʾ��ͼ��Ҳ��ͬ��
{
	if (0 == iWorkmode)
	{

	}
	else if (1 == iWorkmode)
	{

	}
	else if (2 == iWorkmode)
	{
		if(iNumOfFrames >= 11)
		{
			circle(imageToShow_color, pPosOfBase, 5, Scalar(255, 0, 0), 3);//���ƹؼ���
		}
		imageToShow_color.copyTo(imagetoshow);
		
	}
	return 0;
}
//private:
int lavaDetector::colorToBinary(Mat src_color, Mat& dst_binary, double dThreshold)//�������ԭʼ��ɫ֡ת��Ϊ��ֵ֡
{
	Mat gray;
	cvtColor(src_color,gray,CV_BGR2GRAY);
	threshold(gray, dst_binary, dThreshold, 255, THRESH_BINARY);//��ֵ�ָ�
	return 0;
}

int lavaDetector::selectContours(Mat& binary)//�Զ�ֵͼ�еĶ���������ɸѡ��ȥ�������㣬����׶�
{
	vector< vector<Point> > contours;   // ����   
	vector< vector<Point> > filterContours; // ɸѡ�������
	vector< Vec4i > hierarchy;    // �����Ľṹ��Ϣ 
	contours.clear();
	hierarchy.clear();
	filterContours.clear();

	findContours(binary, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	// ȥ��α���� 
	for (size_t i = 0; i < contours.size(); i++)
	{
		if (fabs(contourArea(Mat(contours[i]))) > 2000)  //�ж��������ֵ
		{
			filterContours.push_back(contours[i]);
		}
	}
	binary.setTo(0);
	drawContours(binary, filterContours, -1, Scalar(255), CV_FILLED); //8, hierarchy);   
	return 0;
}

int lavaDetector::getPosOfBase(Point& pposofbase, int frames)//����ȷ����׼���λ��
{
	/*	vector<Mat> vDescriptors;//*
	vector<vector<KeyPoint>> vKeyPoints;//ÿ֡��ORB�ؼ���*
	vector<vector<Point>> vPoints;//ƥ���ÿ֡�ı�ѡ��**/
	const double rate = 0.4;//ȥ���ϰ벿�ֵı���
	Mat grayImage;
	cvtColor(curImage_color, grayImage, CV_BGR2GRAY);
	Mat ROIImage = grayImage(Range((int)(rate*sizeOfCur.height), sizeOfCur.height), Range(0, sizeOfCur.width));
	/*��ǰ֡����������*/
	OrbFeatureDetector featureDetector;//ORB������
	vector<KeyPoint> keyPoints;
	Mat descriptors;
	featureDetector.detect(ROIImage, keyPoints);//��������ؼ���

	OrbDescriptorExtractor featureExtractor;//ORB��������ȡ��
	featureExtractor.compute(ROIImage, keyPoints, descriptors);
	//�����Ա����
	vDescriptors.push_back(descriptors);//ÿ֡��ORB�ؼ���������
	vKeyPoints.push_back(keyPoints);//ÿ֡��ORB�ؼ���*

	/*��֡������ƥ��*/
	if (iNumOfFrames >= 4)
	{
		//ʵ����һ��ƥ����
		BruteForceMatcher <L2<float>> matcher;
		vector<DMatch> matches;//ƥ����
		matcher.match(vDescriptors[iNumOfFrames-4], vDescriptors[iNumOfFrames-1], matches);
		//ƥ�������
		nth_element(matches.begin(), matches.begin() + 2, matches.end());
		matches.erase(matches.begin() + 3, matches.end());//����
		//�ѹؼ���ת��Ϊ��ͨ��,������vector<vector<Point>> vPoints����Ϊ��ѡ��
		vector<Point> points;
		for (int i = 0; i < 3; i++)
		{
			points.push_back(keyPoints[matches[i].trainIdx].pt);
			points[i].y += (int)(rate*sizeOfCur.height);//�ָ�Ϊԭͼ���е������
		}
		vPoints.push_back(points);
		//����
		points.clear();
		matcher.clear();
		matches.clear();
	}
	keyPoints.clear();

	if (10 == iNumOfFrames)//��10֡ʱ��ȷ����׼��
	{
		pPosOfBase = vPoints[0][0];
		for (int j = 0; j < 7; j++)
		{
			for (int i = 0; i < 3; i++)
			{
				if (vPoints[j][i].y > pPosOfBase.y)
				{
					pPosOfBase = vPoints[j][i];
				}
			}
		}
	}
	return 0;
}

int lavaDetector::getPosOfBase2_collect(Point& pposofbase, int frames)//����ȷ����׼���λ��
{
	/*	vector<Mat> vDescriptors;//*
	vector<vector<KeyPoint>> vKeyPoints;//ÿ֡��ORB�ؼ���*
	vector<vector<Point>> vPoints;//ƥ���ÿ֡�ı�ѡ��**/
	const double rate = 0.4;//ȥ���ϰ벿�ֵı���
	Mat grayImage;
	cvtColor(curImage_color, grayImage, CV_BGR2GRAY);
	Mat ROIImage = grayImage(Range((int)(rate*sizeOfCur.height), sizeOfCur.height), Range(0, sizeOfCur.width));
	/*��ǰ֡����������*/
	OrbFeatureDetector featureDetector;//ORB������
	vector<KeyPoint> keyPoints;
	Mat descriptors;
	featureDetector.detect(ROIImage, keyPoints);//��������ؼ���

	OrbDescriptorExtractor featureExtractor;//ORB��������ȡ��
	featureExtractor.compute(ROIImage, keyPoints, descriptors);
	//�����Ա����
	vDescriptors.push_back(descriptors);//ÿ֡��ORB�ؼ���������
	vKeyPoints.push_back(keyPoints);//ÿ֡��ORB�ؼ���*

	/*��֡������ƥ��*/
	if (iNumOfFrames >= 4)
	{
		//ʵ����һ��ƥ����
		BruteForceMatcher <L2<float>> matcher;
		vector<DMatch> matches;//ƥ����
		matcher.match(vDescriptors[iNumOfFrames-4], vDescriptors[iNumOfFrames-1], matches);
		//ƥ�������
		nth_element(matches.begin(), matches.begin() + 2, matches.end());
		matches.erase(matches.begin() + 3, matches.end());//����
		//�ѹؼ���ת��Ϊ��ͨ��,������vector<vector<Point>> vPoints����Ϊ��ѡ��
		vector<Point> points;
		for (int i = 0; i < 3; i++)
		{
			points.push_back(keyPoints[matches[i].trainIdx].pt);
			points[i].y += (int)(rate*sizeOfCur.height);//�ָ�Ϊԭͼ���е������
		}
		vPoints.push_back(points);
		vMatches.push_back(matches);
		//����
		points.clear();
		matcher.clear();
		matches.clear();
	}
	keyPoints.clear();
	int thei = 0;
	int thej = 0;
	if (10 == iNumOfFrames)//��10֡ʱ��ȷ����׼��
	{
		pPosOfBase = vPoints[0][0];
		for (int j = 0; j < 7; j++)
		{
			for (int i = 0; i < 3; i++)
			{
				if (vPoints[j][i].y > pPosOfBase.y)
				{
					pPosOfBase = vPoints[j][i];
					thej = j;
					thei = i;
				}
			}
		}
		int index = vMatches[thej][thei].trainIdx;
		cout<<thej<<endl;
		cout<<thei<<endl;
		cout<<index<<endl;
		cout<<vDescriptors[thej+3].size().height<<endl;
		cout<<vDescriptors[thej+3].type()<<endl;
		Mat des = vDescriptors[thej+3](Rect(0,index,vDescriptors[thej+3].cols,1));//    Range(index,index),Range(0,vDescriptors[thej+3].cols));
		imwrite("vDescriptors.png",des);
	}
	
	return 0;
}

int lavaDetector::getPosOfBase2(Point& pposofbase, int frames)//����ȷ����׼���λ��
{
	/*	vector<Mat> vDescriptors;//*
	vector<vector<KeyPoint>> vKeyPoints;//ÿ֡��ORB�ؼ���*
	vector<vector<Point>> vPoints;//ƥ���ÿ֡�ı�ѡ��**/
	Mat theNormal = imread("vDescriptors.png",0);
	const double rate = 0.4;//ȥ���ϰ벿�ֵı���
	Mat grayImage;
	cvtColor(curImage_color, grayImage, CV_BGR2GRAY);
	Mat ROIImage = grayImage(Range((int)(rate*sizeOfCur.height), sizeOfCur.height), Range(0, sizeOfCur.width));
	/*��ǰ֡����������*/
	OrbFeatureDetector featureDetector;//ORB������
	vector<KeyPoint> keyPoints;
	Mat descriptors;
	featureDetector.detect(ROIImage, keyPoints);//��������ؼ���

	OrbDescriptorExtractor featureExtractor;//ORB��������ȡ��
	featureExtractor.compute(ROIImage, keyPoints, descriptors);
	//�����Ա����
	vDescriptors.push_back(descriptors);//ÿ֡��ORB�ؼ���������
	vKeyPoints.push_back(keyPoints);//ÿ֡��ORB�ؼ���*

	//ʵ����һ��ƥ����
	BruteForceMatcher <L2<float>> matcher;
	vector<DMatch> matches;//ƥ����
	matcher.match(theNormal, descriptors, matches);

	//�ѹؼ���ת��Ϊ��ͨ��,������vector<vector<Point>> vPoints����Ϊ��ѡ��
	vector<Point> points;
	points.push_back(keyPoints[matches[0].trainIdx].pt);
	distance[iNumOfFrames-1] = matches[0].distance;
	points[0].y += (int)(rate*sizeOfCur.height);//�ָ�Ϊԭͼ���е������

	vPoints.push_back(points);
	//����
	points.clear();
	matcher.clear();
	matches.clear();
	keyPoints.clear();

	if (10 == iNumOfFrames)//��10֡ʱ��ȷ����׼��
	{
		int small = 0;
		int num = distance[0];
		for(int i = 0;i<10;i++)
		{
			if(distance[i] < num)
			{
				num = distance[i];
				small = i;
			}
		}
		pPosOfBase = vPoints[small][0];
	}
	return 0;
}