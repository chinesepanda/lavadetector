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
lavaDetector::lavaDetector():iPeriod(300)/*�������*/,dThreshold(25)/*�ָ����ֵ*/,iRadiusOfCircle_mm(400)/*���Ԥ���õİ뾶*/,iPeriodOfMod1(10)/*��̬��Ƶ������ڣ���֡��Ϊ��λ*/,THEMODE(0)
{
	iNumOfFrames = 0;//mod2��ǰ֡���
	iThickOfWool_p = 0;
	iPosOfZero_p = 1614;
	pPosOfBase_tmp.x = 1614;//
	pPosOfBase_tmp.y = 1349;
	WRONG = 0;
	WRONG_MOD1 = 0;
	iNumOfFramesOfMod1 = 0;//mod1��ǰ֡���
}
lavaDetector::lavaDetector(int period, double thresh, int posOfZero_p,int mode) : iPeriod(period), dThreshold(thresh),iRadiusOfCircle_mm(400),iPeriodOfMod1(10)/*��̬��Ƶ������ڣ���֡��Ϊ��λ*/,THEMODE(mode)
{
	iPosOfZero_p = posOfZero_p;//��ʼ���x����
	iNumOfFrames = 0;
	iThickOfWool_p = 0;
	pPosOfBase_tmp.x = 1614;
	pPosOfBase_tmp.y = 1349;
	WRONG = 0;
	 WRONG_MOD1 = 0;
	iNumOfFramesOfMod1 = 0;//mod1��ǰ֡���
}
lavaDetector::~lavaDetector()
{

}
//public:
int lavaDetector::readCurrentFrame(Mat src, int mode/*�趨��ǰ����ģʽ*/)//��ȡ��ǰ֡
{
	WRONG = 0;
	iWorkmode = mode;
	if (iWorkmode != 0 && iWorkmode != 1 && iWorkmode != 2)
	{
		return -1;//����ģʽ����
	}
	if(1 == iWorkmode)//�����ģʽ2,��֡�����м���
	{
		iNumOfFramesOfMod1++;
	}
	if (2 == iWorkmode)//�����ģʽ3������iPeriodΪ���ڣ���֡�����м���
	{
		iNumOfFrames++;
		if (iPeriod <= iNumOfFrames)//���ڽ�����֡������
		{
			iNumOfFrames = 1;
		}
	}
	/*ͨ�ô���*/
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
	if (iWorkmode != 1)//���ģʽ�Ƿ�ƥ��
	{
		return -1;//ģʽ����
	}
	
	if( iNumOfFramesOfMod1<= iPeriodOfMod1)//1-10֡
	{
		/*���������*/
		vector< vector<Point> > contours;   // ����   
		vector< Vec4i > hierarchy;    // �����Ľṹ��Ϣ 
		contours.clear();
		hierarchy.clear();
		Mat tmp_binary = curImage_binary.clone();
		findContours(tmp_binary, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
		long area = 0;
		for (size_t i = 0; i < contours.size(); i++)
		{
			area += fabs(contourArea(Mat(contours[i])));
		}
		vArea.push_back(area);//��������
		/*��������*/
		//��ԭͼ��ת��Ϊ�Ҷ�ͼ��������
		const double rate = 0.4;//ȥ���ϰ벿�ֵı���
		Mat grayImage;
		cvtColor(curImage_color, grayImage, CV_BGR2GRAY);
		Mat ROIImage = grayImage(Range((int)(rate*sizeOfCur.height), sizeOfCur.height), Range(0, sizeOfCur.width));
		//��ǰ֡����������
		OrbFeatureDetector featureDetector;//ORB������
		vector<KeyPoint> keyPoints;//�ؼ���
		Mat descriptors;//�ؼ�������������
		featureDetector.detect(ROIImage, keyPoints);//��������ؼ���
		OrbDescriptorExtractor featureExtractor;//ORB��������ȡ��
		featureExtractor.compute(ROIImage, keyPoints, descriptors);//����ؼ��������������
		//�����Ա��������
		double numOfKeypoints;
		if(keyPoints.size() > 10)
		{
			numOfKeypoints = 10.0;
		}
		else
		{
			numOfKeypoints = (double)(keyPoints.size());
		}
		vNumOfKeyPoints.push_back(numOfKeypoints);//���뵱ǰ֡������������
		keyPoints.clear();
		/*������ڽ�����ͳ��*/
		if(iPeriodOfMod1 == iNumOfFramesOfMod1)//mod1��10֡ʱ������ƽ���������
		{
			long meanArea = 0;
			double meanNumOfKeypoints = 0.0;
			for(int i = 0;i < iPeriodOfMod1;i++)
			{
				meanArea += (vArea[i]/iPeriodOfMod1);
				meanNumOfKeypoints += vNumOfKeyPoints[i]/iPeriodOfMod1;
			}
			vArea.clear();//����
			vNumOfKeyPoints.clear();
			cout<<"meanArea"<<meanArea<<endl;
			if(meanArea < 120000)
			{
				 WRONG_MOD1 = -1;//��������С
			}
			if(meanNumOfKeypoints < 5)
			{
				 WRONG_MOD1 = -2;//���������
			}
		}
	}
	return 0;
}
int lavaDetector::imageDetect()//���޼�� mode 2:
{
	WRONG = 0;
	if (iWorkmode != 2)
	{
		return -1;//ģʽ����
	}

	if (iNumOfFrames >= 1 && iNumOfFrames <= 10)//ǰ10֡�ؼ��㶨λ
	{
		if (1 == iNumOfFrames)//��һ֡��ز�������
		{
			vDescriptors.clear();
			vKeyPoints.clear();//ÿ֡��ORB�ؼ���*
			vPoints.clear();//ƥ���ÿ֡�ı�ѡ��**/
			pPosOfBase.x = 0;
			pPosOfBase.y = 0;
		}

		if(THEMODE == 0)//�Զ����ģʽ
		{
			getPosOfBase2();//ȷ����ǰ���ڵĻ�׼��
		}
		else//Ԥ����ģʽ
		{
			pPosOfBase = pPosOfBase_tmp;//Ԥ���û�׼��
		}
	}
	else if (iNumOfFrames >= 11 && iNumOfFrames <= 21)//11-15֡ȷ�����޼��㡢���ɼ��㡢������
	{
		if(THEMODE == 0)//�Զ����ģʽ
		{
			getPosOfDetect();//ȷ�����޼��㡢�����ߡ����Ҽ���
		}
		else//Ԥ����ģʽ
		{
			pPosThickDetect.x = 1255;
			pPosThickDetect.y = 1018;
			pPosStreamDetect.x = 1614;
			pPosStreamDetect.y = 687;
			dScale = 1.11421;
		}
		
	}
	else//��22֡��ʼ������������� 
	{
		detectThickOfWool();//�����˺��
		detectWidthOfStream();//������ɿ�ȡ������
	}

	return 0;
}
int lavaDetector::setPosOfZero_p(int& posOfZero_p)//�������x����
{
	iPosOfZero_p = posOfZero_p;
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
		Point pointZero_tmp;
		pointZero_tmp.x = iPosOfZero_p;
		pointZero_tmp.y = 0;
		plotLine(imageToShow_color,pointZero_tmp,Scalar(255,0,0),0);//���

		if(iNumOfFrames >= 11)
		{
			plotPointTarget(imageToShow_color,pPosOfBase,Scalar(0,0,255),1);
		}
		if(iNumOfFrames > 21)
		{	
			plotPointTarget(imageToShow_color,pPosThickDetect,Scalar(255,0,0),2);
			plotPointTarget(imageToShow_color,pPosStreamDetect,Scalar(255,0,0),2);
			//���Ƴ��˴������ pPosThickPlot
			plotPointTarget(imageToShow_color,pPosThickPlot,Scalar(255,255,255),0);//���˻��Ƶ������
			plotLine(imageToShow_color,pPosThickDetect,Scalar(255,255,255),1);//���˼���
			//cout<<iThickOfWool_p<<endl;//������˺��dThickOfWool_mm
			//�������ɴ������
			plotPointTarget(imageToShow_color,pPosStreamPlot,Scalar(255,255,255),0);//���ɻ��Ƶ������
			//cout<<iWidthOfStream_p<<endl;
			//�������������
			plotLine(imageToShow_color,pPosStreamPlot,Scalar(255,255,255),0);//���ɻ��Ƶ�


		}
	}
	imageToShow_color.copyTo(imagetoshow);
	return 0;
}
int lavaDetector::getParameters_p(int& thickOfWool_p/*���˺��*/, int& widthOfStream_p/*���ɿ��*/,int& rePosOfDrop_p/*���ƫ����*/)//��ȡ�����������������Ϊ��λ��
{
	thickOfWool_p = iThickOfWool_p;
	widthOfStream_p = iWidthOfStream_p;
	rePosOfDrop_p = iRePosOfDrop_p;
	return 0;
}
int lavaDetector::getParameters_mm(double& thickOfWool_mm,double& widthOfStream_mm, double& rePosOfDrop_mm)//��ȡ�������������mmΪ��λ��
{
	//������ֵת��Ϊmmֵ
	dThickOfWool_mm = ((double)iThickOfWool_p) * dScale;
	thickOfWool_mm = dThickOfWool_mm;
	//������ֵת��Ϊmmֵ
	dWidthOfStream_mm = ((double)iWidthOfStream_p) * dScale;
	widthOfStream_mm = dWidthOfStream_mm;
	//������ֵת��Ϊmmֵ
	dRePosOfDrop_mm = ((double)iRePosOfDrop_p) * dScale;
	rePosOfDrop_mm = dRePosOfDrop_mm;
	return 0;
}
int lavaDetector::getWRONG_2(int& wrong)
{
	wrong = WRONG;
	return 0;
}
int lavaDetector::getWRONG_1(int& wrong)
{
	wrong = WRONG_MOD1;
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
	//���ȫ��Ϊ��
	if(filterContours.size() == 0)
	{
		WRONG = -4;//���󣺵�ǰ֡��Ϊ��Ŀ��
	}
	binary.setTo(0);
	drawContours(binary, filterContours, -1, Scalar(255), CV_FILLED); //8, hierarchy);   
	return 0;
}

int lavaDetector::getPosOfBase()//����ȷ����׼���λ��
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

int lavaDetector::getPosOfBase2_collect()//����ȷ����׼���λ��
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

int lavaDetector::getPosOfBase2()//����ȷ����׼���λ�ã���������
{
	/*	vector<Mat> vDescriptors;//*
	vector<vector<KeyPoint>> vKeyPoints;//ÿ֡��ORB�ؼ���*
	vector<vector<Point>> vPoints;//ƥ���ÿ֡�ı�ѡ��**/
	Mat theNormal = imread("vDescriptors.png",0);//����Ŀ������
	const double rate = 0.4;//ȥ���ϰ벿�ֵı���
	//��ԭͼ��ת��Ϊ�Ҷ�ͼ��������
	Mat grayImage;
	cvtColor(curImage_color, grayImage, CV_BGR2GRAY);
	Mat ROIImage = grayImage(Range((int)(rate*sizeOfCur.height), sizeOfCur.height), Range(0, sizeOfCur.width));
	//��ǰ֡����������
	OrbFeatureDetector featureDetector;//ORB������
	vector<KeyPoint> keyPoints;//�ؼ���
	Mat descriptors;//�ؼ�������������
	featureDetector.detect(ROIImage, keyPoints);//��������ؼ���
	OrbDescriptorExtractor featureExtractor;//ORB��������ȡ��
	featureExtractor.compute(ROIImage, keyPoints, descriptors);//����ؼ��������������
	//�����Ա��������
	vDescriptors.push_back(descriptors);//ÿ֡��ORB�ؼ���������
	vKeyPoints.push_back(keyPoints);//ÿ֡��ORB�ؼ���
	
	if(keyPoints.size() != 0)//�����ǰ֡���ڹؼ���
	{
		//ʵ����һ��ƥ����
		BruteForceMatcher <L2<float>> matcher;
		vector<DMatch> matches;//ƥ����
		matcher.match(theNormal, descriptors, matches);//Ѱ�ҵ�ǰ֡��ƥ���������

		//�ѹؼ���ת��Ϊ��ͨ��,������vector<vector<Point>> vPoints����Ϊ��ѡ�㣨Ϊ��֤���ַ����ļ��ݣ���������Ƕ�ף�
		vector<Point> points;//�ؼ���ת��Ϊ��ͨ��
		points.push_back(keyPoints[matches[0].trainIdx].pt);
		distance[iNumOfFrames-1] = matches[0].distance;//��ǰ֡��ƥ�����������Ŀ������������
		//cout<<"���룺 "<<matches[0].distance<<endl;
		points[0].y += (int)(rate*sizeOfCur.height);//�ָ�Ϊԭͼ���е������
		vPoints.push_back(points);//��ƥ�����������Ա��������
		index[iNumOfFrames-1] = 1;//index[]==1,˵����ǰ֡��������Ч
		points.clear();
	    matcher.clear();
	    matches.clear();
	}
	else//�����ǰ֡�����ڹؼ���
	{	
		vector<Point> points;
		vPoints.push_back(points);//��ƥ�����������Ա����������ռλ��
		index[iNumOfFrames-1] = 0;//index[]==0,˵����ǰ֡��������Ч
		distance[iNumOfFrames-1] = 80000;
		points.clear();
		//cout<<"�޹ؼ���"<<endl;
	}
	//��������
	keyPoints.clear();

	if (10 == iNumOfFrames)//��10֡ʱ��ȷ����׼��
	{
		double num;//�������
		int sIndex;//������������ֵ
		int k;//֡������
		int numOfFrames = 0;
		for(k = 0;k<10;k++)//Ѱ�ҵ�һ���йؼ����֡
		{
			num = distance[k];
			sIndex = k;
			if(1 == index[k])
			{
				break;
			}
		}
		for(int i = k;i<10;i++)//�����йؼ����֡������
		{
			if(1 == index[k])
			{
				numOfFrames++;
			}
		}
		
		if(10 == k || numOfFrames<=5)//���10֡��û�йؼ�������йؼ����֡��С�ڵ���5
		{
			pPosOfBase = pPosOfBase_tmp;
			WRONG = -1;//������Ϣ�����ٵ�ƥ���
			//cout<<"WRONG = -1;//������Ϣ�����ٵ�ƥ���"<<endl;
		}
		else
		{
			for(int i = k;i<10;i++)
			{
				if(distance[i] < num && index[i]==1)
				{
					num = distance[i];
					sIndex = i;
				}
			}
			if(num<=100)
			{
				pPosOfBase = vPoints[sIndex][0];
			}
			else
			{
				pPosOfBase = pPosOfBase_tmp;
			}
		}
		cout<<"pPosOfBase.x"<<pPosOfBase.x<<" pPosOfBase.y "<<pPosOfBase.y<<endl;
		/*Ѱ�ұ�������߽�*/
		/*
		edgeX = pPosOfBase.x;
		for(int a = 0;a < pPosOfBase.x;a++)
		{
			int bre = 0;//����ѭ��
			for(int b = 0;b <pPosOfBase.y;b++ )
			{
				if(255 == curImage_binary.ptr<uchar>(b)[a])
				{
					edgeX = a;
					bre = 1;
					break;
				}
			}
			if(1 == bre)
			{
				break;
			}
			if()
			{
				WRONG = -2;//������Ϣ����ⲻ��������
			}
		}*/
	}
	return 0;
}

int lavaDetector::getPosOfDetect()//11-21֡ȷ�����޼��㡢�����ߡ����Ҽ���
{
	//double t1=(double)getTickCount();
	const Mat& binary = curImage_binary;
	/*����Ѱ��������޼��λ�ã�11-21֡�ڶ�ִ�У�21֡����ȷ����*/
	const int baseX = pPosOfBase.x;//��׼���x
	const int baseY = pPosOfBase.y;//��׼���y
	vector<int> vDistance;//��¼���¾���
	vector<int> vX;

	/*Ѱ�ұ�������߽�*/
	for(int a = 0;a < pPosOfBase.x;a++)
	{
		int bre = 0;//����ѭ��
		for(int b = 0;b <pPosOfBase.y;b++ )
		{
			if(255 == curImage_binary.ptr<uchar>(b)[a])
			{
				edgeX = a;
				bre = 1;
				break;
			}
		}
		if(1 == bre)
		{
			break;
		}
		if(a == (pPosOfBase.x - 1))
		{
			edgeX = pPosOfBase.x;
			WRONG = -2;//������Ϣ����ⲻ��������
			//cout<<"WRONG = -2;//������Ϣ����ⲻ��������"<<endl;
		}
	}
	
	/*��ʼ����������*/
	for(int x = baseX;x >= edgeX; x--)//�ӻ�׼�����������������
	{
		vX.push_back(x);
		if(binary.ptr<uchar>(baseY)[x] == 255)//�����ǰ��ʼ����Ϊ255��������������
		{
			vDistance.push_back(0);//�ѵ�ǰ������Ϊ���ܵ���Сֵ0
		}
		else//�����ǰ��ʼ����Ϊ0����ʼ���ϱ���
		{
			for(int y = baseY; y >= 0; y--)//��ǰ�����ϱ���
			{
				if(binary.ptr<uchar>(y)[x] == 255 && binary.ptr<uchar>(y+1)[x] == 0)//����õ����ȸպ���0��Ϊ255����ֹͣ����
				{
					vDistance.push_back(baseY - y);//��¼����
					break;	
				}
				if(0 == y)
				{
					vDistance.push_back(0);//���ϱ�������ˣ���Ȼû������������������պ���0��Ϊ255�����������Ϊ0
				}
			}	
		}
	}
	//Ѱ��vD�е�������
	vector<int>::iterator biggest = max_element(begin(vDistance), end(vDistance));//������
	iBiggestDistance[iNumOfFrames - 11] = *biggest;//����������
	iCorresX[iNumOfFrames - 11] = vX[std::distance(std::begin(vDistance), biggest)];//�����������Ӧ��x����
	//����
	vDistance.clear();
	vX.clear();
	/*ȷ�����޼��㡢�����ߡ����Ҽ���*/
	if(21 == iNumOfFrames)
	{
		/*���˼���*/
		int iBiggestDistance_sorted[11] = {0};
		memcpy(iBiggestDistance_sorted,iBiggestDistance,11*sizeof(int));//����һ����������
		select_sort(iBiggestDistance_sorted, 11);//�Կ��������������
		int noZero;//iBiggestDistance_sorted��������Ԫ������
		for(int k = 0;k<11;k++)
		{
			noZero = k;
			if(iBiggestDistance_sorted[k] == 0)
			{
				break;
			}
			 
		}
		if(noZero <= 2)//���û�����㹻��֡�����ҵ����ʵľ���
		{
			WRONG = -3;//������Ϣ��û�����㹻��֡�����ҵ����ʵľ���
			//cout<<"WRONG = -3;//������Ϣ��û�����㹻��֡�����ҵ����ʵľ���"<<endl;
		}
		for(int i = 0;i<11;i++)
		{
			if(iBiggestDistance[i] == iBiggestDistance_sorted[noZero/2])//��ȡ������λ����Ӧ��index����Ӧ��x����
			{
				pPosThickDetect.x = iCorresX[i];//���˼��������
				pPosThickDetect.y = baseY - iBiggestDistance_sorted[noZero/2];
				cout<<"pPosThickDetect.x "<<pPosThickDetect.x<<"pPosThickDetect.y "<<pPosThickDetect.y <<endl;
				iRadiusOfCircle_p = baseX - iCorresX[i];//��а뾶��������
				dScale = ((double)iRadiusOfCircle_mm) /((double)iRadiusOfCircle_p);//���������
				cout<<"dScale"<<dScale<<endl;
				pPosStreamDetect.x = baseX;//���ɼ����x�������׼��x������ͬ
				pPosStreamDetect.y = pPosThickDetect.y - iBiggestDistance_sorted[noZero/2];//���ɼ����y������ڳ��˼����y�����ȥ����ݰ뾶
				cout<<"pPosStreamDetect.x "<<pPosStreamDetect.x<<"pPosStreamDetect.y "<<pPosStreamDetect.y <<endl;
				break;
			}
		}
	}
	//double t2=(double)getTickCount();
	//double time = (t2-t1)/getTickFrequency();
	//cout<<time<<endl;
	return 0;
}

int lavaDetector::getPosOfDetect2()//ÿ֡ȷ�����޼��㡢�����ߡ����Ҽ���
{
	
	double t1=(double)getTickCount();
	const Mat& binary = curImage_binary;
	/*����Ѱ��������޼��λ�ã���*/
	const int baseX = pPosOfBase.x;//��׼���x
	const int baseY = pPosOfBase.y;//��׼���y
	vector<int> vDistance;//��¼���¾���
	vector<int> vX;
	for(int x = baseX;x >= edgeX; x--)//�ӻ�׼�����������������
	{
		vX.push_back(x);
		if(binary.ptr<uchar>(baseY)[x] == 255)//�����ǰ��ʼ����Ϊ255��������������
		{
			vDistance.push_back(0);//�ѵ�ǰ������Ϊ���ܵ���Сֵ0
		}
		else//�����ǰ��ʼ����Ϊ0����ʼ���ϱ���
		{
			for(int y = baseY; y >= 0; y--)//��ǰ�����ϱ���
			{
				if(binary.ptr<uchar>(y)[x] == 255 && binary.ptr<uchar>(y+1)[x] == 0)//����õ����ȸպ���0��Ϊ255����ֹͣ����
				{
					vDistance.push_back(baseY - y);//��¼����
					break;	
				}
				if(0 == y)
				{
					vDistance.push_back(0);//���ϱ�������ˣ���Ȼû������������������պ���0��Ϊ255�����������Ϊ0
				}
			}	
		}
	}
	//Ѱ��vD�е�������
	vector<int>::iterator biggest = max_element(begin(vDistance), end(vDistance));//������
	int biggestY = *biggest;//����������
	int biggestX = vX[std::distance(std::begin(vDistance), biggest)];//�����������Ӧ��x����
	//����
	vDistance.clear();
	vX.clear();
	/*ȷ�����޼��㡢�����ߡ����Ҽ���*/

	pPosThickDetect.x = biggestX;//���˼��������
	pPosThickDetect.y = baseY - biggestY;

	iRadiusOfCircle_p = baseX - biggestX;//��а뾶��������
	dScale = iRadiusOfCircle_mm /((double)iRadiusOfCircle_p);//���������

	pPosStreamDetect.x = baseX;//���ɼ����x�������׼��x������ͬ
	pPosStreamDetect.y = pPosThickDetect.y - biggestY;//���ɼ����y������ڳ��˼����y�����ȥ����ݰ뾶


	double t2=(double)getTickCount();
	double time = (t2-t1)/getTickFrequency();
	cout<<time<<endl;
	return 0;
}

int lavaDetector::plotPointTarget(Mat& src_color/*�����Ƶ�ͼ��*/,Point pointForPlot,const Scalar& color/*���Ƶ���ɫ*/,int modeOfPlot/*���Ʒ�ʽ*/)//��ͼ�л��Ƶ��target
{
	/*modeOfPlot���Ʒ�ʽ��
	0:target + ����
	1:target+ԲȦ
	2:ֱ�ӻ���ԲȦ
	*/
	if(0 == modeOfPlot)//target + ����
	{
		Point pt1[4] = {pointForPlot,pointForPlot,pointForPlot,pointForPlot};//target����׼���ߵ���ʼ��
		Point pt2[4] = {pointForPlot,pointForPlot,pointForPlot,pointForPlot};//target����׼���ߵĽ�����
		Point pt3[4] = {pointForPlot,pointForPlot,pointForPlot,pointForPlot};//�����Ľǵ�

		//�������ƫ����
		pt1[0].y -= 30; pt1[1].x += 30;	pt1[2].y += 30;	pt1[3].x -= 30;
		pt2[0].y -= 70;	pt2[1].x += 70;	pt2[2].y += 70;	pt2[3].x -= 70;
		pt3[0].x -= 80; pt3[0].y -= 80;	pt3[1].x += 80; pt3[1].y -= 80;	pt3[2].x += 80; pt3[2].y += 80;	pt3[3].x -= 80; pt3[3].y += 80;	
		Point pt4[4] = {pt3[0],pt3[1],pt3[2],pt3[3]};//�����Ľ�����ʼ��
		Point pt5[4] = {pt3[0],pt3[1],pt3[2],pt3[3]};//�����Ľ�����ʼ��
		pt4[0].y =pt3[0].y + 40; pt4[1].x = pt3[1].x - 40; pt4[2].y = pt3[2].y - 40; pt4[3].x = pt3[3].x + 40;
		pt5[0].x =pt3[0].x + 40; pt5[1].y = pt3[1].y + 40; pt5[2].x = pt3[2].x - 40; pt5[3].y = pt3[3].y - 40;
		//����
		for(int i = 0;i<4;i++)
		{
			line(src_color,pt1[i],pt2[i],color,4);//����׼����
			line(src_color,pt3[i],pt4[i],color,4);//���Ʊ߿���
			line(src_color,pt3[i],pt5[i],color,4);//���Ʊ߿���
		}
		//circle(src_color,pointForPlot,5,color,4);//��������Բ
	}
	else if(1 == modeOfPlot)//target+ԲȦ
	{
		Point pt1[4] = {pointForPlot,pointForPlot,pointForPlot,pointForPlot};//target����׼���ߵ���ʼ��
		Point pt2[4] = {pointForPlot,pointForPlot,pointForPlot,pointForPlot};//target����׼���ߵĽ�����
		//�������ƫ����
		pt1[0].y -= 20;	pt1[1].x += 20;	pt1[2].y += 20;	pt1[3].x -= 20;
		pt2[0].y -= 70;	pt2[1].x += 70;	pt2[2].y += 70;	pt2[3].x -= 70;
		for(int i = 0;i<4;i++)
		{
			line(src_color,pt1[i],pt2[i],color,4);//����׼����
		}
		circle(src_color,pointForPlot,5,color,4);//��������Բ
	}
	else if(2 == modeOfPlot)//ֱ�ӻ���ԲȦ
	{
			circle(src_color,pointForPlot,5,color,4);
	}
	return 0;
}

int lavaDetector::plotLine(Mat& src_color,Point pointForPlot,const Scalar& color,int modeOfPlot)//��ͼ�л���ֱ��
{
	/*modeOfPlot���Ʒ�ʽ��
	0:����
	1:����
	*/
	if(0 == modeOfPlot)//����
	{
		Point p1 = pointForPlot;
		p1.y = 0;
		Point p2 = pointForPlot;
		p2.y = sizeOfCur.height;
		line(src_color,p1,p2,color,1,CV_AA );
	}
	else if (1 == modeOfPlot)//����
	{
		Point p1 = pointForPlot;
		p1.x = 0;
		Point p2 = pointForPlot;
		p2.x = sizeOfCur.width;
		line(src_color,p1,p2,color,1,CV_AA );
	}
	return 0 ;
}
int lavaDetector::detectThickOfWool()//������޳��˺�ȣ����ؼ�mmΪ��λ��
{
	/*��س�Ա������
		double dScale;//������
		int iThickOfWool_p;//���޵ĺ�ȣ������ؼƣ�
		double dThickOfWool_mm;//���޵ĺ�ȣ�ͨ������������Ϊ��mm�ƣ�
		Point pPosOfBase;//��׼�����������
		Point pPosThickDetect;//���˼��������
		Mat curImage_binary;//��ǰ��ֵͼԭʼ֡
		Point pPosThickPlot;//���˻��Ƶ������
	*/
	int iThickOfWool_p_tmp = iThickOfWool_p;//����һ֡�ĺ���ݴ�
	iThickOfWool_p = 0;//��ʼ����ǰ֡�ĺ��
	Point pStartPoint;//������ʼ��
	pStartPoint.x = pPosThickDetect.x;
	pStartPoint.y = pPosOfBase.y;
	//�ֲ������ĳ�ʼ��
	int highY = 0;//�������Ϸ���y����
	int lowY = 0;//�������·���y����
	int pre = 0;//ǰһ֡�����������
	int current = 0;//��ǰ֡�����������
	int i =0;//y�������
	//���¶��ϵر�������ȡ���ɺ���Լ������м��
	for(i = pStartPoint.y; i >= 0;i--)
	{
		current = curImage_binary.ptr<uchar>(i)[pStartPoint.x];
		if(255 == current)
		{
			iThickOfWool_p++;
		}
		if(current==255 && pre ==0)
		{
			lowY = i;
			break;
		}
		pre = current;
	}
	for(int j = i; j >= 0;j--)
	{
		current = curImage_binary.ptr<uchar>(j)[pStartPoint.x];
		if(255 == current)
		{
			iThickOfWool_p++;
		}
		if(current==0 && pre ==255)
		{
			highY = j;
			break;
		}
		pre = current;
	}
	//�쳣ֵ��0�ȣ�������о��ȴ���ͬʱ�����˻��Ƶ������ά�ֲ���
	if(((double)iThickOfWool_p) <= (0.3 * (double)iThickOfWool_p_tmp))
	{
		iThickOfWool_p = iThickOfWool_p_tmp * 0.8;
	}
	else//���û���쳣ֵ����ֱ�ӻ�ȡ��ǰ֡�ĳ��˺�Ȳ����ͳ��˻��Ƶ������
	{
		//ȷ�����˻��Ƶ������
		pPosThickPlot.x = pStartPoint.x;
		pPosThickPlot.y = (int)((lowY + highY) * 0.5);
	}

	return 0;
}

int lavaDetector::detectWidthOfStream()//����������ɿ�ȣ����ؼ�mmΪ��λ��������������������
{
	/*��س�Ա������
		double dScale;//������
		int iWidthOfStream_p;//�������ɵĿ�ȣ������ؼƣ�
	    double dWidthOfStream_mm;//�������ɵĿ�ȣ�ͨ������������Ϊ��mm�ƣ�
		Point pPosStreamDetect;//���ɼ��������
		Mat curImage_binary;//��ǰ��ֵͼԭʼ֡
		Point pPosStreamDetect;//���ɼ��������
		Point pPosStreamPlot;//���ɻ��Ƶ������
	*/
	int iWidthOfStream_p_tmp = iWidthOfStream_p;//����һ֡�ĺ���ݴ�
	iWidthOfStream_p = 0;//��ʼ����ǰ֡�ĺ��
	//�ֲ������ĳ�ʼ��
	int rightX = 0;//����������x����
	int leftX = 0;//�������Ҳ��x����
	int pre = 0;//ǰһ֡�����������
	int current = 0;//��ǰ֡�����������
	int i =0;//x�������
	//������ҵر�������ȡ���ɿ���Լ������м��
	for(i = pPosStreamDetect.x; i < sizeOfCur.width;i++)
	{
		current = curImage_binary.ptr<uchar>(pPosStreamDetect.y)[i];
		if(255 == current)
		{
			iWidthOfStream_p++;
		}
		if(current==255 && pre ==0)
		{
			leftX = i;
			break;
		}
		pre = current;
	}
	for(int j = i; j < sizeOfCur.width;j++)
	{
		current = curImage_binary.ptr<uchar>(pPosStreamDetect.y)[j];
		if(255 == current)
		{
			iWidthOfStream_p++;
		}
		if(current==0 && pre ==255)
		{
			rightX = j;
			break;
		}
		pre = current;
	}
	//�쳣ֵ��0�ȣ�������о��ȴ���ͬʱ�����˻��Ƶ������ά�ֲ���
	if(((double)iWidthOfStream_p) <= (0.3 * (double)iWidthOfStream_p_tmp))
	{
		iWidthOfStream_p = iWidthOfStream_p * 0.8;
	}
	else//���û���쳣ֵ����ֱ�ӻ�ȡ��ǰ֡�ĳ��˺�Ȳ����ͳ��˻��Ƶ������
	{
		//ȷ�����˻��Ƶ������
		pPosStreamPlot.y = pPosStreamDetect.y;
		pPosStreamPlot.x = (int)((rightX + leftX) * 0.5);
	}

	//��������x����
	iPosOfDrop_p = pPosStreamPlot.x;
	iRePosOfDrop_p = iPosOfDrop_p - iPosOfZero_p;//��ȡ���ƫ���������أ�
	
	return 0;
}

void lavaDetector::select_sort(int array[], int n)//ѡ������
{
	for(int i=0;i<n;i++)
	{
		for(int j=i+1;j<n;j++)
		{
			if(array[j]>array[i])
			{
				int t;
				t=array[i];
				array[i]=array[j];
				array[j]=t;
			}
		}
	}
}