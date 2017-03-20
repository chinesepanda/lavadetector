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
lavaDetector::lavaDetector():iPeriod(500)/*�������*/,dThreshold(25)/*�ָ����ֵ*/,iRadiusOfCircle_mm(400)/*���Ԥ���õİ뾶*/
{
	iNumOfFrames = 0;
	iThickOfWool_p = 0;
}
lavaDetector::lavaDetector(int period, double thresh) : iPeriod(period), dThreshold(thresh),iRadiusOfCircle_mm(400)
{
	iNumOfFrames = 0;
	iThickOfWool_p = 0;
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
		if (1 == iNumOfFrames)//��һ֡��ز�������
		{
			vDescriptors.clear();
			vKeyPoints.clear();//ÿ֡��ORB�ؼ���*
			vPoints.clear();//ƥ���ÿ֡�ı�ѡ��**/
			pPosOfBase.x = 0;
			pPosOfBase.y = 0;
		}
		getPosOfBase2();//ȷ����ǰ���ڵĹؼ���
	}
	else if (iNumOfFrames >= 11 && iNumOfFrames <= 21)//11-15֡ȷ�����޼��㡢���ɼ��㡢������
	{
		getPosOfDetect();//ȷ�����޼��㡢�����ߡ����Ҽ���
		
	}
	else//��22֡��ʼ������������� 
	{
		detectThickOfWool();//�����˺��
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
			plotPointTarget(imageToShow_color,pPosOfBase,Scalar(0,0,255),1);
		}
		if(iNumOfFrames > 21)
		{	
			plotPointTarget(imageToShow_color,pPosThickDetect,Scalar(255,0,0),2);
			plotPointTarget(imageToShow_color,pPosStreamDetect,Scalar(255,0,0),2);
			//���Ƽ���� pPosThickPlot
			plotPointTarget(imageToShow_color,pPosThickPlot,Scalar(255,255,255),0);//���˻��Ƶ������
			cout<<iThickOfWool_p<<endl;
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

	//ʵ����һ��ƥ����
	BruteForceMatcher <L2<float>> matcher;
	vector<DMatch> matches;//ƥ����
	matcher.match(theNormal, descriptors, matches);//Ѱ�ҵ�ǰ֡��ƥ���������

	//�ѹؼ���ת��Ϊ��ͨ��,������vector<vector<Point>> vPoints����Ϊ��ѡ�㣨Ϊ��֤���ַ����ļ��ݣ���������Ƕ�ף�
	vector<Point> points;//�ؼ���ת��Ϊ��ͨ��
	points.push_back(keyPoints[matches[0].trainIdx].pt);
	distance[iNumOfFrames-1] = matches[0].distance;//��ǰ֡��ƥ�����������Ŀ������������
	points[0].y += (int)(rate*sizeOfCur.height);//�ָ�Ϊԭͼ���е������
	vPoints.push_back(points);//��ƥ�����������Ա��������
	//��������
	points.clear();
	matcher.clear();
	matches.clear();
	keyPoints.clear();

	if (10 == iNumOfFrames)//��10֡ʱ��ȷ����׼��
	{
		int sIndex = 0;//������������ֵ
		int num = distance[0];//�������
		for(int i = 0;i<10;i++)
		{
			if(distance[i] < num)
			{
				num = distance[i];
				sIndex = i;
			}
		}
		pPosOfBase = vPoints[sIndex][0];
		/*Ѱ�ұ�������߽�*/
		for(int a = 0;a < pPosOfBase.x;a++)
		{
			int bre = 0;
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
		}
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
		for(int i = 0;i<11;i++)
		{
			if(iBiggestDistance[i] == iBiggestDistance_sorted[5])//��ȡ������λ����Ӧ��index����Ӧ��x����
			{
				pPosThickDetect.x = iCorresX[i];//���˼��������
				pPosThickDetect.y = baseY - iBiggestDistance_sorted[5];

				iRadiusOfCircle_p = baseX - iCorresX[i];//��а뾶��������
				dScale = iRadiusOfCircle_mm /((double)iRadiusOfCircle_p);//���������

				pPosStreamDetect.x = baseX;//���ɼ����x�������׼��x������ͬ
			    pPosStreamDetect.y = pPosThickDetect.y - iBiggestDistance_sorted[5];//���ɼ����y������ڳ��˼����y�����ȥ����ݰ뾶
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
	const int iSide = 61;//����߳�
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
	int iThickOfWool_p_tmp = iThickOfWool_p;
	iThickOfWool_p = 0;
	Point pStartPoint;//������ʼ��
	pStartPoint.x = pPosThickDetect.x;
	pStartPoint.y = pPosOfBase.y;
	int highY = 0;
	int lowY = 0;
	int pre = 0;
	int current = 0;
	for(int i = pStartPoint.y; i >= 0;i--)
	{
		current = curImage_binary.ptr<uchar>(i)[pStartPoint.x];
		if(255 == current)
		{
			iThickOfWool_p++;
		}
		if(current==255 && pre ==0)
		{
			lowY = i;
		}
		if(current==0 && pre ==255)
		{
			highY = i;
		}
		pre = current;
	}
	//�쳣ֵ��0�ȣ�������о��ȴ���
	if(((double)iThickOfWool_p) <= (0.3 * (double)iThickOfWool_p_tmp))
	{
		iThickOfWool_p = iThickOfWool_p_tmp * 0.9;
	}
	//ȷ�����˻��Ƶ������
	pPosThickPlot.x = pStartPoint.x;
	pPosThickPlot.y = (int)((lowY + highY)*0.5);
	return 0;
}

void lavaDetector::select_sort(int array[], int n)//ѡ������
{
	for(int i=0;i<n;i++)
	{
		for(int j=i+1;j<n;j++)
		{
			if(array[j]<array[i])
			{
				int t;
				t=array[i];
				array[i]=array[j];
				array[j]=t;
			}
		}
	}
}