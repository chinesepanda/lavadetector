#pragma once
#ifndef _LAVADETECTOR_H_
#define _LAVADETECTOR_H_
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;
using namespace cv;

class lavaDetector
{
public:
	//������������
	lavaDetector();
	lavaDetector(int period,double thresh);
	~lavaDetector();

	//ÿ֡�ĳ�ʼ������
	int readCurrentFrame(Mat src,int mode/*�趨��ǰ����ģʽ*/);//��ȡ��ǰ֡
	//����ģʽ�µļ��
	int staticImageDetect();//��ͷ��̬���ݼ�� mode 0
	int dynamicImageDetect();//��ͷ��̬���ݼ�� mode 1
	int imageDetect();//���޼�� mode 2: 
	//ȫ�ֲ�������
	int resetAll();//�����ڲ�����
	//��ø���������ע�ⲻͬģʽ�µ������
	int getBinaryImage(Mat& imagetoshow_binary);
	int getImageToShow(Mat& imagetoshow);//���Ʋ���ȡ����ʾ�ļ����,ע�ⲻͬģʽ����ʾ��ͼ��Ҳ��ͬ��

private://˽�г�Ա����
	int colorToBinary(Mat src_color,Mat& dst_binary,double dThreshold);//�������ԭʼ��ɫ֡ת��Ϊ��ֵ֡
	int selectContours(Mat& binary);//�Զ�ֵͼ�еĶ���������ɸѡ��ȥ�������㣬����׶�
	//ȷ����׼��
	int getPosOfBase();//����ȷ����׼���λ��1����֡ƥ�䣩
	int getPosOfBase2_collect();//����ȷ����׼���λ��2���ռ�������
	int getPosOfBase2();//����ȷ����׼���λ��2������ƥ�䣩
	//ȷ�����޼��㡢�����ߡ����Ҽ���
	int getPosOfDetect();//ȷ�����޼��㡢�����ߡ����Ҽ���
private://˽�г�Ա����
	int iWidthOfStream_p;//�������ɵĿ�ȣ������ؼƣ�
	double dWidthOfStream_mm;//�������ɵĿ�ȣ�ͨ������������Ϊ��mm�ƣ�
	double dScale;//������
	int iThickOfWool_p;//���޵ĺ�ȣ������ؼƣ�
	double dThickOfWool_mm;//���޵ĺ�ȣ�ͨ������������Ϊ��mm�ƣ�
	int iPosOfDrop[2];//��������λ�õ���������
	int iRePosOfDrop[2];//��������λ�ã������㣩����������
	Point pPosOfBase;//��׼�����������
	int iPosOfZero[2];//������������

	Mat curImage_color;//��ǰ��ɫԭʼ֡
	Mat curImage_binary;//��ǰ��ֵͼԭʼ֡
	Mat imageToShow_color;//���մ���ʾ��ͼ��
	Size sizeOfCur;//��ǰͼ��ߴ�

	int iNumOfFrames;//��ǰ֡���*
	const int iPeriod;//������ڣ���֡��Ϊ��λ*
	int iWorkmode;//��ǰ�Ĺ���ģʽ��0����̬��⣻1����̬��⣻2�����޼��*
	double dThreshold;//��ͼתΪ��ֵͼʱ����ֵ*
	//�ؼ���ƥ��ʱ�ı���
	vector<Mat> vDescriptors;//*
	vector<vector<KeyPoint>> vKeyPoints;//ÿ֡��ORB�ؼ���*
	vector<vector<Point>> vPoints;//ƥ���ÿ֡�ı�ѡ��*
	vector<vector<DMatch>> vMatches;//����2
	double distance[10];//ceshi3
};

#endif