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
	lavaDetector(int period,double thresh, int posOfZero_p,int mode);//�������--��ֵ���ָ���ֵ--���x����--���ģʽ
	~lavaDetector();

	//ÿ֡�ĳ�ʼ������
	int readCurrentFrame(Mat src,int mode/*�趨��ǰ����ģʽ*/);//��ȡ��ǰ֡
	//����ģʽ�µļ��
	int staticImageDetect();//��ͷ��̬���ݼ�� mode 0
	int dynamicImageDetect();//��ͷ��̬���ݼ�� mode 1
	int imageDetect();//���޼�� mode 2: 
	//ȫ�ֲ�������
	int setPosOfZero_p();//�������x����
	//��ø���������ע�ⲻͬģʽ�µ������
	int getBinaryImage(Mat& imagetoshow_binary);
	int getImageToShow(Mat& imagetoshow);//���Ʋ���ȡ����ʾ�ļ����,ע�ⲻͬģʽ����ʾ��ͼ��Ҳ��ͬ��
	int getParameters_p(int& thickOfWool_p/*���˺��*/, int& widthOfStream_p/*���ɿ��*/,int& rePosOfDrop_p/*���ƫ����*/);//��ȡ�����������������Ϊ��λ��
	int getParameters_mm(double& thickOfWool_mm,double& widthOfStream_mm, double& rePosOfDrop_mm);//��ȡ�������������mmΪ��λ��
	int getWRONG_2(int& wrong);
	int getWRONG_1(int& wrong);
private://˽�г�Ա����
	int colorToBinary(Mat src_color,Mat& dst_binary,double dThreshold);//�������ԭʼ��ɫ֡ת��Ϊ��ֵ֡
	int selectContours(Mat& binary);//�Զ�ֵͼ�еĶ���������ɸѡ��ȥ�������㣬����׶�
	//ȷ����׼��
	int getPosOfBase();//*����ȷ����׼���λ��1����֡ƥ�䣩
	int getPosOfBase2_collect();//*����ȷ����׼���λ��2���ռ�������
	int getPosOfBase2();//����ȷ����׼���λ��2������ƥ�䣩
	//ȷ�����޼��㡢�����ߡ����Ҽ���
	int getPosOfDetect();//11-21֡ȷ�����޼��㡢�����ߡ����Ҽ���
	int getPosOfDetect2();//*ÿ֡��ȷ�����޼��㡢�����ߡ����Ҽ���
	void select_sort(int array[], int n);//ѡ������
	//������޺�ȣ����ؼ�mmΪ��λ��
	int detectThickOfWool();
	//����������ɿ�ȣ����ؼ�mmΪ��λ��������������������
	int detectWidthOfStream();
	//������������
	int plotPointTarget(Mat& src_color,Point pointForPlot,const Scalar& color,int modeOfPlot);//��ͼ�л��Ƶ��target
	int plotLine(Mat& src_color,Point pointForPlot,const Scalar& color,int modeOfPlot);//��ͼ�л���ֱ��
private://˽�г�Ա����
	/*����ȫ�ֵĲ���*/
	int THEMODE;//���ģʽ��0���Զ���� 1������Ԥ���õĲ������м��
	/*�������������*/
	int iWidthOfStream_p;//�������ɵĿ�ȣ������ؼƣ�
	double dWidthOfStream_mm;//�������ɵĿ�ȣ�ͨ������������Ϊ��mm�ƣ�

	int iThickOfWool_p;//���޵ĺ�ȣ������ؼƣ�
	double dThickOfWool_mm;//���޵ĺ�ȣ�ͨ������������Ϊ��mm�ƣ�

	int iPosOfZero_p;//��������x����
	int iPosOfDrop_p;//��������λ�õ�����x����
	int iRePosOfDrop_p;//�����ƫ�����������㣩��������
	double dRePosOfDrop_mm;//�����ƫ�����������㣩��mm����ͨ������������Ϊ��mm�ƣ�

	Point pPosOfBase;//��׼�����������
	Point pPosOfBase_tmp;//��׼�����ʱ��������(���Ҳ����ؼ���ʱ���ô˻�׼��)
	double dScale;//������
	
	/*��ǰ֡������*/
	Mat curImage_color;//��ǰ��ɫԭʼ֡
	Mat curImage_binary;//��ǰ��ֵͼԭʼ֡
	Mat imageToShow_color;//���մ���ʾ��ͼ��
	Size sizeOfCur;//��ǰͼ��ߴ�
	int WRONG;//������
	/*
	WRONG == -4;//���󣺵�ǰ֡����Ŀ��(ȫ��)

	*/
	/*Ԥ����ȫ�ֲ���*/
	int iNumOfFrames;//��ǰ֡���*
	const int iPeriod;//������ڣ���֡��Ϊ��λ*
	int iWorkmode;//��ǰ�Ĺ���ģʽ��0����̬��⣻1����̬��⣻2�����޼��*
	double dThreshold;//��ͼתΪ��ֵͼʱ����ֵ*
	/*�ؼ���ƥ��ʱ�ı���*/
	vector<Mat> vDescriptors;//ÿ֡��������
	vector<vector<KeyPoint>> vKeyPoints;//ÿ֡��ORB�ؼ���*
	vector<vector<Point>> vPoints;//ƥ���ÿ֡�ı�ѡ��*
	vector<vector<DMatch>> vMatches;//����2
	double distance[10];//ceshi3
	int index[10];
	/*ȷ�����˼���ı���*/
	int edgeX;//�����������߽�
	int iBiggestDistance[11];//��¼11-15֡�������߶����ֵ
	int iCorresX[11];//11-15֡������ȡ�ø߶����ֵ�¶�Ӧ��x����
	Point pPosThickDetect;//���˼��������
	Point pPosThickPlot;//���˻��Ƶ������
	int iRadiusOfCircle_p;//��а뾶��������
    const double iRadiusOfCircle_mm;//��а뾶�ĺ�����
	Point pPosStreamDetect;//���ɼ��������
	Point pPosStreamPlot;//���ɻ��Ƶ������
	/*��̬��Ƶ���ı���*/
	vector<long> vArea;
	vector<double> vNumOfKeyPoints;
	const int iPeriodOfMod1;//��̬��Ƶ������ڣ���֡��Ϊ��λ
	int iNumOfFramesOfMod1;//mod1��ǰ֡���
	int WRONG_MOD1;//������
	/*�����㣺A�������±�Ե�㣬B�������ϱ�Ե�㣬C���������㣬D���������Ƶ�*/
	Point pPosThickDown;//A
	Point pPosThickUp;//B
	Point pPosStreamDropDtect;//C
	Point pPosStreamDropPlot;//D
};

#endif