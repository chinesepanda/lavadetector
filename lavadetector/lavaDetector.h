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
	//构造析构函数
	lavaDetector();
	lavaDetector(int period,double thresh);
	~lavaDetector();

	//每帧的初始化步骤
	int readCurrentFrame(Mat src,int mode/*设定当前工作模式*/);//读取当前帧
	//三种模式下的检测
	int staticImageDetect();//镜头静态内容检测 mode 0
	int dynamicImageDetect();//镜头动态内容检测 mode 1
	int imageDetect();//岩棉检测 mode 2: 
	//全局操作函数
	int resetAll();//重置内部参数
	//获得各项检测结果（注意不同模式下的情况）
	int getBinaryImage(Mat& imagetoshow_binary);
	int getImageToShow(Mat& imagetoshow);//绘制并获取待显示的检测结果,注意不同模式下显示的图像也不同！

private://私有成员函数
	int colorToBinary(Mat src_color,Mat& dst_binary,double dThreshold);//把输入的原始彩色帧转换为二值帧
	int selectContours(Mat& binary);//对二值图中的多轮廓进行筛选，去除噪声点，并填补孔洞
	//确定基准点
	int getPosOfBase();//用于确定基准点的位置1（跨帧匹配）
	int getPosOfBase2_collect();//用于确定基准点的位置2（收集特征）
	int getPosOfBase2();//用于确定基准点的位置2（特征匹配）
	//确定岩棉检测点、比例尺、熔岩检测点
	int getPosOfDetect();//确定岩棉检测点、比例尺、熔岩检测点
private://私有成员变量
	int iWidthOfStream_p;//下落流股的宽度（以像素计）
	double dWidthOfStream_mm;//下落流股的宽度（通过比例尺折算为以mm计）
	double dScale;//比例尺
	int iThickOfWool_p;//岩棉的厚度（以像素计）
	double dThickOfWool_mm;//岩棉的厚度（通过比例尺折算为以mm计）
	int iPosOfDrop[2];//下落点绝对位置的像素坐标
	int iRePosOfDrop[2];//下落点相对位置（相对零点）的像素坐标
	Point pPosOfBase;//基准点的像素坐标
	int iPosOfZero[2];//零点的像素坐标

	Mat curImage_color;//当前彩色原始帧
	Mat curImage_binary;//当前二值图原始帧
	Mat imageToShow_color;//最终待显示的图像
	Size sizeOfCur;//当前图像尺寸

	int iNumOfFrames;//当前帧序号*
	const int iPeriod;//检测周期，以帧数为单位*
	int iWorkmode;//当前的工作模式：0：静态检测；1：动态检测；2：岩棉检测*
	double dThreshold;//彩图转为二值图时的阈值*
	//关键点匹配时的变量
	vector<Mat> vDescriptors;//*
	vector<vector<KeyPoint>> vKeyPoints;//每帧的ORB关键点*
	vector<vector<Point>> vPoints;//匹配后每帧的备选点*
	vector<vector<DMatch>> vMatches;//测试2
	double distance[10];//ceshi3
};

#endif