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
	lavaDetector(int period,double thresh, int posOfZero_p,int mode);//检测周期--二值化分割阈值--零点x坐标--检测模式
	~lavaDetector();

	//每帧的初始化步骤
	int readCurrentFrame(Mat src,int mode/*设定当前工作模式*/);//读取当前帧
	//三种模式下的检测
	int staticImageDetect();//镜头静态内容检测 mode 0
	int dynamicImageDetect();//镜头动态内容检测 mode 1
	int imageDetect();//岩棉检测 mode 2: 
	//全局操作函数
	int setPosOfZero_p();//设置零点x坐标
	//获得各项检测结果（注意不同模式下的情况）
	int getBinaryImage(Mat& imagetoshow_binary);
	int getImageToShow(Mat& imagetoshow);//绘制并获取待显示的检测结果,注意不同模式下显示的图像也不同！
	int getParameters_p(int& thickOfWool_p/*成纤厚度*/, int& widthOfStream_p/*流股宽度*/,int& rePosOfDrop_p/*落点偏移量*/);//获取三项待检测参数（像素为单位）
	int getParameters_mm(double& thickOfWool_mm,double& widthOfStream_mm, double& rePosOfDrop_mm);//获取三项待检测参数（mm为单位）
	int getWRONG_2(int& wrong);
	int getWRONG_1(int& wrong);
private://私有成员函数
	int colorToBinary(Mat src_color,Mat& dst_binary,double dThreshold);//把输入的原始彩色帧转换为二值帧
	int selectContours(Mat& binary);//对二值图中的多轮廓进行筛选，去除噪声点，并填补孔洞
	//确定基准点
	int getPosOfBase();//*用于确定基准点的位置1（跨帧匹配）
	int getPosOfBase2_collect();//*用于确定基准点的位置2（收集特征）
	int getPosOfBase2();//用于确定基准点的位置2（特征匹配）
	//确定岩棉检测点、比例尺、熔岩检测点
	int getPosOfDetect();//11-21帧确定岩棉检测点、比例尺、熔岩检测点
	int getPosOfDetect2();//*每帧都确定岩棉检测点、比例尺、熔岩检测点
	void select_sort(int array[], int n);//选择排序
	//检测岩棉厚度（像素及mm为单位）
	int detectThickOfWool();
	//检测熔岩流股宽度（像素及mm为单位）、检测熔岩流股下落点
	int detectWidthOfStream();
	//绘制特殊点程序
	int plotPointTarget(Mat& src_color,Point pointForPlot,const Scalar& color,int modeOfPlot);//在图中绘制点的target
	int plotLine(Mat& src_color,Point pointForPlot,const Scalar& color,int modeOfPlot);//在图中绘制直线
private://私有成员变量
	/*控制全局的参数*/
	int THEMODE;//检测模式：0：自动检测 1：基于预设置的参数进行检测
	/*待检测的诸项参数*/
	int iWidthOfStream_p;//下落流股的宽度（以像素计）
	double dWidthOfStream_mm;//下落流股的宽度（通过比例尺折算为以mm计）

	int iThickOfWool_p;//岩棉的厚度（以像素计）
	double dThickOfWool_mm;//岩棉的厚度（通过比例尺折算为以mm计）

	int iPosOfZero_p;//零点的像素x坐标
	int iPosOfDrop_p;//下落点绝对位置的像素x坐标
	int iRePosOfDrop_p;//下落点偏移量（相对零点）的像素数
	double dRePosOfDrop_mm;//下落点偏移量（相对零点）的mm数（通过比例尺折算为以mm计）

	Point pPosOfBase;//基准点的像素坐标
	Point pPosOfBase_tmp;//基准点的临时像素坐标(当找不到关键点时采用此基准点)
	double dScale;//比例尺
	
	/*当前帧内数据*/
	Mat curImage_color;//当前彩色原始帧
	Mat curImage_binary;//当前二值图原始帧
	Mat imageToShow_color;//最终待显示的图像
	Size sizeOfCur;//当前图像尺寸
	int WRONG;//检测错误
	/*
	WRONG == -4;//错误：当前帧内无目标(全黑)

	*/
	/*预设置全局参数*/
	int iNumOfFrames;//当前帧序号*
	const int iPeriod;//检测周期，以帧数为单位*
	int iWorkmode;//当前的工作模式：0：静态检测；1：动态检测；2：岩棉检测*
	double dThreshold;//彩图转为二值图时的阈值*
	/*关键点匹配时的变量*/
	vector<Mat> vDescriptors;//每帧的描述子
	vector<vector<KeyPoint>> vKeyPoints;//每帧的ORB关键点*
	vector<vector<Point>> vPoints;//匹配后每帧的备选点*
	vector<vector<DMatch>> vMatches;//测试2
	double distance[10];//ceshi3
	int index[10];
	/*确定成纤监测点的变量*/
	int edgeX;//向左遍历的左边界
	int iBiggestDistance[11];//记录11-15帧成纤区高度最大值
	int iCorresX[11];//11-15帧成纤区取得高度最大值下对应的x坐标
	Point pPosThickDetect;//成纤检测点的坐标
	Point pPosThickPlot;//成纤绘制点的坐标
	int iRadiusOfCircle_p;//轴承半径的像素数
    const double iRadiusOfCircle_mm;//轴承半径的毫米数
	Point pPosStreamDetect;//流股检测点的坐标
	Point pPosStreamPlot;//流股绘制点的坐标
	/*动态视频检测的变量*/
	vector<long> vArea;
	vector<double> vNumOfKeyPoints;
	const int iPeriodOfMod1;//动态视频检测周期，以帧数为单位
	int iNumOfFramesOfMod1;//mod1当前帧序号
	int WRONG_MOD1;//检测错误
	/*待描绘点：A成纤区下边缘点，B成纤区上边缘点，C流股落点检测点，D流股落点绘制点*/
	Point pPosThickDown;//A
	Point pPosThickUp;//B
	Point pPosStreamDropDtect;//C
	Point pPosStreamDropPlot;//D
};

#endif