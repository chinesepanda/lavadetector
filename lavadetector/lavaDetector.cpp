#include "lavaDetector.h"
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/legacy/legacy.hpp>
using namespace std;
using namespace cv;

//构造与析构函数
lavaDetector::lavaDetector():iPeriod(20),dThreshold(25)
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
int lavaDetector::readCurrentFrame(Mat src, int mode/*设定当前工作模式*/)//读取当前帧
{
	iWorkmode = mode;
	if (iWorkmode != 0 && iWorkmode != 1 && iWorkmode != 2)
	{
		return -1;//工作模式错误
	}
	if (2 == iWorkmode)//如果是模式3，则以iPeriod为周期，对帧数进行计数
	{
		iNumOfFrames++;
		if (iPeriod <= iNumOfFrames)//周期结束，帧数重置
		{
			iNumOfFrames = 1;
		}
	}
	src.copyTo(curImage_color);//获取当前彩色帧
	colorToBinary(curImage_color, curImage_binary, dThreshold);//彩色帧转为二值帧
	selectContours(curImage_binary);//对轮廓大小进行筛选，排除噪声点
	src.copyTo(imageToShow_color);//复制到待显示的帧中
	sizeOfCur = curImage_color.size();//获取尺寸
	return 0;
}
int lavaDetector::staticImageDetect()//镜头静态内容检测 mode 0
{
	return 0;
}

int lavaDetector::dynamicImageDetect()//镜头动态内容检测 mode 1
{
	return 0;
}
int lavaDetector::imageDetect()//岩棉检测 mode 2:
{
	if (iWorkmode != 2)
	{
		return -1;//模式错误
	}

	if (iNumOfFrames >= 1 && iNumOfFrames <= 10)//前10帧关键点定位
	{
		if (1 == iNumOfFrames)//第一帧相关参数置零
		{
			vDescriptors.clear();
			vKeyPoints.clear();//每帧的ORB关键点*
			vPoints.clear();//匹配后每帧的备选点**/
			pPosOfBase.x = 0;
			pPosOfBase.y = 0;
		}
		getPosOfBase2();//确定当前周期的关键点
	}
	else if (iNumOfFrames >= 11 && iNumOfFrames <= 15)//11-15帧确定岩棉检测点、流股监测点、比例尺
	{
		

	}
	else//检测三项参数
	{
		

	}
	return 0;
}
int lavaDetector::resetAll()//重置内部参数,预留接口
{
	return 0;
}
int lavaDetector::getBinaryImage(Mat& imagetoshow_binary)//测试时用于显示当前二值图
{
	curImage_binary.copyTo(imagetoshow_binary);
	return 0;
}
int lavaDetector::getImageToShow(Mat& imagetoshow)//绘制并获取待显示的检测结果,注意不同模式下显示的图像也不同！
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
			circle(imageToShow_color, pPosOfBase, 5, Scalar(255, 0, 0), 3);//绘制关键点
		}
		imageToShow_color.copyTo(imagetoshow);
		
	}
	return 0;
}
//private:
int lavaDetector::colorToBinary(Mat src_color, Mat& dst_binary, double dThreshold)//把输入的原始彩色帧转换为二值帧
{
	Mat gray;
	cvtColor(src_color,gray,CV_BGR2GRAY);
	threshold(gray, dst_binary, dThreshold, 255, THRESH_BINARY);//阈值分割
	return 0;
}

int lavaDetector::selectContours(Mat& binary)//对二值图中的多轮廓进行筛选，去除噪声点，并填补孔洞
{
	vector< vector<Point> > contours;   // 轮廓   
	vector< vector<Point> > filterContours; // 筛选后的轮廓
	vector< Vec4i > hierarchy;    // 轮廓的结构信息 
	contours.clear();
	hierarchy.clear();
	filterContours.clear();

	findContours(binary, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	// 去除伪轮廓 
	for (size_t i = 0; i < contours.size(); i++)
	{
		if (fabs(contourArea(Mat(contours[i]))) > 2000)  //判断区域的阈值
		{
			filterContours.push_back(contours[i]);
		}
	}
	binary.setTo(0);
	drawContours(binary, filterContours, -1, Scalar(255), CV_FILLED); //8, hierarchy);   
	return 0;
}

int lavaDetector::getPosOfBase()//用于确定基准点的位置
{
	/*	vector<Mat> vDescriptors;//*
	vector<vector<KeyPoint>> vKeyPoints;//每帧的ORB关键点*
	vector<vector<Point>> vPoints;//匹配后每帧的备选点**/
	const double rate = 0.4;//去掉上半部分的比例
	Mat grayImage;
	cvtColor(curImage_color, grayImage, CV_BGR2GRAY);
	Mat ROIImage = grayImage(Range((int)(rate*sizeOfCur.height), sizeOfCur.height), Range(0, sizeOfCur.width));
	/*当前帧的特征点检测*/
	OrbFeatureDetector featureDetector;//ORB点检测器
	vector<KeyPoint> keyPoints;
	Mat descriptors;
	featureDetector.detect(ROIImage, keyPoints);//检测特征关键点

	OrbDescriptorExtractor featureExtractor;//ORB点特征提取器
	featureExtractor.compute(ROIImage, keyPoints, descriptors);
	//存入成员变量
	vDescriptors.push_back(descriptors);//每帧的ORB关键点描述子
	vKeyPoints.push_back(keyPoints);//每帧的ORB关键点*

	/*跨帧特征点匹配*/
	if (iNumOfFrames >= 4)
	{
		//实例化一个匹配器
		BruteForceMatcher <L2<float>> matcher;
		vector<DMatch> matches;//匹配结果
		matcher.match(vDescriptors[iNumOfFrames-4], vDescriptors[iNumOfFrames-1], matches);
		//匹配点排序
		nth_element(matches.begin(), matches.begin() + 2, matches.end());
		matches.erase(matches.begin() + 3, matches.end());//擦除
		//把关键点转换为普通点,并存入vector<vector<Point>> vPoints，作为备选点
		vector<Point> points;
		for (int i = 0; i < 3; i++)
		{
			points.push_back(keyPoints[matches[i].trainIdx].pt);
			points[i].y += (int)(rate*sizeOfCur.height);//恢复为原图像中的坐标点
		}
		vPoints.push_back(points);
		//清理
		points.clear();
		matcher.clear();
		matches.clear();
	}
	keyPoints.clear();

	if (10 == iNumOfFrames)//第10帧时，确定基准点
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

int lavaDetector::getPosOfBase2_collect()//用于确定基准点的位置
{
	/*	vector<Mat> vDescriptors;//*
	vector<vector<KeyPoint>> vKeyPoints;//每帧的ORB关键点*
	vector<vector<Point>> vPoints;//匹配后每帧的备选点**/
	const double rate = 0.4;//去掉上半部分的比例
	Mat grayImage;
	cvtColor(curImage_color, grayImage, CV_BGR2GRAY);
	Mat ROIImage = grayImage(Range((int)(rate*sizeOfCur.height), sizeOfCur.height), Range(0, sizeOfCur.width));
	/*当前帧的特征点检测*/
	OrbFeatureDetector featureDetector;//ORB点检测器
	vector<KeyPoint> keyPoints;
	Mat descriptors;
	featureDetector.detect(ROIImage, keyPoints);//检测特征关键点

	OrbDescriptorExtractor featureExtractor;//ORB点特征提取器
	featureExtractor.compute(ROIImage, keyPoints, descriptors);
	//存入成员变量
	vDescriptors.push_back(descriptors);//每帧的ORB关键点描述子
	vKeyPoints.push_back(keyPoints);//每帧的ORB关键点*

	/*跨帧特征点匹配*/
	if (iNumOfFrames >= 4)
	{
		//实例化一个匹配器
		BruteForceMatcher <L2<float>> matcher;
		vector<DMatch> matches;//匹配结果
		matcher.match(vDescriptors[iNumOfFrames-4], vDescriptors[iNumOfFrames-1], matches);
		//匹配点排序
		nth_element(matches.begin(), matches.begin() + 2, matches.end());
		matches.erase(matches.begin() + 3, matches.end());//擦除
		//把关键点转换为普通点,并存入vector<vector<Point>> vPoints，作为备选点
		vector<Point> points;
		for (int i = 0; i < 3; i++)
		{
			points.push_back(keyPoints[matches[i].trainIdx].pt);
			points[i].y += (int)(rate*sizeOfCur.height);//恢复为原图像中的坐标点
		}
		vPoints.push_back(points);
		vMatches.push_back(matches);
		//清理
		points.clear();
		matcher.clear();
		matches.clear();
	}
	keyPoints.clear();
	int thei = 0;
	int thej = 0;
	if (10 == iNumOfFrames)//第10帧时，确定基准点
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

int lavaDetector::getPosOfBase2()//用于确定基准点的位置（方法二）
{
	/*	vector<Mat> vDescriptors;//*
	vector<vector<KeyPoint>> vKeyPoints;//每帧的ORB关键点*
	vector<vector<Point>> vPoints;//匹配后每帧的备选点**/
	Mat theNormal = imread("vDescriptors.png",0);//加载目标特征
	const double rate = 0.4;//去掉上半部分的比例
	//把原图像转换为灰度图，并裁切
	Mat grayImage;
	cvtColor(curImage_color, grayImage, CV_BGR2GRAY);
	Mat ROIImage = grayImage(Range((int)(rate*sizeOfCur.height), sizeOfCur.height), Range(0, sizeOfCur.width));
	//当前帧的特征点检测
	OrbFeatureDetector featureDetector;//ORB点检测器
	vector<KeyPoint> keyPoints;//关键点
	Mat descriptors;//关键点特征描述子
	featureDetector.detect(ROIImage, keyPoints);//检测特征关键点
	OrbDescriptorExtractor featureExtractor;//ORB点特征提取器
	featureExtractor.compute(ROIImage, keyPoints, descriptors);//计算关键点的特征描述子
	//存入成员变量容器
	vDescriptors.push_back(descriptors);//每帧的ORB关键点描述子
	vKeyPoints.push_back(keyPoints);//每帧的ORB关键点

	//实例化一个匹配器
	BruteForceMatcher <L2<float>> matcher;
	vector<DMatch> matches;//匹配结果
	matcher.match(theNormal, descriptors, matches);//寻找当前帧最匹配的特征点

	//把关键点转换为普通点,并存入vector<vector<Point>> vPoints，作为备选点（为保证两种方法的兼容，采用容器嵌套）
	vector<Point> points;//关键点转换为普通点
	points.push_back(keyPoints[matches[0].trainIdx].pt);
	distance[iNumOfFrames-1] = matches[0].distance;//当前帧最匹配的特征点与目标点的特征距离
	points[0].y += (int)(rate*sizeOfCur.height);//恢复为原图像中的坐标点
	vPoints.push_back(points);//最匹配点坐标存入成员变量容器
	//清理容器
	points.clear();
	matcher.clear();
	matches.clear();
	keyPoints.clear();

	if (10 == iNumOfFrames)//第10帧时，确定基准点
	{
		int sIndex = 0;//最近距离的索引值
		int num = distance[0];//最近距离
		for(int i = 0;i<10;i++)
		{
			if(distance[i] < num)
			{
				num = distance[i];
				sIndex = i;
			}
		}
		pPosOfBase = vPoints[sIndex][0];
	}
	return 0;
}

int lavaDetector::getPosOfDetect()//确定岩棉检测点、比例尺、熔岩检测点
{
	const Mat& binary = curImage_binary;
	/*遍历寻找左侧岩棉检测位置（11-15帧内都执行，15帧最终确定）*/
	int baseX = pPosOfBase.x;//基准点的x
	int baseY = pPosOfBase.y;//基准点的y
	vector<int> distance;//记录上下距离
	for(int x = baseX;x < (sizeOfCur.width - baseX); x++)//从基准点出发，向左横向遍历
	{
		if(binary.ptr<uchar>(baseY)[x] == 255)//如果当前起始像素为255，则继续向左遍历
		{
			distance.push_back(baseY);
		}
		else//如果当前起始像素为0，则开始向上遍历
		{
			for(int y = baseY; y > 0; y--)//当前点向上遍历
			{
				if(binary.ptr<uchar>(y)[x] == 255 && binary.ptr<uchar>(y+1)[x] == 0)//如果该点亮度刚好由0变为255，则停止向上
				{
					distance.push_back(baseY - y);//记录距离
					break;	
				}
			}
		}
	}

	/*确定岩棉检测点、比例尺、熔岩检测点*/
	if(15 == iNumOfFrames)
	{
	
	}
	return 0;
}