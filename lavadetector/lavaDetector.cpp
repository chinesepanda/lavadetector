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
lavaDetector::lavaDetector():iPeriod(300)/*检测周期*/,dThreshold(25)/*分割的阈值*/,iRadiusOfCircle_mm(400)/*轴承预设置的半径*/,iPeriodOfMod1(10)/*动态视频检测周期，以帧数为单位*/,THEMODE(0)
{
	iNumOfFrames = 0;//mod2当前帧序号
	iThickOfWool_p = 0;
	iPosOfZero_p = 1614;
	pPosOfBase_tmp.x = 1614;//
	pPosOfBase_tmp.y = 1349;
	WRONG = 0;
	WRONG_MOD1 = 0;
	iNumOfFramesOfMod1 = 0;//mod1当前帧序号
}
lavaDetector::lavaDetector(int period, double thresh, int posOfZero_p,int mode) : iPeriod(period), dThreshold(thresh),iRadiusOfCircle_mm(400),iPeriodOfMod1(10)/*动态视频检测周期，以帧数为单位*/,THEMODE(mode)
{
	iPosOfZero_p = posOfZero_p;//初始零点x坐标
	iNumOfFrames = 0;
	iThickOfWool_p = 0;
	pPosOfBase_tmp.x = 1614;
	pPosOfBase_tmp.y = 1349;
	WRONG = 0;
	 WRONG_MOD1 = 0;
	iNumOfFramesOfMod1 = 0;//mod1当前帧序号
}
lavaDetector::~lavaDetector()
{

}
//public:
int lavaDetector::readCurrentFrame(Mat src, int mode/*设定当前工作模式*/)//读取当前帧
{
	WRONG = 0;
	iWorkmode = mode;
	if (iWorkmode != 0 && iWorkmode != 1 && iWorkmode != 2)
	{
		return -1;//工作模式错误
	}
	if(1 == iWorkmode)//如果是模式2,对帧数进行计数
	{
		iNumOfFramesOfMod1++;
	}
	if (2 == iWorkmode)//如果是模式3，则以iPeriod为周期，对帧数进行计数
	{
		iNumOfFrames++;
		if (iPeriod <= iNumOfFrames)//周期结束，帧数重置
		{
			iNumOfFrames = 1;
		}
	}
	/*通用代码*/
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
	if (iWorkmode != 1)//检测模式是否匹配
	{
		return -1;//模式错误
	}
	
	if( iNumOfFramesOfMod1<= iPeriodOfMod1)//1-10帧
	{
		/*①轮廓面积*/
		vector< vector<Point> > contours;   // 轮廓   
		vector< Vec4i > hierarchy;    // 轮廓的结构信息 
		contours.clear();
		hierarchy.clear();
		Mat tmp_binary = curImage_binary.clone();
		findContours(tmp_binary, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
		long area = 0;
		for (size_t i = 0; i < contours.size(); i++)
		{
			area += fabs(contourArea(Mat(contours[i])));
		}
		vArea.push_back(area);//存入容器
		/*②特征点*/
		//把原图像转换为灰度图，并裁切
		const double rate = 0.4;//去掉上半部分的比例
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
		double numOfKeypoints;
		if(keyPoints.size() > 10)
		{
			numOfKeypoints = 10.0;
		}
		else
		{
			numOfKeypoints = (double)(keyPoints.size());
		}
		vNumOfKeyPoints.push_back(numOfKeypoints);//存入当前帧的特征点数量
		keyPoints.clear();
		/*检测周期结束，统计*/
		if(iPeriodOfMod1 == iNumOfFramesOfMod1)//mod1第10帧时，计算平均轮廓面积
		{
			long meanArea = 0;
			double meanNumOfKeypoints = 0.0;
			for(int i = 0;i < iPeriodOfMod1;i++)
			{
				meanArea += (vArea[i]/iPeriodOfMod1);
				meanNumOfKeypoints += vNumOfKeyPoints[i]/iPeriodOfMod1;
			}
			vArea.clear();//清理
			vNumOfKeyPoints.clear();
			cout<<"meanArea"<<meanArea<<endl;
			if(meanArea < 120000)
			{
				 WRONG_MOD1 = -1;//检测面积过小
			}
			if(meanNumOfKeypoints < 5)
			{
				 WRONG_MOD1 = -2;//特征点过少
			}
		}
	}
	return 0;
}
int lavaDetector::imageDetect()//岩棉检测 mode 2:
{
	WRONG = 0;
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

		if(THEMODE == 0)//自动检测模式
		{
			getPosOfBase2();//确定当前周期的基准点
		}
		else//预设置模式
		{
			pPosOfBase = pPosOfBase_tmp;//预设置基准点
		}
	}
	else if (iNumOfFrames >= 11 && iNumOfFrames <= 21)//11-15帧确定岩棉检测点、流股监测点、比例尺
	{
		if(THEMODE == 0)//自动检测模式
		{
			getPosOfDetect();//确定岩棉检测点、比例尺、熔岩检测点
		}
		else//预设置模式
		{
			pPosThickDetect.x = 1255;
			pPosThickDetect.y = 1018;
			pPosStreamDetect.x = 1614;
			pPosStreamDetect.y = 687;
			dScale = 1.11421;
		}
		
	}
	else//从22帧开始，检测三项数据 
	{
		detectThickOfWool();//检测成纤厚度
		detectWidthOfStream();//检测流股宽度、下落点
	}

	return 0;
}
int lavaDetector::setPosOfZero_p(int& posOfZero_p)//设置零点x坐标
{
	iPosOfZero_p = posOfZero_p;
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
		Point pointZero_tmp;
		pointZero_tmp.x = iPosOfZero_p;
		pointZero_tmp.y = 0;
		plotLine(imageToShow_color,pointZero_tmp,Scalar(255,0,0),0);//零点

		if(iNumOfFrames >= 11)
		{
			plotPointTarget(imageToShow_color,pPosOfBase,Scalar(0,0,255),1);
		}
		if(iNumOfFrames > 21)
		{	
			plotPointTarget(imageToShow_color,pPosThickDetect,Scalar(255,0,0),2);
			plotPointTarget(imageToShow_color,pPosStreamDetect,Scalar(255,0,0),2);
			//绘制成纤处检测结果 pPosThickPlot
			plotPointTarget(imageToShow_color,pPosThickPlot,Scalar(255,255,255),0);//成纤绘制点的坐标
			plotLine(imageToShow_color,pPosThickDetect,Scalar(255,255,255),1);//成纤检测点
			//cout<<iThickOfWool_p<<endl;//输出成纤厚度dThickOfWool_mm
			//绘制流股处检测结果
			plotPointTarget(imageToShow_color,pPosStreamPlot,Scalar(255,255,255),0);//流股绘制点的坐标
			//cout<<iWidthOfStream_p<<endl;
			//绘制下落点检测结果
			plotLine(imageToShow_color,pPosStreamPlot,Scalar(255,255,255),0);//流股绘制点


		}
	}
	imageToShow_color.copyTo(imagetoshow);
	return 0;
}
int lavaDetector::getParameters_p(int& thickOfWool_p/*成纤厚度*/, int& widthOfStream_p/*流股宽度*/,int& rePosOfDrop_p/*落点偏移量*/)//获取三项待检测参数（像素为单位）
{
	thickOfWool_p = iThickOfWool_p;
	widthOfStream_p = iWidthOfStream_p;
	rePosOfDrop_p = iRePosOfDrop_p;
	return 0;
}
int lavaDetector::getParameters_mm(double& thickOfWool_mm,double& widthOfStream_mm, double& rePosOfDrop_mm)//获取三项待检测参数（mm为单位）
{
	//把像素值转换为mm值
	dThickOfWool_mm = ((double)iThickOfWool_p) * dScale;
	thickOfWool_mm = dThickOfWool_mm;
	//把像素值转换为mm值
	dWidthOfStream_mm = ((double)iWidthOfStream_p) * dScale;
	widthOfStream_mm = dWidthOfStream_mm;
	//把像素值转换为mm值
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
	//如果全景为黑
	if(filterContours.size() == 0)
	{
		WRONG = -4;//错误：当前帧内为无目标
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
	
	if(keyPoints.size() != 0)//如果当前帧存在关键点
	{
		//实例化一个匹配器
		BruteForceMatcher <L2<float>> matcher;
		vector<DMatch> matches;//匹配结果
		matcher.match(theNormal, descriptors, matches);//寻找当前帧最匹配的特征点

		//把关键点转换为普通点,并存入vector<vector<Point>> vPoints，作为备选点（为保证两种方法的兼容，采用容器嵌套）
		vector<Point> points;//关键点转换为普通点
		points.push_back(keyPoints[matches[0].trainIdx].pt);
		distance[iNumOfFrames-1] = matches[0].distance;//当前帧最匹配的特征点与目标点的特征距离
		//cout<<"距离： "<<matches[0].distance<<endl;
		points[0].y += (int)(rate*sizeOfCur.height);//恢复为原图像中的坐标点
		vPoints.push_back(points);//最匹配点坐标存入成员变量容器
		index[iNumOfFrames-1] = 1;//index[]==1,说明当前帧的数据有效
		points.clear();
	    matcher.clear();
	    matches.clear();
	}
	else//如果当前帧不存在关键点
	{	
		vector<Point> points;
		vPoints.push_back(points);//最匹配点坐标存入成员变量容器（占位）
		index[iNumOfFrames-1] = 0;//index[]==0,说明当前帧的数据无效
		distance[iNumOfFrames-1] = 80000;
		points.clear();
		//cout<<"无关键点"<<endl;
	}
	//清理容器
	keyPoints.clear();

	if (10 == iNumOfFrames)//第10帧时，确定基准点
	{
		double num;//最近距离
		int sIndex;//最近距离的索引值
		int k;//帧数索引
		int numOfFrames = 0;
		for(k = 0;k<10;k++)//寻找第一个有关键点的帧
		{
			num = distance[k];
			sIndex = k;
			if(1 == index[k])
			{
				break;
			}
		}
		for(int i = k;i<10;i++)//计算有关键点的帧的总数
		{
			if(1 == index[k])
			{
				numOfFrames++;
			}
		}
		
		if(10 == k || numOfFrames<=5)//如果10帧都没有关键点或者有关键点的帧数小于等于5
		{
			pPosOfBase = pPosOfBase_tmp;
			WRONG = -1;//错误信息：过少的匹配点
			//cout<<"WRONG = -1;//错误信息：过少的匹配点"<<endl;
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
		/*寻找遍历的左边界*/
		/*
		edgeX = pPosOfBase.x;
		for(int a = 0;a < pPosOfBase.x;a++)
		{
			int bre = 0;//跳出循环
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
				WRONG = -2;//错误信息：检测不到左侧成纤
			}
		}*/
	}
	return 0;
}

int lavaDetector::getPosOfDetect()//11-21帧确定岩棉检测点、比例尺、熔岩检测点
{
	//double t1=(double)getTickCount();
	const Mat& binary = curImage_binary;
	/*遍历寻找左侧岩棉检测位置（11-21帧内都执行，21帧最终确定）*/
	const int baseX = pPosOfBase.x;//基准点的x
	const int baseY = pPosOfBase.y;//基准点的y
	vector<int> vDistance;//记录上下距离
	vector<int> vX;

	/*寻找遍历的左边界*/
	for(int a = 0;a < pPosOfBase.x;a++)
	{
		int bre = 0;//跳出循环
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
			WRONG = -2;//错误信息：检测不到左侧成纤
			//cout<<"WRONG = -2;//错误信息：检测不到左侧成纤"<<endl;
		}
	}
	
	/*开始向左横向遍历*/
	for(int x = baseX;x >= edgeX; x--)//从基准点出发，向左横向遍历
	{
		vX.push_back(x);
		if(binary.ptr<uchar>(baseY)[x] == 255)//如果当前起始像素为255，则继续向左遍历
		{
			vDistance.push_back(0);//把当前距离设为可能的最小值0
		}
		else//如果当前起始像素为0，则开始向上遍历
		{
			for(int y = baseY; y >= 0; y--)//当前点向上遍历
			{
				if(binary.ptr<uchar>(y)[x] == 255 && binary.ptr<uchar>(y+1)[x] == 0)//如果该点亮度刚好由0变为255，则停止向上
				{
					vDistance.push_back(baseY - y);//记录距离
					break;	
				}
				if(0 == y)
				{
					vDistance.push_back(0);//向上遍历至最顶端，仍然没有满足条件的情况（刚好由0变为255），则距离设为0
				}
			}	
		}
	}
	//寻找vD中的最大距离
	vector<int>::iterator biggest = max_element(begin(vDistance), end(vDistance));//迭代器
	iBiggestDistance[iNumOfFrames - 11] = *biggest;//存入最大距离
	iCorresX[iNumOfFrames - 11] = vX[std::distance(std::begin(vDistance), biggest)];//存入最大距离对应的x坐标
	//清理
	vDistance.clear();
	vX.clear();
	/*确定岩棉检测点、比例尺、熔岩检测点*/
	if(21 == iNumOfFrames)
	{
		/*成纤监测点*/
		int iBiggestDistance_sorted[11] = {0};
		memcpy(iBiggestDistance_sorted,iBiggestDistance,11*sizeof(int));//创建一个拷贝数组
		select_sort(iBiggestDistance_sorted, 11);//对拷贝数组进行排序
		int noZero;//iBiggestDistance_sorted数组非零的元素数量
		for(int k = 0;k<11;k++)
		{
			noZero = k;
			if(iBiggestDistance_sorted[k] == 0)
			{
				break;
			}
			 
		}
		if(noZero <= 2)//如果没有有足够的帧可以找到合适的距离
		{
			WRONG = -3;//错误信息：没有有足够的帧可以找到合适的距离
			//cout<<"WRONG = -3;//错误信息：没有有足够的帧可以找到合适的距离"<<endl;
		}
		for(int i = 0;i<11;i++)
		{
			if(iBiggestDistance[i] == iBiggestDistance_sorted[noZero/2])//获取数组中位数对应的index及对应的x坐标
			{
				pPosThickDetect.x = iCorresX[i];//成纤监测点的坐标
				pPosThickDetect.y = baseY - iBiggestDistance_sorted[noZero/2];
				cout<<"pPosThickDetect.x "<<pPosThickDetect.x<<"pPosThickDetect.y "<<pPosThickDetect.y <<endl;
				iRadiusOfCircle_p = baseX - iCorresX[i];//轴承半径的像素数
				dScale = ((double)iRadiusOfCircle_mm) /((double)iRadiusOfCircle_p);//计算比例尺
				cout<<"dScale"<<dScale<<endl;
				pPosStreamDetect.x = baseX;//流股检测点的x坐标与基准点x坐标相同
				pPosStreamDetect.y = pPosThickDetect.y - iBiggestDistance_sorted[noZero/2];//流股检测点的y坐标等于成纤监测点的y坐标减去轴承纵半径
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

int lavaDetector::getPosOfDetect2()//每帧确定岩棉检测点、比例尺、熔岩检测点
{
	
	double t1=(double)getTickCount();
	const Mat& binary = curImage_binary;
	/*遍历寻找左侧岩棉检测位置（）*/
	const int baseX = pPosOfBase.x;//基准点的x
	const int baseY = pPosOfBase.y;//基准点的y
	vector<int> vDistance;//记录上下距离
	vector<int> vX;
	for(int x = baseX;x >= edgeX; x--)//从基准点出发，向左横向遍历
	{
		vX.push_back(x);
		if(binary.ptr<uchar>(baseY)[x] == 255)//如果当前起始像素为255，则继续向左遍历
		{
			vDistance.push_back(0);//把当前距离设为可能的最小值0
		}
		else//如果当前起始像素为0，则开始向上遍历
		{
			for(int y = baseY; y >= 0; y--)//当前点向上遍历
			{
				if(binary.ptr<uchar>(y)[x] == 255 && binary.ptr<uchar>(y+1)[x] == 0)//如果该点亮度刚好由0变为255，则停止向上
				{
					vDistance.push_back(baseY - y);//记录距离
					break;	
				}
				if(0 == y)
				{
					vDistance.push_back(0);//向上遍历至最顶端，仍然没有满足条件的情况（刚好由0变为255），则距离设为0
				}
			}	
		}
	}
	//寻找vD中的最大距离
	vector<int>::iterator biggest = max_element(begin(vDistance), end(vDistance));//迭代器
	int biggestY = *biggest;//存入最大距离
	int biggestX = vX[std::distance(std::begin(vDistance), biggest)];//存入最大距离对应的x坐标
	//清理
	vDistance.clear();
	vX.clear();
	/*确定岩棉检测点、比例尺、熔岩检测点*/

	pPosThickDetect.x = biggestX;//成纤监测点的坐标
	pPosThickDetect.y = baseY - biggestY;

	iRadiusOfCircle_p = baseX - biggestX;//轴承半径的像素数
	dScale = iRadiusOfCircle_mm /((double)iRadiusOfCircle_p);//计算比例尺

	pPosStreamDetect.x = baseX;//流股检测点的x坐标与基准点x坐标相同
	pPosStreamDetect.y = pPosThickDetect.y - biggestY;//流股检测点的y坐标等于成纤监测点的y坐标减去轴承纵半径


	double t2=(double)getTickCount();
	double time = (t2-t1)/getTickFrequency();
	cout<<time<<endl;
	return 0;
}

int lavaDetector::plotPointTarget(Mat& src_color/*待绘制的图像*/,Point pointForPlot,const Scalar& color/*绘制的颜色*/,int modeOfPlot/*绘制方式*/)//在图中绘制点的target
{
	/*modeOfPlot绘制方式：
	0:target + 方框
	1:target+圆圈
	2:直接绘制圆圈
	*/
	if(0 == modeOfPlot)//target + 方框
	{
		Point pt1[4] = {pointForPlot,pointForPlot,pointForPlot,pointForPlot};//target四条准星线的起始点
		Point pt2[4] = {pointForPlot,pointForPlot,pointForPlot,pointForPlot};//target四条准星线的结束点
		Point pt3[4] = {pointForPlot,pointForPlot,pointForPlot,pointForPlot};//方框四角点

		//设置相对偏移量
		pt1[0].y -= 30; pt1[1].x += 30;	pt1[2].y += 30;	pt1[3].x -= 30;
		pt2[0].y -= 70;	pt2[1].x += 70;	pt2[2].y += 70;	pt2[3].x -= 70;
		pt3[0].x -= 80; pt3[0].y -= 80;	pt3[1].x += 80; pt3[1].y -= 80;	pt3[2].x += 80; pt3[2].y += 80;	pt3[3].x -= 80; pt3[3].y += 80;	
		Point pt4[4] = {pt3[0],pt3[1],pt3[2],pt3[3]};//方框四角左起始点
		Point pt5[4] = {pt3[0],pt3[1],pt3[2],pt3[3]};//方框四角右起始点
		pt4[0].y =pt3[0].y + 40; pt4[1].x = pt3[1].x - 40; pt4[2].y = pt3[2].y - 40; pt4[3].x = pt3[3].x + 40;
		pt5[0].x =pt3[0].x + 40; pt5[1].y = pt3[1].y + 40; pt5[2].x = pt3[2].x - 40; pt5[3].y = pt3[3].y - 40;
		//绘制
		for(int i = 0;i<4;i++)
		{
			line(src_color,pt1[i],pt2[i],color,4);//绘制准星线
			line(src_color,pt3[i],pt4[i],color,4);//绘制边框线
			line(src_color,pt3[i],pt5[i],color,4);//绘制边框线
		}
		//circle(src_color,pointForPlot,5,color,4);//绘制中心圆
	}
	else if(1 == modeOfPlot)//target+圆圈
	{
		Point pt1[4] = {pointForPlot,pointForPlot,pointForPlot,pointForPlot};//target四条准星线的起始点
		Point pt2[4] = {pointForPlot,pointForPlot,pointForPlot,pointForPlot};//target四条准星线的结束点
		//设置相对偏移量
		pt1[0].y -= 20;	pt1[1].x += 20;	pt1[2].y += 20;	pt1[3].x -= 20;
		pt2[0].y -= 70;	pt2[1].x += 70;	pt2[2].y += 70;	pt2[3].x -= 70;
		for(int i = 0;i<4;i++)
		{
			line(src_color,pt1[i],pt2[i],color,4);//绘制准星线
		}
		circle(src_color,pointForPlot,5,color,4);//绘制中心圆
	}
	else if(2 == modeOfPlot)//直接绘制圆圈
	{
			circle(src_color,pointForPlot,5,color,4);
	}
	return 0;
}

int lavaDetector::plotLine(Mat& src_color,Point pointForPlot,const Scalar& color,int modeOfPlot)//在图中绘制直线
{
	/*modeOfPlot绘制方式：
	0:竖线
	1:横线
	*/
	if(0 == modeOfPlot)//竖线
	{
		Point p1 = pointForPlot;
		p1.y = 0;
		Point p2 = pointForPlot;
		p2.y = sizeOfCur.height;
		line(src_color,p1,p2,color,1,CV_AA );
	}
	else if (1 == modeOfPlot)//横线
	{
		Point p1 = pointForPlot;
		p1.x = 0;
		Point p2 = pointForPlot;
		p2.x = sizeOfCur.width;
		line(src_color,p1,p2,color,1,CV_AA );
	}
	return 0 ;
}
int lavaDetector::detectThickOfWool()//检测岩棉成纤厚度（像素及mm为单位）
{
	/*相关成员变量：
		double dScale;//比例尺
		int iThickOfWool_p;//岩棉的厚度（以像素计）
		double dThickOfWool_mm;//岩棉的厚度（通过比例尺折算为以mm计）
		Point pPosOfBase;//基准点的像素坐标
		Point pPosThickDetect;//成纤检测点的坐标
		Mat curImage_binary;//当前二值图原始帧
		Point pPosThickPlot;//成纤绘制点的坐标
	*/
	int iThickOfWool_p_tmp = iThickOfWool_p;//把上一帧的厚度暂存
	iThickOfWool_p = 0;//初始化当前帧的厚度
	Point pStartPoint;//遍历起始点
	pStartPoint.x = pPosThickDetect.x;
	pStartPoint.y = pPosOfBase.y;
	//局部参数的初始化
	int highY = 0;//流股最上方的y坐标
	int lowY = 0;//流股最下方的y坐标
	int pre = 0;//前一帧遍历点的亮度
	int current = 0;//当前帧遍历点的亮度
	int i =0;//y坐标参数
	//自下而上地遍历，获取流股厚度以及流股中间点
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
	//异常值（0等），则进行均匀处理。同时，成纤绘制点的坐标维持不变
	if(((double)iThickOfWool_p) <= (0.3 * (double)iThickOfWool_p_tmp))
	{
		iThickOfWool_p = iThickOfWool_p_tmp * 0.8;
	}
	else//如果没有异常值，则直接获取当前帧的成纤厚度参数和成纤绘制点的坐标
	{
		//确定成纤绘制点的坐标
		pPosThickPlot.x = pStartPoint.x;
		pPosThickPlot.y = (int)((lowY + highY) * 0.5);
	}

	return 0;
}

int lavaDetector::detectWidthOfStream()//检测熔岩流股宽度（像素及mm为单位）、检测熔岩流股下落点
{
	/*相关成员变量：
		double dScale;//比例尺
		int iWidthOfStream_p;//下落流股的宽度（以像素计）
	    double dWidthOfStream_mm;//下落流股的宽度（通过比例尺折算为以mm计）
		Point pPosStreamDetect;//流股检测点的坐标
		Mat curImage_binary;//当前二值图原始帧
		Point pPosStreamDetect;//流股检测点的坐标
		Point pPosStreamPlot;//流股绘制点的坐标
	*/
	int iWidthOfStream_p_tmp = iWidthOfStream_p;//把上一帧的厚度暂存
	iWidthOfStream_p = 0;//初始化当前帧的厚度
	//局部参数的初始化
	int rightX = 0;//流股最左侧的x坐标
	int leftX = 0;//流股最右侧的x坐标
	int pre = 0;//前一帧遍历点的亮度
	int current = 0;//当前帧遍历点的亮度
	int i =0;//x坐标参数
	//自左而右地遍历，获取流股宽度以及流股中间点
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
	//异常值（0等），则进行均匀处理。同时，成纤绘制点的坐标维持不变
	if(((double)iWidthOfStream_p) <= (0.3 * (double)iWidthOfStream_p_tmp))
	{
		iWidthOfStream_p = iWidthOfStream_p * 0.8;
	}
	else//如果没有异常值，则直接获取当前帧的成纤厚度参数和成纤绘制点的坐标
	{
		//确定成纤绘制点的坐标
		pPosStreamPlot.y = pPosStreamDetect.y;
		pPosStreamPlot.x = (int)((rightX + leftX) * 0.5);
	}

	//获得下落点x坐标
	iPosOfDrop_p = pPosStreamPlot.x;
	iRePosOfDrop_p = iPosOfDrop_p - iPosOfZero_p;//获取相对偏移量（像素）
	
	return 0;
}

void lavaDetector::select_sort(int array[], int n)//选择排序
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