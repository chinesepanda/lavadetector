#include "lavaDetector.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

using namespace std;
using namespace cv;
//测试框架
int main()
{
	string filePath  = "C:\\Users\\fy01\\Desktop\\lava\\data\\mod3\\";
	string fileName = "5000_3.5_6.5(1).avi";//5000_3.5_6.5(1).avi
	string fileAllPath = filePath + fileName;//视频文件全路径
	VideoCapture video(fileAllPath);
	if(!video.isOpened())
	{
		cout<<"读取视频失败"<<endl;
		return -1;
	}
	Mat firstImage;
	Mat imagetoshow;
	Mat imagetoshow_binary;
	lavaDetector myDetector;

	for(int i = 0;i<200;i++)  //137! //170!
	{
		video>>firstImage;
	}
	int frame = 0;
	while(true)
	{
		video>>firstImage;
		if(firstImage.empty())
		{
			return -1;
		}
		frame++;

		if(frame <= 10)
		{
			int w;
			myDetector.readCurrentFrame(firstImage,1);
			myDetector.dynamicImageDetect();//镜头动态内容检测 mode 1
			myDetector.getWRONG_1(w);
			if(w!=0)
			{
				cout<<"1:"<<w<<endl;
				cout<<"---------------"<<endl;
			}
		}
		else
		{
			int w;
			myDetector.readCurrentFrame(firstImage,2);
			myDetector.imageDetect();
			myDetector.getWRONG_2(w);
			if(w!=0)
			{
				cout<<"2:"<<w<<endl;
				cout<<"---------------"<<endl;
			}
			myDetector.getWRONG_2(w);
			if(w!=0)
			{
				cout<<"2:"<<w<<endl;
				cout<<"---------------"<<endl;
			}
		}
		
		myDetector.getBinaryImage(imagetoshow_binary);
		myDetector.getImageToShow(imagetoshow);

		resize(imagetoshow,imagetoshow,Size(),0.4,0.4);
		resize(imagetoshow_binary,imagetoshow_binary,Size(),0.4,0.4);
		imshow("imagetoshow",imagetoshow);
		imshow("image_binary",imagetoshow_binary);
		//myDetector.getImageToShow(imagetoshow);
		//imshow("image",imagetoshow);
		if(waitKey(30) == 27 )
		{
			break;
		}
	}

	return 0;
}