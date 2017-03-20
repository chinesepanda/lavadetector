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

	for(int i = 0;i<100;i++)  //137! //170!
	{
		video>>firstImage;
	}

	while(true)
	{
		video>>firstImage;
		if(firstImage.empty())
		{
			return -1;
		}
		
		myDetector.readCurrentFrame(firstImage,2);
		myDetector.imageDetect();
		myDetector.getBinaryImage(imagetoshow_binary);
		myDetector.getImageToShow(imagetoshow);

		resize(imagetoshow,imagetoshow,Size(),0.4,0.4);
		resize(imagetoshow_binary,imagetoshow_binary,Size(),0.4,0.4);
		imshow("imagetoshow",imagetoshow);
		imshow("image_binary",imagetoshow_binary);
		//myDetector.getImageToShow(imagetoshow);
		//imshow("image",imagetoshow);
		if(waitKey(400) == 27 )
		{
			break;
		}
	}

	return 0;
}