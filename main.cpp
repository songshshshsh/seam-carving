#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv/cv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
 
using namespace cv;
using namespace std;


struct Node
{
	int number;
	int last;
	Node()
	{
		number = INT_MAX;
		last = 0;
	}
};

enum type
{
	Row,Col
};

struct newPoint
{
	int x;
	int y;
	type dir;
};


int* currentMap;

bool inImage(int i,int j,int row,int col)
{
	return (i >= 0 && j >= 0 && i < row && j < col);
}

void calculateEnergy(Mat& input,unsigned short** energy,int row,int col)
{
	Mat energyOutput;
	// Mat energyX,energyY,abs_grad_x,abs_grad_y;
	// Sobel(input,energyX,-1,1,0,3,1,0,BORDER_DEFAULT);
	// Sobel(input,energyY,-1,0,1,3,1,0,BORDER_DEFAULT);
	// convertScaleAbs( energyX, abs_grad_x );
 // 	convertScaleAbs( energyY, abs_grad_y );
 // 	addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0,energyOutput);
	Laplacian(input,energyOutput,-1,1,1,0,BORDER_DEFAULT);
	for (int i = 0;i < row;++i)
		for (int j = 0;j < col;++j)
		{
			energy[i][j] = 0;
			for (int k = 0;k < 3;++k)
				energy[i][j] += (int)(energyOutput.at<Vec3b>(i,j)[k]);
		}
}

void DP(Node** seam, unsigned short**energy,int row,int col)
{
	for (int i = 0;i < col;++i)
	{
		seam[0][i].number = energy[0][i];
		seam[0][i].last = -1;
	}
	for (int i = 1;i < row;++i)
		for (int j = 0;j < col;++j)
		{
			for (int p = j-1;p <= j+1;++p)
				if (p < 0 || p >= col) continue;
				else
				{
					if (seam[i-1][p].number + energy[i][j] <= seam[i][j].number)
					{
						seam[i][j].number = seam[i-1][p].number + energy[i][j];
						seam[i][j].last = p;
					}
				}
		}
}

int calculateMin(Node** seam,int row,int col)
{
	int colmax = 0;
	int nummax = INT_MAX;
	for (int j = 0;j < col-1;++j)
	{
		if (seam[row-1][j].number < nummax)
		{
			colmax = j;
			nummax = seam[row-1][j].number;
		}
	}
	return colmax;
} 

Mat removeLine(Node** seam,Mat temp,int removenewPoint,int row,int col,vector<vector<newPoint > >& routes,type which)
{
	Mat now = Mat(row,col-1,CV_8UC3);
	int nowrow = row - 1;
	vector<newPoint> nowRoute;
	while (removenewPoint != -1 && nowrow >= 0)
	{
		temp.at<Vec3b>(nowrow,removenewPoint) = Vec3b(0,0,255);
		int num = 0;
		newPoint nownewPoint;
		if (which == Col)
		{
			nownewPoint.y = nowrow;
			nownewPoint.x = removenewPoint;
			nownewPoint.dir = Col;
		}
		else 
		{
			nownewPoint.y = removenewPoint;
			nownewPoint.x = nowrow;
			nownewPoint.dir = Row;
		}
		nowRoute.push_back(nownewPoint);
		for (int i = 0;i < col;++i)
			if (i != removenewPoint)
				now.at<Vec3b>(nowrow,num++) = temp.at<Vec3b>(nowrow,i);
		removenewPoint = seam[nowrow][removenewPoint].last;
		nowrow--;
	}
	routes.push_back(nowRoute);
	return now;
}

Mat addLine(Node** seam,Mat temp,int removenewPoint,int row,int col,type which)
{
	Mat now = Mat(row,col+1,CV_8UC3);
	int nowrow = row - 1;
	while (removenewPoint != -1 && nowrow >= 0)
	{
		int num = 0;
		for (int i = 0;i < col;++i)
		{
			if (i == removenewPoint)
				now.at<Vec3b>(nowrow,num++) = temp.at<Vec3b>(nowrow,i);
			now.at<Vec3b>(nowrow,num++) = temp.at<Vec3b>(nowrow,i);
		}
		removenewPoint = seam[nowrow][removenewPoint].last;
		nowrow--;
	}
	return now;
}

int main(int argc,char** argv)
{
	std::string filename(argv[1]);
	Mat input = imread(filename);
	Mat temp = input;
	currentMap = new int[temp.rows];
	Mat origin = input;
	int totTime = atof(argv[3])*temp.cols + atof(argv[4])*temp.rows;
	int totcol = atof(argv[3])*temp.cols;
	if (string(argv[2]) == "cut")
	{
		type nowType;
		if (totcol == 0) 
		{
			temp = temp.t();
			nowType = Row;
		}
		else nowType = Col;
		vector<vector<newPoint > > routes;
		while (totTime--)
		{
			totcol--;
			int row = temp.rows;
			int col = temp.cols;
			unsigned short** energy = new unsigned short*[row];
			for (int i = 0;i < row;++i)
				energy[i] = new unsigned short[col];
			calculateEnergy(temp,energy,row,col);
			Node** seam = new Node*[row];
			for (int i = 0;i < row;++i)
				seam[i] = new Node[col];
			for (int i = 0;i < row;++i)
				for (int j = 0;j < col;++j)
					seam[i][j].number = INT_MAX;
			DP(seam,energy,row,col);
			int removenewPoint = calculateMin(seam,row,col);
			temp = removeLine(seam,temp,removenewPoint,row,col,routes,nowType);
			if (totcol == 0) temp = temp.t(),nowType = Row;
			for (int i  = 0;i < row;++i)
				delete[] energy[i],delete[] seam[i];
			delete[] energy,delete[] seam;
		}
		temp = temp.t();
		Mat seamPic = temp;
		for (int i = routes.size()-1;i >= 0;--i)
		{
			if (routes[i][0].dir == Col)
			{
				Mat pic = Mat(seamPic.rows,seamPic.cols+1,CV_8UC3);
				for (int j = seamPic.rows-1;j >= 0;--j)
				{
					int num = 0;
					for (int p = 0;p < seamPic.cols;++p)
						if (p == routes[i][j].x)
						{
							pic.at<Vec3b>(routes[i][j].y,num++) = Vec3b(0,0,255);
							pic.at<Vec3b>(routes[i][j].y,num++) = seamPic.at<Vec3b>(routes[i][j].y,p);
						}
						else
							pic.at<Vec3b>(routes[i][j].y,num++) = seamPic.at<Vec3b>(routes[i][j].y,p);
				}
				seamPic = pic;
			}
			else
			{
				Mat pic = Mat(seamPic.rows+1,seamPic.cols,CV_8UC3);
				for (int j = seamPic.cols-1;j >= 0;--j)
				{
					int num = 0;
					for (int p = 0;p < seamPic.rows;++p)
						if (p == routes[i][j].y)
						{
							pic.at<Vec3b>(num++,routes[i][j].x) = Vec3b(0,0,255);
							pic.at<Vec3b>(num++,routes[i][j].x) = seamPic.at<Vec3b>(p,routes[i][j].x);
						}
						else
							pic.at<Vec3b>(num++,routes[i][j].x) = seamPic.at<Vec3b>(p,routes[i][j].x);
				}
				seamPic = pic;
			}
		}
		imshow("window",seamPic);
		imwrite("result.jpg",temp);
		waitKey(0);
	}
	else if (string(argv[2]) == "amplify")
	{
		type nowType;
		if (totcol == 0) 
		{
			temp = temp.t();
			nowType = Row;
		}
		else nowType = Col;
		while (totTime--)
		{
			totcol--;
			int row = temp.rows;
			int col = temp.cols;
			unsigned short** energy = new unsigned short*[row];
			for (int i = 0;i < row;++i)
				energy[i] = new unsigned short[col];
			calculateEnergy(temp,energy,row,col);
			Node** seam = new Node*[row];
			for (int i = 0;i < row;++i)
				seam[i] = new Node[col];
			for (int i = 0;i < row;++i)
				for (int j = 0;j < col;++j)
					seam[i][j].number = INT_MAX;
			DP(seam,energy,row,col);
			int removenewPoint = calculateMin(seam,row,col);
			temp = addLine(seam,temp,removenewPoint,row,col,nowType);
			if (totcol == 0) temp = temp.t(),nowType = Row;
			for (int i  = 0;i < row;++i)
				delete[] energy[i],delete[] seam[i];
			delete[] energy,delete[] seam;
		}
		temp = temp.t();
		imshow("window",temp);
		imwrite("result.jpg",temp);
		waitKey(0);
	}
	else printf("Wrong Command!\n");
	return 0;
}

