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
	newPoint operator = (newPoint& point)
	{
		x = point.x;
		y = point.y;
		dir = point.dir;
		return *this;
	}
};

struct T
{
	Mat pic;
	Mat colpic;
	Mat rowpic;
	vector<newPoint> colseam;
	vector<newPoint> rowseam;
	double cseam,rseam;
	double nowgain;
	type choose;
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

int calculateMin(Node** seam,int row,int col,double& nummax)
{
	int colmax = 0;
	nummax = INT_MAX;
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

Mat removeLine(Node** seam,Mat temp,int removenewPoint,int row,int col,vector<newPoint >& nowRoute,type which)
{
	Mat now = Mat(row,col-1,CV_8UC3);
	int nowrow = row - 1;
	while (removenewPoint != -1 && nowrow >= 0)
	{
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

void getInfo(T& picture)
{
	int row = picture.pic.rows;
	int col = picture.pic.cols;
	unsigned short** energy = new unsigned short*[row];
	for (int i = 0;i < row;++i)
		energy[i] = new unsigned short[col];
	calculateEnergy(picture.pic,energy,row,col);
	Node** seam = new Node*[row];
	for (int i = 0;i < row;++i)
		seam[i] = new Node[col];
	for (int i = 0;i < row;++i)
		for (int j = 0;j < col;++j)
			seam[i][j].number = INT_MAX;
	DP(seam,energy,row,col);
	int removenewPoint = calculateMin(seam,row,col,picture.cseam);
	type nowType = Col;
	picture.colpic = removeLine(seam,picture.pic,removenewPoint,row,col,picture.colseam,nowType);
	picture.pic = picture.pic.t(),nowType = Row;
	row = picture.pic.rows;
	col = picture.pic.cols;
	unsigned short** energyr = new unsigned short*[row];
	for (int i = 0;i < row;++i)
		energyr[i] = new unsigned short[col];
	calculateEnergy(picture.pic,energyr,row,col);
	Node** seamr = new Node*[row];
	for (int i = 0;i < row;++i)
		seamr[i] = new Node[col];
	for (int i = 0;i < row;++i)
		for (int j = 0;j < col;++j)
			seamr[i][j].number = INT_MAX;
	DP(seamr,energyr,row,col);
	removenewPoint = calculateMin(seamr,row,col,picture.rseam);
	picture.rowpic = removeLine(seamr,picture.pic,removenewPoint,row,col,picture.rowseam,nowType).t();
	picture.pic = picture.pic.t();
}

int main(int argc,char** argv)
{
	std::string filename(argv[1]);
	Mat input = imread(filename);
	Mat temp = input;
	currentMap = new int[temp.rows];
	Mat origin = input;
	int totrow= atof(argv[4])*temp.rows;
	int totcol = atof(argv[3])*temp.cols;
	int row = input.rows;
	int col = input.cols;
	if (string(argv[2]) == "cut")
	{
		vector<vector<T> > dpphoto;
		vector<T> gg(totcol + 1);
		for (int i = 0;i < totrow + 1;++i)
			dpphoto.push_back(gg);
		dpphoto[0][0].pic = input;
		dpphoto[0][0].nowgain = 0;
		getInfo(dpphoto[0][0]);
		for (int i = 0;i <= totrow;++i)
			for (int j = 0;j <= totcol;++j)
			{
				if (i == 0 && j == 0) continue;
				else if (i == 0)
				{
					dpphoto[i][j].pic = dpphoto[i][j-1].colpic;
					dpphoto[i][j].nowgain += dpphoto[i][j-1].cseam;
					dpphoto[i][j].choose = Col;
					getInfo(dpphoto[i][j]);
				}
				else if (j == 0)
				{
					dpphoto[i][j].pic = dpphoto[i-1][j].rowpic;
					dpphoto[i][j].nowgain += dpphoto[i-1][j].rseam;
					dpphoto[i][j].choose = Row;
					getInfo(dpphoto[i][j]);
				}
				else
				{
					if (dpphoto[i-1][j].nowgain + dpphoto[i-1][j].rseam > dpphoto[i][j-1].nowgain + dpphoto[i][j-1].cseam)
					{
						dpphoto[i][j].nowgain = dpphoto[i][j-1].nowgain + dpphoto[i][j-1].cseam;
						dpphoto[i][j].choose = Col;
						dpphoto[i][j].pic = dpphoto[i][j-1].colpic;
						getInfo(dpphoto[i][j]);
					}
					else
					{
						dpphoto[i][j].nowgain = dpphoto[i-1][j].nowgain + dpphoto[i-1][j].rseam;
						dpphoto[i][j].choose = Row;
						dpphoto[i][j].pic = dpphoto[i-1][j].rowpic;
						getInfo(dpphoto[i][j]);
					}
				}
			}
		Mat seamPic = dpphoto[totrow][totcol].pic;
		imwrite("result.jpg",dpphoto[totrow][totcol].pic);
		int r = totrow,c = totcol;
		while (r >= 0 && c >= 0)
		{
			if (r == 0 && c == 0) break;
			// printf("%d %d\n",r,c);
			if (dpphoto[r][c].choose == Col)
			{
				--c;
				if (c < 0) break;
				Mat pic = Mat(seamPic.rows,seamPic.cols+1,CV_8UC3);
				for (int j = seamPic.rows-1;j >= 0;--j)
				{
					// printf("%d\n",j );
					int num = 0;
					for (int p = 0;p < seamPic.cols;++p)
						if (p == dpphoto[r][c].colseam[j].x)
						{
							pic.at<Vec3b>(dpphoto[r][c].colseam[j].y,num++) = Vec3b(0,0,255);
							pic.at<Vec3b>(dpphoto[r][c].colseam[j].y,num++) = seamPic.at<Vec3b>(dpphoto[r][c].colseam[j].y,p);
						}
						else
							pic.at<Vec3b>(dpphoto[r][c].colseam[j].y,num++) = seamPic.at<Vec3b>(dpphoto[r][c].colseam[j].y,p);
				}
				seamPic = pic;
			}
			else
			{
				--r;
				if (r < 0) break;
				Mat pic = Mat(seamPic.rows+1,seamPic.cols,CV_8UC3);
				for (int j = seamPic.cols-1;j >= 0;--j)
				{
					int num = 0;
					for (int p = 0;p < seamPic.rows;++p)
						if (p == dpphoto[r][c].rowseam[j].y)
						{
							pic.at<Vec3b>(num++,dpphoto[r][c].rowseam[j].x) = Vec3b(0,0,255);
							pic.at<Vec3b>(num++,dpphoto[r][c].rowseam[j].x) = seamPic.at<Vec3b>(p,dpphoto[r][c].rowseam[j].x);
						}
						else
							pic.at<Vec3b>(num++,dpphoto[r][c].rowseam[j].x) = seamPic.at<Vec3b>(p,dpphoto[r][c].rowseam[j].x);
				}
				seamPic = pic;
			}
		}
		imshow("window",seamPic);
		waitKey(0);
	}
	else if (string(argv[2]) == "amplify")
	{
		unsigned short** energy = new unsigned short*[row];
		for (int i = 0;i < row;++i)
			energy[i] = new unsigned short[col];
		unsigned short** addtime = new unsigned short*[row];
		for (int i = 0;i < row;++i)
			addtime[i] = new unsigned short[col];
		for (int i = 0;i < row;++i)
			for (int j = 0;j < col;++j)
				addtime[i][j] = 0;
		calculateEnergy(input,energy,row,col);
		Node** seam = new Node*[row];
		for (int i = 0;i < row;++i)
			seam[i] = new Node[col];
		for (int i = 0;i < row;++i)
			for (int j = 0;j < col;++j)
				seam[i][j].number = INT_MAX;
		DP(seam,energy,row,col);
		double meiyongde = 0;
		while (--totcol)
		{
			int removenewPoint = calculateMin(seam,row,col,meiyongde);
			int t = row-1;
			seam[t][removenewPoint].number = INT_MAX;
			while (removenewPoint != -1)
			{
				addtime[t][removenewPoint]++;
				removenewPoint = seam[t][removenewPoint].last;
			}
		}
		// for (int i  = 0;i < row;++i)
		// 	delete[] energy[i],seam[i],addtime[i];
		// delete[] energy,seam,addtime;
		Mat amplifyPic = Mat(temp.rows,temp.cols + atof(argv[3])*temp.cols,CV_8UC3);
		int k = 0;
		for (int i = 0;i < temp.rows;++i)
			for (int j = 0;j < temp.cols;++j)
				if (addtime[i][j] == 0)
					amplifyPic.at<Vec3b>(i,++k) = temp.at<Vec3b>(i,j);
				else
					for (int p = 0;p <= addtime[i][j];++p)
						amplifyPic.at<Vec3b>(i,++k) = temp.at<Vec3b>(i,j);
		amplifyPic = amplifyPic.t();
		row = amplifyPic.rows;
		col = amplifyPic.cols;
		unsigned short** energyr = new unsigned short*[row];
		for (int i = 0;i < row;++i)
			energyr[i] = new unsigned short[col];
		calculateEnergy(amplifyPic,energyr,row,col);
		Node** seamr = new Node*[row];
		for (int i = 0;i < row;++i)
			seamr[i] = new Node[col];
		addtime = new unsigned short*[row];
		for (int i = 0;i < row;++i)
			addtime[i] = new unsigned short[col];
		for (int i = 0;i < row;++i)
			for (int j = 0;j < col;++j)
				addtime[i][j] = 0;
		for (int i = 0;i < row;++i)
			for (int j = 0;j < col;++j)
				seamr[i][j].number = INT_MAX;
		DP(seamr,energyr,row,col);
		while (--totrow)
		{
			int removenewPoint = calculateMin(seamr,row,col,meiyongde);
			int t = row-1;
			seamr[t][removenewPoint].number = INT_MAX;
			while (removenewPoint != -1)
			{
				addtime[removenewPoint][t]++;
				removenewPoint = seamr[t][removenewPoint].last;
			}
		}
		Mat amplify = Mat(temp.rows + atof(argv[4])*temp.rows,temp.cols + atof(argv[3])*temp.cols,CV_8UC3);
		k = 0;
		for (int i = 0;i < temp.rows;++i)
			for (int j = 0;j < temp.cols;++j)
				if (addtime[i][j] == 0)
					amplify.at<Vec3b>(i,++k) = amplifyPic.at<Vec3b>(i,j);
				else
					for (int p = 0;p <= addtime[i][j];++p)
						amplify.at<Vec3b>(i,++k) = amplifyPic.at<Vec3b>(i,j);
		amplify = amplify.t();
		imshow("window",amplify);
		imwrite("result.jpg",amplify);
		waitKey(0);
	}
	else printf("Wrong Command!\n");
	return 0;
}

