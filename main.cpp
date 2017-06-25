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
	double number;
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

uchar jueduizhi(uchar a)
{
	if (a > 0) return a;
	else return -a;
}

void calculateEnergy(Mat& image,double** energy,int row,int col)
{
	Mat imageXY8UC = image.clone();    
	Mat imageX=Mat::zeros(image.size(),CV_8UC3);  
    Mat imageY=Mat::zeros(image.size(),CV_8UC3);     
    Mat imageXY=Mat::zeros(image.size(),CV_8UC3);    
    Mat imageX8UC;  
    Mat imageY8UC;  
    for(int i=0;i<image.rows;i++)  
    {  
        for(int j=0;j<image.cols;j++)  
        {  
            //通过指针遍历图像上每一个像素  
            for (int k = 0;k < 3;++k)
	            {
		            imageX.at<Vec3b>(i,j)[k] =jueduizhi(
						(i > 0 && j < image.cols-1 ? image.at<Vec3b>(i-1,j+1)[k] : 0)
						+ (j < image.cols-1 ? image.at<Vec3b>(i,j+1)[k]*2 : 0)
						+ (i < image.rows-1 && j < image.cols-1 ? image.at<Vec3b>(i+1,j+1)[k] : 0)
						- (i > 0 && j > 0 ? image.at<Vec3b>(i-1,j-1)[k] : 0)
						- (j > 0 ? image.at<Vec3b>(i,j-1)[k]*2 : 0)
						- (i < image.rows-1 && j > 0 ? image.at<Vec3b>(i+1,j-1)[k] : 0)
					);  
		            imageY.at<Vec3b>(i,j)[k]=jueduizhi(
						(i < image.rows-1 && j > 0 ? image.at<Vec3b>(i+1,j-1)[k] : 0)
						+ (i < image.rows-1 ? image.at<Vec3b>(i+1,j)[k]*2 : 0)
						+ (i < image.rows-1 && j < image.cols-1 ? image.at<Vec3b>(i+1,j+1)[k] : 0)
						- (i > 0 && j > 0 ? image.at<Vec3b>(i-1,j-1)[k] : 0)
						- (i > 0 ? image.at<Vec3b>(i-1,j)[k]*2 : 0)
						- (i > 0 && j < image.cols-1 ? image.at<Vec3b>(i-1,j+1)[k] : 0)
					);  
	        	}
        }  
    }  
    addWeighted(imageX,0.5,imageY,0.5,0,imageXY);//融合X、Y方向    
    convertScaleAbs(imageX,imageX8UC);  
    convertScaleAbs(imageY,imageY8UC);  
    convertScaleAbs(imageXY,imageXY8UC);   //转换为8bit图像  
    // printf("%d\n", imageXY8UC.channels());
     //    for(int i=0;i<image.rows;i++)  
	    // {  
	    //     for(int j=0;j<image.cols;j++)  
	    //     {  
	    //         //通过指针遍历图像上每一个像素  
	    //         for (int k = 0;k < 3;++k)
		   //          {
			  //           imageXY8UC.at<Vec3b>(i,j)[k] =jueduizhi(
					// 		- (i > 0 ? image.at<Vec3b>(i-1,j)[k] : 0)
					// 		- (i < image.rows-1 ? image.at<Vec3b>(i+1,j)[k]*2 : 0)
					// 		- (j < image.cols-1 ? image.at<Vec3b>(i,j+1)[k] : 0)
					// 		- (j > 0 ? image.at<Vec3b>(i,j-1)[k] : 0)
					// 		+ (image.at<Vec3b>(i,j)[k]*4)
					// 	);  
		   //      	}
     //   		}  
     //   	}
	for (int i = 0;i < row;++i)
	{
		for (int j = 0;j < col;++j)
		{
			energy[i][j] = 0;
			for (int k = 0;k < 3;++k)
			{
				energy[i][j] += (int)(imageXY8UC.at<cv::Vec3b>(i,j)[k]);
			}
			// energy[i][j] += (int)(imageXY8UC.at<uchar>(i,j));
			// printf("%f ",energy[i][j] );
		}
		// printf("\n");
	}
}

void DP(Node** seam, double** energy,int row,int col)
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

void getInfo(T& picture,bool** choosed = NULL)
{
	int row = picture.pic.rows;
	int col = picture.pic.cols;
	double** energy = new double*[row];
	for (int i = 0;i < row;++i)
		energy[i] = new double[col];
	calculateEnergy(picture.pic,energy,row,col);
	if (choosed != NULL)
		for (int i = 0;i < row;++i)
			for (int j = 0;j < col;++j)
				if (choosed[i][j]) energy[i][j] = -INT_MAX;
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
	double** energyr = new double*[row];
	for (int i = 0;i < row;++i)
		energyr[i] = new double[col];
	calculateEnergy(picture.pic,energyr,row,col);
	if (choosed != NULL)
		for (int i = 0;i < col;++i)
			for (int j = 0;j < row;++j)
				if (choosed[i][j]) energyr[j][i] = -INT_MAX;
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

typedef cv::Vec3b Color;
typedef cv::Mat_<Color> Image;

struct MouseArgs // 用于和回调函数交互，至于为什么要特意攒一个struct后面会讲~
{
	Image &img; // 显示在窗口中的那张图
	std::vector<std::vector<int > > &mask; // 用户绘制的选区（删除/保留）
	Color color; // 用来高亮选区的颜色
	MouseArgs(Image &img, std::vector<std::vector<int > > &mask, const Color color)
		: img(img), mask(mask), color(color) {}
};



// 3. 定义一个回调函数，这里主要实现2个功能：
// 1） 用半透明颜色高亮选区
// 2） 将选区存到数组中
// 这个回调函数有一个固定的格式（函数名随意）：
void onMouse(int event, int x, int y, int flags, void *param);
/**
 * event: 触发回调函数的鼠标事件（如移动/点击/松开等）
 * (x, y): 鼠标事件发生的位置
 * flags: 貌似是用位运算的方式表示左/中/右三个键是否被按下
 * param: 传给回调函数的参数，可以在回调函数内部被读/写【我们将从这里获取数据】
 *
 * 以上这堆参数只有param需要自行在外部准备，其余的opencv都搞定了，咱们不用管直接用就行了
 */
 
 void onMouse(int event, int x, int y, int flags, void *param)
{
  	// C++没有类似Java的单根继承机制，为了支持多类型的交互数据这里只能传入void *再强制转换
  	// 为什么必须定义一个MouseArgs结构体：不然没法同时给回调函数传入多个数据
	MouseArgs *args = (MouseArgs *)param;

  	// 按下鼠标左键拖动时
	if ((event == CV_EVENT_MOUSEMOVE || event == CV_EVENT_LBUTTONDOWN)
	 && (flags & CV_EVENT_FLAG_LBUTTON))
	{
		int brushRadius = 10;	// 笔刷半径			
		int rows = args->img.rows, cols = args->img.cols;

      	// 以下的双重for循环遍历的是半径为10的圆形区域，实现笔刷效果
      	// 注意传回给回调函数的x, y是【窗口坐标】下的，所以y是行，x是列
		for (int i = max(0, y - brushRadius); i < min(rows, y + brushRadius); i++)
		{
			int halfChord = sqrt(pow(brushRadius, 2) - pow(i - y, 2)); // 半弦长
			for (int j = max(0, x - halfChord); j < min(cols, x + halfChord); j++)
				if (args->mask[i][j] == 0)
				{
                    // 高亮这一笔
					args->img(i, j) = args->img(i, j) * 0.7 + args->color * 0.3;
                    // 将这一笔添加到选区
					args->mask[i][j] = 1;
				}
		}
	}
}


int main(int argc,char** argv)
{
	std::string filename(argv[1]);
	Mat input = imread(filename);
	printf("%d %d\n",input.channels(),input.type() );
	Mat temp = input;
	currentMap = new int[temp.rows];
	Mat origin = input;
	int row = input.rows;
	int col = input.cols;
	if (string(argv[2]) == "cut")
	{
		int totrow= atof(argv[4])*temp.rows;
		int totcol = atof(argv[3])*temp.cols;
		vector<vector<T> > dpphoto;
		vector<T> gg(totcol + 1);
		for (int i = 0;i < totrow + 1;++i)
			dpphoto.push_back(gg);
		dpphoto[0][0].pic = input;
		dpphoto[0][0].nowgain = 0;
		getInfo(dpphoto[0][0]);
		printf("infook\n");
		for (int i = 0;i <= totrow;++i)
			for (int j = 0;j <= totcol;++j)
			{
				printf("%d %d\n",i,j );
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
		printf("233\n");
		Mat seamPic = dpphoto[totrow][totcol].pic;
		imwrite("result.jpg",dpphoto[totrow][totcol].pic);
		int r = totrow,c = totcol;
		while (r >= 0 && c >= 0)
		{
			if (r == 0 && c == 0) break;
			printf("%d %d\n",r,c);
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
		int totrow= atof(argv[4])*temp.rows;
		int totcol = atof(argv[3])*temp.cols;
		double** energy = new double*[row];
		for (int i = 0;i < row;++i)
			energy[i] = new double[col];
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
		double meiyongde = 0;
		while (totcol--)
		{
			for (int i = 0;i < row;++i)
				for (int j = 0;j < col;++j)
					seam[i][j].number = INT_MAX;
			DP(seam,energy,row,col);
			int removenewPoint = calculateMin(seam,row,col,meiyongde);
			int t = row-1;
			while (removenewPoint != -1)
			{
				energy[t][removenewPoint] = INT_MAX;
				addtime[t][removenewPoint]++;
				removenewPoint = seam[t][removenewPoint].last;
				t--;
			}
		}
		// for (int i  = 0;i < row;++i)
		// 	delete[] energy[i],seam[i],addtime[i];
		// delete[] energy,seam,addtime;
		Mat amplifyPic = Mat(temp.rows,temp.cols + atof(argv[3])*temp.cols,CV_8UC3);
		for (int i = 0;i < temp.rows;++i)
			for (int j = 0,k = 0;j < temp.cols;++j)
				for (int p = 0;p <= addtime[i][j];++p)
					amplifyPic.at<Vec3b>(i,k++) = temp.at<Vec3b>(i,j);
		amplifyPic = amplifyPic.t();
		row = amplifyPic.rows;
		col = amplifyPic.cols;
		double** energyr = new double*[row];
		for (int i = 0;i < row;++i)
			energyr[i] = new double[col];
		calculateEnergy(amplifyPic,energyr,row,col);
		Node** seamr = new Node*[row];
		for (int i = 0;i < row;++i)
			seamr[i] = new Node[col];
		unsigned short** addtimenew = new unsigned short*[row];
		for (int i = 0;i < row;++i)
			addtimenew[i] = new unsigned short[col];
		for (int i = 0;i < row;++i)
			for (int j = 0;j < col;++j)
				addtimenew[i][j] = 0;
		while (totrow--)
		{
			for (int i = 0;i < row;++i)
				for (int j = 0;j < col;++j)
					seamr[i][j].number = INT_MAX;
			DP(seamr,energyr,row,col);
			int removenewPoint = calculateMin(seamr,row,col,meiyongde);
			int t = row-1;
			while (removenewPoint != -1)
			{
				energyr[t][removenewPoint] = INT_MAX;
				addtimenew[t][removenewPoint]++;
				removenewPoint = seamr[t][removenewPoint].last;
				t--;
			}
		}
		Mat amplify = Mat(temp.cols + atof(argv[3])*temp.cols,temp.rows + atof(argv[4])*temp.rows,CV_8UC3);
		for (int i = 0;i < amplifyPic.rows;++i)
			for (int j = 0,k = 0;j < amplifyPic.cols;++j)
				if (addtimenew[i][j] == 0)
					amplify.at<Vec3b>(i,k++) = amplifyPic.at<Vec3b>(i,j);
				else
				{
					for (int p = 0;p <= addtimenew[i][j];++p)
						amplify.at<Vec3b>(i,k++) = amplifyPic.at<Vec3b>(i,j);
				}
		amplify = amplify.t();
		imshow("window",amplify);
		imwrite("result.jpg",amplify);
		waitKey(0);
	}
	else if (string(argv[2]) == "remove")
	{
		// 2. 创建一个可交互的窗口
		Image showImg = input.clone(); // 拷贝一张图用于显示（因为需要在显示的图上面高亮标注，从而造成修改）
		cv::namedWindow("Draw ROI", CV_WINDOW_AUTOSIZE); // 新建一个窗口
		vector<vector<int > > maskRemove(row, vector<int >(col, 0)); // 希望获取的待删除选区
		MouseArgs *args = new MouseArgs(showImg, maskRemove, Color(0, 0, 255)); // 攒一个MouseArgs结构体用于交互
		cv::setMouseCallback("Draw ROI", onMouse, (void*)args); // 给窗口设置回调函数

		// 拖动鼠标作画
		while (1)
		{
			cv::imshow("Draw ROI", args->img);
			// 按 esc 键退出绘图模式，获得选区
			if (cv::waitKey(100) == 27)
				break;
		}
		// maskRemove[200][400] = 1;
		bool** choosed = new bool*[row];
		for (int i = 0;i < row;++i)
			choosed[i] = new bool[col];
		for (int i = 0;i < row;++i)
			for (int j = 0;j < col;++j)
					choosed[i][j] = (maskRemove[i][j] == 1);
		while(1)
		{
			row = temp.rows;
			col = temp.cols;
			bool finished = true;
			for (int i = 0;i < row;++i)
				for (int j = 0;j < col;++j)
				{
					// printf("233%d %d\n",i,j);
					if (choosed[i][j]) finished = false;
				}
			if (finished) break;
			T nowpic;
			nowpic.pic = temp;
			getInfo(nowpic,choosed);
			if (nowpic.rseam < nowpic.cseam)
			{
				temp = nowpic.rowpic;
				row = temp.rows;
				col = temp.cols;
				printf("%d %d\n",row,col);
				bool** gg = new bool*[row];
				for (int i = 0;i < row;++i)
					gg[i] = new bool[col];
				for (int i = col-1;i;--i)
					for (int j = 0,k = 0;j < row;++j)
						if (nowpic.rowseam[col-1-i].y != j)
							gg[k++][i] = choosed[j][i];
				for (int i = 0;i < row + 1;++i)
					delete[] choosed[i];
				delete[] choosed;
				choosed = gg;
			}
			else
			{
				temp = nowpic.colpic;
				row = temp.rows;
				col = temp.cols;
				printf("%d %d\n",row,col);
				bool** gg = new bool*[row];
				for (int i = 0;i < row;++i)
					gg[i] = new bool[col];
				for (int i = row-1;i;--i)
					for (int j = 0,k = 0;j < col;++j)
						if (nowpic.colseam[row-1-i].x != j)
							gg[i][k++] = choosed[i][j];
				for (int i = 0;i < row;++i)
					delete[] choosed[i];
				delete[] choosed;
				choosed = gg;
			}
		}
		// 4. 垃圾回收
		cv::setMouseCallback("Draw ROI", NULL, NULL); // 取消回调函数
		delete args; // 垃圾回收
		// waitKey(0);

		// imshow("window",temp);
		imwrite("result.jpg",temp);
		waitKey(0);
	}
	else printf("Wrong Command!\n");
	return 0;
}

