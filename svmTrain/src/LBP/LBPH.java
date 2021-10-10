package LBP;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.util.Arrays;

import static org.opencv.core.CvType.CV_32FC1;

public class LBPH {
    public static Mat getLBPH(Mat src,int numPatterns,int gridX,int gridY,boolean normed){
        //每一块的宽度
        int width=src.cols()/gridX;
        //每一块的高度
        int height=src.rows()/gridY;
        //定义LBPH的行和列，grid_x*grid_y表示将图像分割成这么些块，numPatterns表示LBP值的模式种类
        //每一行为一块分割图像的统计直方图
        Mat result = Mat.zeros(gridX * gridY,numPatterns,CV_32FC1);
        if(src.empty())
        {
            return result.reshape(1,1);
        }
        int resultRowIndex = 0;
        for(int i=0;i<gridX;i++)
            for(int j=0;j<gridY;j++){
                //图像分块
                Mat src_cell =new Mat(src,new Range(i*height,(i+1)*height),new Range(j*width,(j+1)*width));
                //计算直方图
                Mat hist_cell = getLocalRegionLBPH(src_cell,0,(numPatterns-1),true);
                //将直方图放到result中
                Mat rowResult = result.row(resultRowIndex);
                hist_cell.reshape(1,1).convertTo(rowResult,CV_32FC1);
                resultRowIndex++;
            }
        //整个图像的LBPH特征向量的大小为1*（numPatterns*4),此处为9*4=36
        return result.reshape(1,1);
    }

    //计算一个LBP特征图像块的直方图
    private static Mat getLocalRegionLBPH(final Mat src,int minValue,int maxValue,boolean normed)
    {
        //Mat中元素总数
        int totalNum=src.rows()*src.cols();
        //输出统计直方图的矩阵
        Mat result=new Mat();
        //计算得到直方图bin的数目，直方图数组的大小
        MatOfInt histSize=new MatOfInt(maxValue - minValue + 1);
        //定义直方图每一维的bin的变化范围,(0,60)
        MatOfFloat ranges=new MatOfFloat((float)minValue,(float)(maxValue+1));
        //计算直方图，src是要计算直方图的图像
        //Mat()是要用的掩模，result为输出的直方图，1为输出的直方图的维度，histSize直方图在每一维的变化范围
        //ranges，所有直方图的变化范围（起点和终点）
        Imgproc.calcHist(Arrays.asList(src),new MatOfInt(0),new Mat(),result,histSize,ranges);
        //用于归一化统计直方图
        Mat norm=new Mat();
        Core.multiply(Mat.ones(result.rows(),result.cols(),CV_32FC1),Mat.ones(result.rows(),result.cols(),CV_32FC1),norm,totalNum);
        //归一化
        if(normed)
        {
            Mat temp=new Mat();
            Core.divide(result,norm,temp,1.0);
            result=temp;
        }
        //释放内存
        histSize.release();
        ranges.release();
        norm.release();
        //结果表示成只有1行的矩阵
        return result.reshape(1,1);
    }
}
