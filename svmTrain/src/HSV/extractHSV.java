package HSV;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.util.Arrays;

import static org.opencv.imgproc.Imgproc.COLOR_RGB2HSV;

public class extractHSV {
    public static Mat getHSV(Mat srcImage){
        if(srcImage==null||srcImage.channels()<3)
            new Exception("Error!");
        //输出矩阵result
        Mat result=new Mat(6,5, CvType.CV_32FC1);
        Mat srcImageHSV=new Mat();
        Imgproc.cvtColor(srcImage,srcImageHSV,COLOR_RGB2HSV);
        //H、S两个维度对应直方图的bin的数量
        int hBins=3,sBins=4;
        MatOfInt histSize=new MatOfInt(hBins,sBins);
        //H:[0,179],S:[0,255]
        MatOfFloat ranges=new MatOfFloat(0f,180f,0f,256f);
        MatOfInt channels=new MatOfInt(0,1);
        //mask:new Mat()
        Imgproc.calcHist(Arrays.asList(srcImageHSV),channels,new Mat(),result,histSize,ranges);
        //归一化
        Core.normalize(result,result,0,1, Core.NORM_MINMAX);
        result=result.reshape(1,1);
        //释放内存
        srcImageHSV.release();
        histSize.release();
        ranges.release();
        channels.release();
        return result;
    }
}
