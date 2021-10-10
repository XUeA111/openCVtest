package LBP;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;

import java.lang.reflect.Array;
import java.util.Arrays;

import static java.lang.StrictMath.*;

public class extractLBP {
    public static void getUniformLBP(Mat src, Mat dst,int radius,int neighbors){
        Mat src1= new Mat();
        src.copyTo(src1);
        //将图像转换为单通道
        src1=src1.reshape(1);
        dst.create(src1.rows()-2*radius,src1.cols()-2*radius, CvType.CV_32FC1);
        //清零输出矩阵dst
        dst.setTo(new Scalar(0));
        int temp=1;
        //LBP特征值对应图像灰度编码表，直接默认采样点为8位
        int[] table=new int[256];
        //存放中心点像素值
        double center;
        for(int i=0;i<table.length;i++){
            if(getHopTimes(i)<3)
            {
                table[i] = temp;
                temp++;
            }
        }
        //是否进行等价模式编码的标志
        boolean flag=false;
        for(int k=0;k<neighbors;k++) {
            //用于存放移位后的值
            char[] lbpCode = new char[]{'0', '0', '0', '0', '0', '0', '0', '0'};
            if (k == neighbors - 1)
                flag = true;
            //计算采样点对于中心点坐标的偏移量rx，ry
            double rx = radius * cos(2.0 * PI * k / neighbors);
            double ry = radius * sin(2.0 * PI * k / neighbors);
            //为双线性插值做准备
            //对采样点偏移量分别进行上下取整
            int x1 = (int) floor(rx);
            int x2 = (int) (ceil(rx));
            int y1 = (int) (floor(ry));
            int y2 = (int) (ceil(ry));
            //将坐标偏移量映射到0-1之间
            double tx = rx - x1;
            double ty = ry - y1;
            //根据0-1之间的x，y的权重计算公式计算权重，权重与坐标具体位置无关，与坐标间的差值有关
            double w1 = (1 - tx) * (1 - ty);
            double w2 = tx * (1 - ty);
            double w3 = (1 - tx) * ty;
            double w4 = tx * ty;
            for (int i = radius; i < src1.rows() - radius; i++)
                for (int j = radius; j < src1.cols() - radius; j++) {
                    center = src1.get(i, j)[0];
                    //根据双线性插值公式计算第k个采样点的灰度值
                    double neighbor = src1.get(i + x1, j + y1)[0] * w1 + src1.get(i + x1, j + y2)[0] * w2
                            + src1.get(i + x2, j + y1)[0] * w3 + src1.get(i + x2, j + y2)[0] * w4;
                    //LBP特征图像的每个邻居的LBP值累加，累加通过与操作完成，对应的LBP值通过移位取得
                    if (neighbor > center)
                        lbpCode[k] = '1';
                    else
                        lbpCode[k] = '0';
                    int tmp=charArrayToInt(lbpCode);
                    int temp2 = (int) dst.get(i - radius, j - radius)[0];
                    temp2 |= tmp;
                    dst.put(i - radius, j - radius, temp2);
                    if (flag) {
                        int index=(int) dst.get(i - radius, j - radius)[0];
                        double val=table[index];
                        dst.put(i - radius, j - radius, val);
                    }
                }
        }
            //进行旋转不变处理
            for(int i=0;i<dst.rows();i++)
                for(int j=0;j<dst.cols();j++){
                    int currentValue=(int)dst.get(i,j)[0];
                    int minValue=currentValue;
                    for(int k=1;k<neighbors;k++){
                        int temp2=(currentValue>>(neighbors-k))|(currentValue<<k);
                        if(temp2<minValue)
                            minValue=temp2;
                    }
                    dst.put(i,j,minValue);
                }
        }

    //计算跳变次数
    private static int getHopTimes(int n)
    {
        int count = 0;
        //获取n对应的二进制表示
        char[] binary=Integer.toBinaryString(n).toCharArray();
        char[] code={'0','0','0','0','0','0','0','0'};
        //使得到的二进制字符数组占满8位
        if(n>=128)
            code=binary;
        else{
            for(int i=0;i<binary.length;i++){
                code[8-binary.length+i]=binary[i];
            }
        }
        /*for(int i=7,j=1;i>=0&&binary.length-j>=0;i--,j++){
            code[i]=binary[binary.length-j];
        }
        */
        //判断相邻两位是否相等
        for(int i=0;i<8;i++)
        {
            if(code[i] != code[(i+1)%8])
            {
                count++;
            }
        }
        return count;
    }

    /**
     * 将8位字符数组转换对应的int型的值
     * @param array
     * @return
     */
    private static int charArrayToInt(char[] array){
        int value=0;
        for(int i=0;i<array.length;i++){
            if(array[i]=='1'){
                value+=Math.pow(2,array.length-1-i);
            }
        }
        return value;
    }
}
