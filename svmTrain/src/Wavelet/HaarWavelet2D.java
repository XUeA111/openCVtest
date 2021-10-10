package Wavelet;

import org.opencv.core.CvType;
import org.opencv.core.Mat;

import java.util.ArrayList;

public class HaarWavelet2D {
    /**
     * 计算背景图片与烟雾图片的高、低频能量之比并提取该特征
     * 提取烟雾图片中的算术均值、峰值比、标准差、偏态、峰度
     * 返回两比值和其余特征值的特征向量
     */
    public static Mat getRatio(Mat background,Mat smoke){
        Mat backgroundC1=new Mat(),smokeC1=new Mat();
        background.copyTo(backgroundC1);
        smoke.copyTo(smokeC1);
        //转换为单通道图片
        backgroundC1=backgroundC1.reshape(1);
        smokeC1=smokeC1.reshape(1);
        //Background、Smoke存放二维Haar变换分解得到的高低频子图
        ArrayList<Mat> Background=getHaarWavelet2D(backgroundC1,false);
        ArrayList<Mat> Smoke=getHaarWavelet2D(smokeC1,true);
        Mat feature=getFeatureVector(Background,Smoke);
        //释放内存
        backgroundC1.release();
        smokeC1.release();
        for(Mat tmp:Background){
            tmp.release();
        }
        for(Mat tmp:Smoke){
            tmp.release();
        }
        return feature;
    }

    /**
     * 计算图像的高频总能量、低频总能量
     * @param src
     */
    private static ArrayList<Mat> getHaarWavelet2D(Mat src,boolean flag){
        Mat imgtmp=new Mat();
        src.copyTo(imgtmp);
        //处理的图像为单通道
        imgtmp.convertTo(imgtmp,CvType.CV_32FC1);
        int width=imgtmp.rows();
        int height=imgtmp.cols();
        double pixel1st,pixel2nd;
        //定义分解的深度
        int depth=3;
        int depthCount=1;
        Mat tmp=Mat.ones(width,height, CvType.CV_32FC1);
        Mat wavelet=Mat.ones(width,height, CvType.CV_32FC1);

        while(depthCount<=depth){
            width=src.rows()/depthCount;
            height=src.cols()/depthCount;
            //对行进行一维Haar小波变换
            for(int i=0;i<width;i++) {
                for (int j = 0; j < height / 2; j++) {
                    pixel1st = (imgtmp.get(i, 2 * j)[0] + imgtmp.get(i, 2 * j + 1)[0]) / 2;
                    tmp.put(i, j, pixel1st);
                    pixel2nd = (imgtmp.get(i, 2 * j)[0] - imgtmp.get(i, 2 * j + 1)[0]) / 2;
                    tmp.put(i, j + height / 2, pixel2nd);
                }
            }
            //对列进行一维Haar小波变换
            for(int i=0;i<width/2;i++) {
                for (int j = 0; j < height; j++) {
                    pixel1st = (tmp.get(2 * i, j)[0] + tmp.get(2 * i + 1, j)[0]) / 2;
                    wavelet.put(i, j, pixel1st);
                    pixel2nd = (tmp.get(2 * i, j)[0] - tmp.get(2 * i + 1, j)[0]) / 2;
                    wavelet.put(i + width / 2, j, pixel2nd);
                }
            }
            imgtmp=wavelet;
            depthCount++;
        }

        //变换结果的行列值
        width=imgtmp.rows();
        height=imgtmp.cols();
        //低频子图LL
        Mat LL=imgtmp.colRange(0,height/2);
        LL=LL.rowRange(0,width/2);
        //3个高频子图HL、LH、HH
        Mat HL=imgtmp.colRange(height/2,height);
        HL=HL.rowRange(0,width/2);
        Mat LH=imgtmp.colRange(0,height/2);
        LH=LH.rowRange(width/2,width);
        Mat HH=imgtmp.colRange(height/2,height);
        HH=HH.rowRange(width/2,width);

        //将图像的低频、高频子图存入结果链表
        ArrayList<Mat> result=new ArrayList<Mat>();
        result.add(LL);
        result.add(HL);
        result.add(LH);
        result.add(HH);

        return result;
    }

    /**
     * 计算高频子图的5个特征
     */
    private static Mat getFeatureVector(ArrayList<Mat> Background,ArrayList<Mat> Smoke){
        Mat feature=new Mat();
        //存放子图的算术均值
        double aver;
        //提取3幅高频子图的算术均值、标准差、峰度、偏态、峰值
        Mat featureRest5=new Mat(1,5,CvType.CV_32FC1);
        featureRest5=featureRest5.reshape(1);
        for(int i=0;i<3;i++) {
            aver = getAveragePixel(Smoke.get(i + 1));
            featureRest5.put(0, 0, aver);
            featureRest5.put(0, 1, getStanDeviation(Smoke.get(i + 1), aver));
            featureRest5.put(0,2,getKurtosis(Smoke.get(i+1),aver));
            featureRest5.put(0,3,getSkewness(Smoke.get(i+1),aver));
            featureRest5.put(0,4,getPeak(Smoke.get(i+1))/getPeak(Background.get(i+1)));
            feature.push_back(featureRest5);
        }
        double ratioEL=getLowFrequencyEnerge(Smoke.get(0))/getLowFrequencyEnerge(Background.get(0));
        double ratioEH=getHighFrequencyEnerge(Smoke.get(1),Smoke.get(2),Smoke.get(3))/getHighFrequencyEnerge(Background.get(1),Background.get(2),Background.get(3));
        Mat tmp=new Mat(1,5,CvType.CV_32FC1);
        tmp=tmp.reshape(1);
        tmp.put(0,0,ratioEL);
        tmp.put(0,1,ratioEH);
        feature.push_back(tmp);
        feature=feature.reshape(1,1);
        feature=feature.colRange(0,17);
        return feature;
    }

    /**
     * 计算低频子图总能量
     */
    private static double getLowFrequencyEnerge(Mat LL){
        //低频总能量
        double EL=0;
        //计算图像低频总能量
        for(int i=0;i<LL.rows();i++) {
            for (int j = 0; j < LL.cols(); j++) {
                EL += Math.pow(LL.get(i, j)[0], 2);
            }
        }
        return EL;
    }

    /**
     * 计算低频子图总能量
     */
    private static double getHighFrequencyEnerge(Mat HL,Mat LH,Mat HH){
        //高频总能量
        double EH=0;
        //计算高频总能量HL
        for(int i=0;i<HL.rows();i++) {
            for (int j = 0; j < HL.cols(); j++) {
                EH += Math.pow(HL.get(i, j)[0], 2);
            }
        }
        //计算高频总能量LH
        for(int i=0;i<LH.rows();i++) {
            for (int j = 0; j < LH.cols(); j++) {
                EH +=Math.pow(LH.get(i, j)[0], 2);
            }
        }
        //计算高频总能量
        for(int i=0;i<HH.rows();i++) {
            for (int j = 0; j < HH.cols(); j++) {
                EH += Math.pow(HH.get(i, j)[0], 2);
            }
        }

        return EH;
    }

    /**
     * 计算高频子图中的算术均值
     */
    private static double getAveragePixel(Mat subimage){
        Mat tmp=new Mat();
        subimage.copyTo(tmp);
        tmp=tmp.reshape(1);
        double sum=0;
        for(int i=0;i<tmp.rows();i++) {
            for (int j = 0; j < tmp.cols(); j++) {
                sum += tmp.get(i, j)[0];
            }
        }
        return sum*1.0/(tmp.rows()*tmp.cols());
    }

    /**
     * 计算高频子图中的标准差
     */
    private static double getStanDeviation(Mat subimage,double aver){
        Mat tmp=new Mat();
        subimage.copyTo(tmp);
        tmp=tmp.reshape(1);
        double sum=0,value;
        for(int i=0;i<tmp.rows();i++) {
            for (int j = 0; j < tmp.cols(); j++) {
                value=subimage.get(i,j)[0];
                sum += Math.pow(value-aver,2);
            }
        }
        return Math.sqrt(sum/(tmp.rows()*tmp.cols()));
    }

    /**
     * 计算高频子图峰度
     */
    private static double getKurtosis(Mat subimage,double aver) {
        Mat tmp = new Mat();
        subimage.copyTo(tmp);
        tmp = tmp.reshape(1);
        double sumNumerator = 0, sumDenominator = 0, valueNume, valueDeno;
        for (int i = 0; i < tmp.rows(); i++) {
            for (int j = 0; j < tmp.cols(); j++) {
                valueNume = Math.pow(tmp.get(i, j)[0] - aver, 4);
                sumNumerator += valueNume;
                valueDeno = Math.pow(tmp.get(i, j)[0] - aver, 2);
                sumDenominator += valueDeno;
            }
        }
        sumDenominator *= sumDenominator;

        return tmp.rows() * tmp.cols() * sumNumerator / sumDenominator;
    }

    /**
     * 计算高频子图偏态
     */
    private static double getSkewness(Mat subimage,double aver){
        Mat tmp = new Mat();
        subimage.copyTo(tmp);
        tmp = tmp.reshape(1);
        double sumNumerator = 0, sumDenominator = 0, valueNume, valueDeno;
        for (int i = 0; i < tmp.rows(); i++) {
            for (int j = 0; j < tmp.cols(); j++) {
                valueNume = Math.pow(tmp.get(i, j)[0] - aver, 3);
                sumNumerator += valueNume;
                valueDeno = Math.pow(tmp.get(i, j)[0] - aver, 2);
                sumDenominator += valueDeno;
            }
        }
        sumDenominator=Math.pow(sumDenominator,1.5);

        return Math.sqrt(tmp.rows()*tmp.cols())*sumNumerator/sumDenominator;
    }

    /**
     * 返回高频子图中的峰值
     */
    private static double getPeak(Mat subimage){
        Mat tmp=new Mat();
        subimage.copyTo(tmp);
        tmp=tmp.reshape(1);
        double val=subimage.get(0,0)[0];
        for(int i=0;i<tmp.rows();i++) {
            for (int j = 0; j < tmp.cols(); j++) {
                if(val<subimage.get(i,j)[0]){
                    val=subimage.get(i,j)[0];
                }
            }
        }
        return val;
    }
}
