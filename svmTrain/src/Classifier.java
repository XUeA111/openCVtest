import HSV.extractHSV;
import LBP.LBPH;
import LBP.extractLBP;
import Wavelet.HaarWavelet2D;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.TermCriteria;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.ml.SVM;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Vector;

import static org.opencv.core.CvType.CV_32FC1;

public class Classifier {
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    private String directory; //数据集地址
    private int num;  //目录下存放图片的文件夹的个数

    /**
     * 设置数据集的路径
     */
    public void setDirectory(String location) {
        directory = location;
    }

    /**
     * 设置目录下的文件夹的个数
     */
    public void setNum(int Num) {
        num = Num;
    }

    /**
     * 提取训练集特征向量
     * 设置标志
     *
     * @param trainingImages
     * @param trainingLabels
     */
    public void getTrainingFeature(Mat trainingImages, Vector<Integer> trainingLabels, int label) {
        Mat dstImage = new Mat();
        String suffix = ".jpg";

        for (int i = 0; i < num; i++) {
            //背景图片目录
            String backgroundDir = directory + i + "\\background\\";
            //样本图片目录
            String outputDir = directory + i + "\\output\\";
            File dir = new File(outputDir);
            String[] name = dir.list();
            //文件夹中图片的个数
            int picNum = name.length;
            for (int j = 0; j < picNum; j++) {
                //背景图片路径
                String backgroundPicLoca = backgroundDir + j + suffix;
                //样本图片路径
                String outputPicLoca = outputDir + j + suffix;
                //背景图片
                Mat backgroundPic = Imgcodecs.imread(backgroundPicLoca);
                //样本图片
                Mat outputPic = Imgcodecs.imread(outputPicLoca);
                //提取样本HSV特征向量
                Mat resultHSV = extractHSV.getHSV(outputPic);
                //提取烟雾图像LBPH特征向量
                extractLBP.getUniformLBP(outputPic, dstImage, 1, 8);
                Mat resultLPBH = LBPH.getLBPH(dstImage, 9, 2, 2, true);
                //提取烟雾图像2维哈尔小波变换特征向量
                Mat resultHaarWavelet = HaarWavelet2D.getRatio(backgroundPic, outputPic);
                List<Mat> feature = Arrays.asList(resultHSV, resultLPBH, resultHaarWavelet);
                Mat dst = new Mat();
                Core.hconcat(feature, dst);
                //输出特征向量
                System.out.println(dst.dump());
                trainingImages.push_back(dst);
                trainingLabels.add(label);
            }
        }
    }

    /**
     * 提取测试集的特征向量
     * 用于分类器的判断
     */
    /*
    public Mat getTestingFeature(Mat testingImages, int totalPositive, int totalNegative) {
        int response;    //存放svm判断的结果
        Mat dstImage = new Mat();
        String suffix = ".jpg";
        String modelPath = "G:\\DataTrial\\svm.xml";
        SVM svm = SVM.load(modelPath);

        for (int i = 0; i < num; i++) {
            //背景图片目录
            String backgroundDir = directory + i + "\\background\\";
            //样本图片目录
            String outputDir = directory + i + "\\output\\";
            File dir = new File(outputDir);
            String[] name = dir.list();
            //文件夹中图片的个数
            int picNum = name.length;
            for (int j = 0; j < picNum; j++) {
                //背景图片路径
                String backgroundPicLoca = backgroundDir + j + suffix;
                //样本图片路径
                String outputPicLoca = outputDir + j + suffix;
                //背景图片
                Mat backgroundPic = Imgcodecs.imread(backgroundPicLoca);
                //样本图片
                Mat outputPic = Imgcodecs.imread(outputPicLoca);
                //提取样本HSV特征向量
                Mat resultHSV = extractHSV.getHSV(outputPic);
                //提取烟雾图像LBPH特征向量
                extractLBP.getUniformLBP(outputPic, dstImage, 1, 8);
                Mat resultLPBH = LBPH.getLBPH(dstImage, 9, 2, 2, true);
                //提取烟雾图像2维哈尔小波变换特征向量
                Mat resultHaarWavelet = HaarWavelet2D.getRatio(backgroundPic, outputPic);
                List<Mat> feature = Arrays.asList(resultHSV, resultLPBH, resultHaarWavelet);
                Mat dst = new Mat();
                Core.hconcat(feature, dst);
                dst.convertTo(dst, CV_32FC1);
                response = (int) svm.predict(dst);

            }
        }
        return dst;
    }
     */

    /**
     * 释放内存资源
     */
    public void release() {
    }
}