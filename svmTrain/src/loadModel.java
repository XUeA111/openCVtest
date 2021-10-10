import HSV.extractHSV;
import LBP.LBPH;
import LBP.extractLBP;
import Wavelet.HaarWavelet2D;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.ml.SVM;

import java.io.File;
import java.util.Arrays;
import java.util.List;

import static org.opencv.core.CvType.CV_32FC1;
import static org.opencv.imgcodecs.Imgcodecs.imread;

public class loadModel {
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static void main(String[] args) {
        int response;  //存放svm判断结果
        int totalPositive=713, totalNegative=510;   //正类图片总数、负类图片总数
        int positiveFolder = 8, negativeFolder = 6;   //正类文件夹总数、负类文件夹总数
        int TruePositives = 0;   //真正类：正样本在测试过程中被正确分类为正样本的个数
        int TrueNegatives = 0;   //真负类：负样本在测试过程中被正确分类为负样本的个数
        int FalsePositives = 0; //假正类：负样本在测试过程中被错误分类为正样本的个数
        int FalseNegatives = 0;  //假负类：正样本在测试过程中被错误分类为负样本的个数
        double FPR = 0;  //误检率：假正类/实际负类
        double FNR = 0;  //漏检率：假负类/实际正类
        double TPR = 0;  //正检测率：真正类/实际正类
        double TNR = 0;  //真负类
        Mat dstImage = new Mat();
        Mat SrcImage = new Mat();     //存放读入的测试集图片
        Mat backgroundPic = new Mat();  //存放背景图片
        String outputPath; //测试集样本的路径
        String backgroundPath;  //测试集背景的路径
        String modelPath = "C:\\Users\\Administrator\\Desktop\\Academic Research\\xml\\20210220.xml";
        SVM svm = SVM.load(modelPath);
        String picPathPositive = "C:\\Users\\Administrator\\Desktop\\data2\\test\\smoke test";
        String picPathNegative = "C:\\Users\\Administrator\\Desktop\\data2\\test\\non-smoke test";
        String suffix = ".jpg";

        //检测正样本测试集
        for (int i = 0; i < positiveFolder; i++) {
            //背景图片目录
            String backgroundDir = picPathPositive +"\\"+ i + "\\background\\";

            //样本图片目录
            String outputDir = picPathPositive +"\\"+ i + "\\output\\";
            File dir = new File(outputDir);
            String[] name = dir.list();
            //文件夹中图片的个数
            int picNum = name.length;
            for (int j = 0; j < picNum; j++) {
                //背景图片路径
                String backgroundPicLoca = backgroundDir + j + suffix;
                //样本图片路径
                String outputPicLoca = outputDir + j + suffix;
                backgroundPic = imread(backgroundPicLoca);
                SrcImage = imread(outputPicLoca);
                //提取烟雾图像HSV特征向量
                Mat resultHSV = extractHSV.getHSV(SrcImage);
                //提取烟雾图像LBPH特征向量
                extractLBP.getUniformLBP(SrcImage, dstImage, 1, 8);
                Mat resultLPBH = LBPH.getLBPH(dstImage, 9, 2, 2, true);
                //提取烟雾图像2维哈尔小波变换特征向量
                Mat resultHaarWavelet = HaarWavelet2D.getRatio(backgroundPic, SrcImage);
                List<Mat> feature = Arrays.asList(resultHSV, resultLPBH, resultHaarWavelet);
                Mat dst = new Mat();
                Core.hconcat(feature, dst);
                dst.convertTo(dst, CV_32FC1);
                response = (int) svm.predict(dst);
                System.out.println("正样本：" + response);
                if (response == 1)
                    TruePositives++;
                else
                    FalseNegatives++;
           }
        }

        //检测负样本测试集
        for (int i = 0; i < negativeFolder; i++) {
            //背景图片目录
            String backgroundDir = picPathNegative +"\\"+ i + "\\background\\";
            //样本图片目录
            String outputDir = picPathNegative +"\\"+ i + "\\output\\";
            File dir = new File(outputDir);
            String[] name = dir.list();
            //文件夹中图片的个数
            int picNum = name.length;
            for (int j = 0; j < picNum; j++) {
                //背景图片路径
                String backgroundPicLoca = backgroundDir + j + suffix;
                //样本图片路径
                String outputPicLoca = outputDir + j + suffix;
                backgroundPic = imread(backgroundPicLoca);
                SrcImage = imread(outputPicLoca);
                //提取烟雾图像HSV特征向量
                Mat resultHSV = extractHSV.getHSV(SrcImage);
                //提取烟雾图像LBPH特征向量
                extractLBP.getUniformLBP(SrcImage, dstImage, 1, 8);
                Mat resultLPBH = LBPH.getLBPH(dstImage, 9, 2, 2, true);
                //提取烟雾图像2维哈尔小波变换特征向量
                Mat resultHaarWavelet = HaarWavelet2D.getRatio(backgroundPic, SrcImage);
                List<Mat> feature = Arrays.asList(resultHSV, resultLPBH, resultHaarWavelet);
                Mat dst = new Mat();
                Core.hconcat(feature, dst);
                dst.convertTo(dst, CV_32FC1);
                dst.reshape(1, 1);
                response = (int) svm.predict(dst);
                System.out.println("负样本" + response);
                if (response == -1)
                    TrueNegatives++;
                else
                    FalsePositives++;
            }
        }

        FPR = FalsePositives * 1.0 / totalNegative;
        FNR = FalseNegatives * 1.0 / totalPositive;
        TPR = TruePositives * 1.0 / totalPositive;
        TNR = TrueNegatives * 1.0 / totalNegative;

        System.out.println("FPR=" + FPR);
        System.out.println("FNR=" + FNR);
        System.out.println("TPR=" + TPR);
        System.out.println("TNR=" + TNR);
    }
}

