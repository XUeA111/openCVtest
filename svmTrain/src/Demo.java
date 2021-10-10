import org.opencv.core.Mat;
import org.opencv.core.TermCriteria;
import org.opencv.ml.SVM;
import org.opencv.utils.Converters;

import java.util.Calendar;
import java.util.Date;
import java.util.Vector;

import static org.opencv.core.CvType.CV_32FC1;
import static org.opencv.ml.Ml.ROW_SAMPLE;

public class Demo {
    public static void main(String[] args) {
        Calendar calendar= Calendar.getInstance();
        Date time=calendar.getTime();
        Classifier clas = new Classifier();
        Mat trainingData = new Mat();
        Mat trainingImages = new Mat();
        Vector<Integer> trainingLabels = new Vector();
        //输出开始提取特征的时间
        System.out.println(time);
        //正样本集
        //设置训练正样本集目录
        clas.setDirectory("C:\\Users\\Administrator\\Desktop\\data2\\smoke\\");
        //设置该路径下的文件夹个数
        clas.setNum(5);
        clas.getTrainingFeature(trainingImages, trainingLabels,1);
        //负样本集
        //设置训练负样本集目录
        clas.setDirectory("C:\\Users\\Administrator\\Desktop\\data2\\non-smoke\\");
        //设置该路径下的文件夹个数
        clas.setNum(4);
        clas.getTrainingFeature(trainingImages, trainingLabels,-1);
        trainingImages.copyTo(trainingData);
        trainingData.convertTo(trainingData, CV_32FC1);
        //存放标记
        Mat labels;
        labels = Converters.vector_int_to_Mat(trainingLabels);
        //输出完成特征提取的时间
        System.out.println(time);

        //配置SVM训练器参数
        SVM svm = SVM.create();
        svm.setType(SVM.C_SVC);
        svm.setKernel(SVM.RBF);
        svm.setGamma(0.01);   //设置初始参数
        svm.setC(10.0);   //设置初始参数
        svm.setTermCriteria(new TermCriteria(TermCriteria.EPS, 100, 5e-3));
        //TrainData DataToBeTrained = TrainData.create(trainingData, ROW_SAMPLE, labels);
        boolean success = svm.trainAuto(trainingData,ROW_SAMPLE,labels,10);
        //保存模型
        System.out.println("training finished");
        svm.save("G:\\DataTrial\\svm.xml");

        //输出训练完成时间
        System.out.println(time);
    }
}
