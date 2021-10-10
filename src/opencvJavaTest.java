import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import static org.opencv.imgcodecs.Imgcodecs.imread;
import static org.opencv.imgcodecs.Imgcodecs.imwrite;
import static org.opencv.imgproc.Imgproc.cvtColor;

public class opencvJavaTest {
    static { System.loadLibrary(Core.NATIVE_LIBRARY_NAME); };

    /**
     * 数据准备
     * @param args
     */
    public static void main(String[] args) {
        String ad;
        int  filename = 0,filenum=0;
        Mat img=imread("G:\\trainData\\digits.png");
        Mat gray=img;
        cvtColor(img, gray, Imgproc.COLOR_BGR2GRAY);
        int b = 20;
        int m = gray.rows() / b;   //原图为1000*2000
        int n = gray.cols() / b;   //裁剪为5000个20*20的小图块

        for (int i = 0; i < m; i++)
        {
            int offsetRow = i*b;  //行上的偏移量
            if(i%5==0&&i!=0)
            {
                filename++;
                filenum=0;
            }
            for (int j = 0; j < n; j++)
            {
                int offsetCol = j*b; //列上的偏移量
                ad=String.format("G:\\trainData\\%d\\%d.jpg",filename,filenum++);
                //截取20*20的小块
                Mat tmp;
                tmp=gray.rowRange(offsetRow, offsetRow + b);
                tmp=tmp.colRange(offsetCol, offsetCol + b);
                imwrite(ad,tmp);
            }
        }
    }
}
