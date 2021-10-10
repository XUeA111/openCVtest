package lab;

import org.opencv.core.Core;

import java.io.File;

public class lab {
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static void main(String[] args) {
        /*
        String dir="C:\\Users\\Administrator\\Desktop\\data2\\smoke\\0\\background"+"\\";
        File dir1=new File(dir);
        File[] name=dir1.listFiles();
        bubbleSort(name);
        for(File i:name)
            System.out.println(i.getName());
        modifyName(name,dir);
         */
        System.out.println(0.6/0.2);
    }

    /**
     * 冒泡排序
     *
     * @param array
     * @return
     */

    public static void bubbleSort(File[] array) {
        int indexOfDot1,indexOfDot2,val1,val2;
        if (array.length == 0)
            return;
        for (int i = 0; i < array.length; i++) {
            for (int j = 0; j < array.length - 1 - i; j++) {
                indexOfDot1=array[j].getName().indexOf(".");
                val1=Integer.parseInt(array[j].getName().substring(0,indexOfDot1));
                indexOfDot2=array[j+1].getName().indexOf(".");
                val2=Integer.parseInt(array[j+1].getName().substring(0,indexOfDot2));
                if (val2 < val1) {
                    File temp = array[j + 1];
                    array[j + 1] = array[j];
                    array[j] = temp;
                }
            }
        }
    }

public static void modifyName(File[] array,String dir){
    int val=0;
    File dest;
    for(int i=0;i<array.length;i++){
            //重命名
            dest=new File(dir+val+".jpg");
            array[i].renameTo(dest);
            val++;
    }
  }
}
