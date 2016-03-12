package ru.spbau.mit.reptilian_detector.detector;

import java.io.*;
import java.net.*;

import org.bytedeco.javacpp.indexer.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacv.*;
import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_imgproc.*;
import static org.bytedeco.javacpp.opencv_imgcodecs.*;
import static org.bytedeco.javacpp.opencv_objdetect.*;
import static org.bytedeco.javacpp.opencv_highgui.*;


public class Main {
	
    static final String frontalfaceXmlPath = "haarcascade_frontalface_alt.xml";
    static final String inputImage = "input.jpg";
    
    public static void main(String[] args) throws Exception {
        IplImage image = cvLoadImage(args[0]);
        if(image != null){
            detectFaces(image);
            namedWindow("Result", WINDOW_NORMAL);
            cvShowImage("Result", image);
            cvWaitKey(0);
            System.out.println("OK.");
        } else {
            System.out.println("Can't load image.");
        }
    }
    
    static void detectFaces(IplImage image) throws Exception {
        CvHaarClassifierCascade cascade = new CvHaarClassifierCascade(
            cvLoad(getPathTo(frontalfaceXmlPath)));
        CvMemStorage storage = CvMemStorage.create();
        CvSeq faces = cvHaarDetectObjects(image, cascade, storage,
            1.5, 3, CV_HAAR_DO_CANNY_PRUNING);
        cvClearMemStorage(storage);
        int totalCount = faces.total();
        for(int i = 0; i < totalCount; i++){
            CvRect rect = new CvRect(cvGetSeqElem(faces,i));
            cvRectangle(image, cvPoint(rect.x(), rect.y()), 
                cvPoint(rect.width() + rect.x(), rect.height() + rect.y()), 
                CvScalar.GREEN, 2, CV_AA, 0);
        }
        
    }

    static String getPathTo(String resourceName) throws Exception{
        Main tmp = new Main();
        URL url = tmp.getClass().getClassLoader().getResource(resourceName);
        File file = Loader.extractResource(url, null, resourceName, resourceName);
        file.deleteOnExit();
        return file.getAbsolutePath();
    }
}

