package ru.spbau.mit.backend;

import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_imgproc.*;
import static org.bytedeco.javacpp.opencv_imgcodecs.*;
import static org.bytedeco.javacpp.opencv_objdetect.*;

public class Main{
	
    static final String frontalfaceXmlPath = "haarcascade_frontalface_alt.xml";
    static final String inputImage = "input.jpg";
    static final String outputImage = "output.jpg";
    static final String filePathPrefix = "C:\\Users\\ArgentumWalker\\Desktop\\AuProject\\reptilian-detector\\backend\\src\\main\\resources\\";
    
    public static void main(String[] arg){
        IplImage image = cvLoadImage(filePathPrefix+inputImage);
        if(image != null){
            detect_faces(image);
            cvSaveImage(filePathPrefix+outputImage,image);
            cvReleaseImage(image);
            System.out.println("OK.");
        } else {
            System.out.println("Can't load image.");
        }
    }
    
    static void detect_faces(IplImage image){
        CvHaarClassifierCascade cascade = new CvHaarClassifierCascade(cvLoad(filePathPrefix+frontalfaceXmlPath));
        CvMemStorage storage = CvMemStorage.create();
        CvSeq faces = cvHaarDetectObjects(image, cascade, storage, 1.5, 3, CV_HAAR_DO_CANNY_PRUNING);
        cvClearMemStorage(storage);
        int totalCount = faces.total();
        for(int i = 0; i < totalCount; i++){
            CvRect rect = new CvRect(cvGetSeqElem(faces,i));
            cvRectangle(image, cvPoint(rect.x(), rect.y(), cvPoint(rect.width() + rect.x(), rect.height() + rect.y()), CvScalar.GREEN, 4, CV_AA, 0);            
        }
    }
	
}