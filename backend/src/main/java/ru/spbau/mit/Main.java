package ru.spbau.mit.backend;

import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_imgproc.*;
import static org.bytedeco.javacpp.opencv_imgcodecs.*;
import static org.bytedeco.javacpp.opencv_objdetect.*;
import static org.bytedeco.javacpp.opencv_highgui.*;

public class Main{
	
    static final String frontalfaceXmlPath = "haarcascade_frontalface_alt.xml";
    static final String inputImage = "input.jpg";
    
    public static void main(String[] arg){
        final ResourceMenager resources = new ResourceMenager();
        IplImage image = cvLoadImage(resources.getPathTo(inputImage));
        if(image != null){
            detect_faces(image);
            namedWindow("Result", WINDOW_NORMAL);
            cvShowImage("Result",image);
            cvWaitKey(0);
            System.out.println("OK.");
        } else {
            System.out.println("Can't load image.");
        }
    }
    
    static void detect_faces(IplImage image){
        final ResourceMenager resources = new ResourceMenager();
        CvHaarClassifierCascade cascade = new CvHaarClassifierCascade(cvLoad(resources.getPathTo(frontalfaceXmlPath)));
        CvMemStorage storage = CvMemStorage.create();
        CvSeq faces = cvHaarDetectObjects(image, cascade, storage, 1.5, 3, CV_HAAR_DO_CANNY_PRUNING);
        cvClearMemStorage(storage);
        int totalCount = faces.total();
        for(int i = 0; i < totalCount; i++){
            CvRect rect = new CvRect(cvGetSeqElem(faces,i));
            cvRectangle(image, cvPoint(rect.x(), rect.y()), cvPoint(rect.width() + rect.x(), rect.height() + rect.y()), CvScalar.GREEN, 4, CV_AA, 0);            
        }
    }
	
}

class ResourceMenager{
    public String getPathTo(String resourceName){
        String path = getClass().getResource("/" + resourceName).getPath();
        return path.substring(1, path.length()); //I don't know, why i need to do this. Maybe it depends on OS
    }
}
