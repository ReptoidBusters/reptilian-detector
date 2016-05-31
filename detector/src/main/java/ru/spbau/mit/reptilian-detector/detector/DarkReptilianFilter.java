package ru.spbau.mit.reptilian_detector.detector;

import java.util.ArrayList;
import org.bytedeco.javacv.*;
import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_imgproc.*;
import static org.bytedeco.javacpp.opencv_imgcodecs.*;
import static org.bytedeco.javacpp.opencv_highgui.*;

public class DarkReptilianFilter implements IFilter {
    static final String TOP_FACE_IMAGE_NAME 
            = "filters_data/DarkReptilianFilter/image.jpg";
            
    private Mat faceTopImage;
    private Mat featuresOnImage;
    
    DarkReptilianFilter() throws Exception{
        reloadImages();
        featuresOnImage = new Mat( 
                510f, 960f,
                1000f, 960f,
                760f, 1255f).reshape(2, 3); //With nose
    }
    
    @Override
    public void applyFilter(Mat image, ArrayList<Face> faces) {
        for (Face f : faces) {
            f.applyFilter(this);
        }
    }
    
    @Override
    public void applyEyeFilter(Mat face, Mat skinMask, TransformationFacePointsCollection points, Rect eye) {
    }
    @Override
    public void applyNoseFilter(Mat face, Mat skinMask, TransformationFacePointsCollection points, Rect nose) {
    }
    @Override
    public void applyMouthFilter(Mat face, Mat skinMask, TransformationFacePointsCollection points, Rect mouth) {
    }
    @Override
    public void applyFaceFilter(Mat face, Mat skinMask, TransformationFacePointsCollection points){
        final Mat transform = getAffineTransform(featuresOnImage, points.getAffinePoints());
        final Mat transformedTopFaceImage = new Mat(face.size(), face.type());
        
        warpAffine(faceTopImage, transformedTopFaceImage, transform, face.size());
        
        final Mat grayFace = new Mat();
        cvtColor(face, grayFace, CV_BGR2GRAY);
        putShadows(transformedTopFaceImage, grayFace);
        double koef = 0.5;
        addWeighted(transformedTopFaceImage, koef, face, 1-koef, 0, transformedTopFaceImage);
        
        transformedTopFaceImage.copyTo(face, skinMask);
    }
    
    public void reloadImages() {
        try {
            Mat tmp = imread(ResourceManager.getPath(TOP_FACE_IMAGE_NAME));
            faceTopImage = tmp;
        }
        catch (Exception e) {
            
        }
    }
    
    public void putShadows(Mat image, Mat shadows) {
        threshold(shadows, shadows, 200d, 255d, THRESH_TRUNC);
        final Mat shadowsRGB = new Mat();
        cvtColor(shadows, shadowsRGB, CV_GRAY2BGR);
        Mat cpImage = new Mat(image);
        addWeighted(image, 1.0, shadowsRGB, 0.5, -127.0, cpImage);
        cpImage.copyTo(image);
    }
}
