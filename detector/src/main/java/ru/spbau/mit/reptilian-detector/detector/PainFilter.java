package ru.spbau.mit.reptilian_detector.detector;

import java.util.ArrayList;
import org.bytedeco.javacv.*;
import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_imgproc.*;
import static org.bytedeco.javacpp.opencv_imgcodecs.*;
import static org.bytedeco.javacpp.opencv_highgui.*;

public class PainFilter implements IFilter {
    static final String FACE_PAIN_NAME 
            = "filters_data/Pain/FacePain.jpg";
    static final String BACK_PAIN_NAME 
            = "filters_data/Pain/trees.jpeg";
            
    private Mat facePain;
    private Mat backPain;
    private Mat featuresOnImage;
    
    PainFilter() throws Exception{
        reloadImages();
        featuresOnImage = new Mat( 
                100f, 200f,
                260f, 200f,
                180f, 300f).reshape(2, 3); //With nose
    }
    
    @Override
    public void applyFilter(Mat image, ArrayList<Face> faces) {
        for (Face f : faces) {
            f.applyFilter(this);
        }
        Mat resizedBackPain = new Mat(image.size(), image.type());
        resize(backPain, resizedBackPain, resizedBackPain.size());
        addWeighted(image, 0.65, resizedBackPain, 0.25, 0, image);
        Mat resultImage = new Mat();
        Mat grayImage = new Mat();
        cvtColor(image, grayImage, CV_BGR2GRAY);
        cvtColor(grayImage, resultImage, CV_GRAY2BGR);
        resultImage.copyTo(image);
    }
    
    @Override
    public void applyEyeFilter(Mat face, Mat skinMask, TransformationFacePointsCollection points, Rect eye) {
        /*Mat tmpEye = new Mat(face, eye);
        Mat eyeMask = new Mat(skinMask, eye);
        Scalar sc = new Scalar(3);
        sc.put(0,0);
        sc.put(1,0);
        sc.put(2,0);
        Mat darkEye = new Mat(tmpEye.size(), tmpEye.type(), sc);
        tmpEye.copyTo(darkEye, eyeMask);
        addWeighted(darkEye, 0.5, tmpEye, 0.5, 0, tmpEye);*/
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
        
        warpAffine(facePain, transformedTopFaceImage, transform, face.size());
        
        final Mat grayFace = new Mat();
        cvtColor(face, grayFace, CV_BGR2GRAY);
        putShadows(transformedTopFaceImage, grayFace);
        addWeighted(transformedTopFaceImage, 0.5, face, 0.5, 0, transformedTopFaceImage);
        
        transformedTopFaceImage.copyTo(face, skinMask);
    }
    
    public void reloadImages() {
        try {
            Mat tmp = imread(ResourceManager.getPath(BACK_PAIN_NAME));
            backPain = tmp;
        }
        catch (Exception e) {
            
        }
        try {
            Mat tmp = imread(ResourceManager.getPath(FACE_PAIN_NAME));
            facePain = tmp;
        }
        catch (Exception e) {
            
        }
    }
    
    public void putShadows(Mat image, Mat shadows) {
        threshold(shadows, shadows, 127d, 255d, THRESH_TRUNC);
        final Mat shadowsRGB = new Mat();
        cvtColor(shadows, shadowsRGB, CV_GRAY2BGR);
        addWeighted(image, 1.0, shadowsRGB, 0.75, -127.0, image);
    }
}