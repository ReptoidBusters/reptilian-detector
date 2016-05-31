package ru.spbau.mit.reptilian_detector.detector;

import java.util.ArrayList;
import org.bytedeco.javacv.*;
import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_imgproc.*;
import static org.bytedeco.javacpp.opencv_imgcodecs.*;
import static org.bytedeco.javacpp.opencv_highgui.*;

public class LightReptilianFilter implements IFilter {
    static final String TOP_FACE_IMAGE_NAME 
            = "filters_data/LightReptilianFilter/topImage.png";
    static final String BOTTOM_FACE_IMAGE_NAME 
            = "filters_data/LightReptilianFilter/bottomImage.png";
    static final String TOP_FACE_MASK_NAME 
            = "filters_data/LightReptilianFilter/topMask.png";
            
    private Mat faceTopImage;
    private Mat maskTopImage;
    private Mat faceBottomImage;
    private Mat featuresOnImage;
    private Mat altFeaturesOnImage;
    
    LightReptilianFilter() throws Exception{
        reloadImages();
        featuresOnImage = new Mat( 
                510f, 960f,
                1000f, 960f,
                760f, 1255f).reshape(2, 3); //With nose
        altFeaturesOnImage = new Mat( 
                510f, 960f,
                1000f, 960f,
                760f, 1600f).reshape(2, 3); //With mouth
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
        //reloadImages();
        
        final Mat altTransform = getAffineTransform(altFeaturesOnImage, points.getAltAffinePoints());
        final Mat transform = getAffineTransform(featuresOnImage, points.getAffinePoints());
        final Mat transformedTopFaceMask = new Mat(face.size(), face.type());
        final Mat transformedTopFaceImage = new Mat(face.size(), face.type());
        final Mat transformedBottomFaceImage = new Mat(face.size(), face.type());
        
        warpAffine(maskTopImage, transformedTopFaceMask, transform, face.size());
        warpAffine(faceTopImage, transformedTopFaceImage, transform, face.size());
        warpAffine(faceBottomImage, transformedBottomFaceImage, altTransform, face.size());
        
        final Mat grayFace = new Mat();
        cvtColor(face, grayFace, CV_BGR2GRAY);
        putShadows(transformedTopFaceImage, grayFace);
        putShadows(transformedBottomFaceImage, grayFace);
        double koef = 0.4;
        addWeighted(transformedTopFaceImage, koef, face, 1-koef, 0, transformedTopFaceImage);
        addWeighted(transformedBottomFaceImage, koef, face, 1-koef, 0, transformedBottomFaceImage);
        
        final Mat finalTopFaceMask = new Mat(face.size(), face.type(), Scalar.BLACK);
        transformedTopFaceMask.copyTo(finalTopFaceMask, skinMask);
        
        transformedBottomFaceImage.copyTo(face, skinMask);
        transformedTopFaceImage.copyTo(face, finalTopFaceMask);
    }
    
    public void reloadImages() {
        try {
            Mat tmp = imread(ResourceManager.getPath(TOP_FACE_IMAGE_NAME));
            faceTopImage = tmp;
        }
        catch (Exception e) {    
        }
        try {
            Mat tmp = imread(ResourceManager.getPath(BOTTOM_FACE_IMAGE_NAME));
            faceBottomImage = tmp;
        }
        catch (Exception e) {
        }
        try {
            Mat tmp = imread(ResourceManager.getPath(TOP_FACE_MASK_NAME));
            maskTopImage = tmp;
        }
        catch (Exception e) {
        }
    }
    
    public void putShadows(Mat image, Mat shadows) {
        threshold(shadows, shadows, 200d, 255d, THRESH_TRUNC);
        final Mat shadowsRGB = new Mat();
        cvtColor(shadows, shadowsRGB, CV_GRAY2BGR);
        Mat cpImage = new Mat(image);
        addWeighted(image, 1.0, shadowsRGB, 0.7, -200.0, cpImage);
        cpImage.copyTo(image);
    }
}
