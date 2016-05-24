package ru.spbau.mit.reptilian_detector.detector;

import java.util.ArrayList;
import org.bytedeco.javacv.*;
import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_imgproc.*;
import static org.bytedeco.javacpp.opencv_imgcodecs.*;
import static org.bytedeco.javacpp.opencv_highgui.*;

public class FaceOrientationTestFilter implements IFilter {
    static final String FACE_IMAGE_NAME 
            = "filters_data/FaceOrientationTest/image.jpg";
    static final String FACE_MASK_NAME 
            = "filters_data/FaceOrientationTest/mask.jpg";
            
    private Mat faceImage;
    private Mat faceMask;
    private Mat featuresOnImage;
    private Mat altFeaturesOnImage;
    
    FaceOrientationTestFilter() throws Exception {
        faceImage = imread(ResourceManager.getPath(FACE_IMAGE_NAME));
        final Mat tmp = imread(ResourceManager.getPath(FACE_MASK_NAME));
        faceMask = new Mat();
        cvtColor(tmp, faceMask, CV_BGR2GRAY);
        featuresOnImage = new Mat(
                820f, 1650f,
                1750f, 1650f,
                1300f, 2190f).reshape(2, 3);
        altFeaturesOnImage = new Mat(
                820f, 1650f,
                1750f, 1650f,
                1300f, 2640f).reshape(2, 3);
    }
    
    public void applyFilter(Mat image, ArrayList<Face> faces) {
        for (Face f : faces) {
            f.applyFilter(this);
        }
    }
    
    @Override
    public void applyEyeFilter(Mat face, Mat skinMask, TransformationFacePointsCollection points, Rect eye) {
        rectangle(face, new Point(eye.x(), eye.y()),
                new Point(eye.x() + eye.width(), eye.y() + eye.height()), Scalar.GREEN,
                2, CV_AA, 0);
    }
    @Override
    public void applyNoseFilter(Mat face, Mat skinMask, TransformationFacePointsCollection points, Rect nose) {
        rectangle(face, new Point(nose.x(), nose.y()),
                new Point(nose.x() + nose.width(), nose.y() + nose.height()), Scalar.BLUE,
                2, CV_AA, 0);
    }
    @Override
    public void applyMouthFilter(Mat face, Mat skinMask, TransformationFacePointsCollection points, Rect mouth) {
        rectangle(face, new Point(mouth.x(), mouth.y()),
                new Point(mouth.x() + mouth.width(), mouth.y() + mouth.height()), Scalar.RED,
                2, CV_AA, 0);        
    }
    @Override
    public void applyFaceFilter(Mat face, Mat skinMask, TransformationFacePointsCollection points) {
        final Mat transform = getAffineTransform(altFeaturesOnImage, points.getAltAffinePoints());
        final Mat transformedMask = new Mat(face.size(), face.type());
        final Mat transformedImage = new Mat(face.size(), face.type());
        warpAffine(faceMask, transformedMask, transform, face.size());
        warpAffine(faceImage, transformedImage, transform, face.size());
        transformedImage.copyTo(face, transformedMask);
        rectangle(face, new Point(2, 2), new Point(face.cols() - 2, face.rows() - 2),
                Scalar.YELLOW, 2, CV_AA, 0);
    }
}
