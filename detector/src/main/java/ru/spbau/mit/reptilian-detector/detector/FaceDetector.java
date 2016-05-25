package ru.spbau.mit.reptilian_detector.detector;

import java.util.ArrayList;

import org.bytedeco.javacpp.indexer.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacv.*;
import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_imgproc.*;
import static org.bytedeco.javacpp.opencv_imgcodecs.*;
import static org.bytedeco.javacpp.opencv_objdetect.*;
import static org.bytedeco.javacpp.opencv_highgui.*;

@SuppressWarnings({"JavadocType", "PMD"})

public class FaceDetector {   
    static final float FACE_DETECT_ACCURACY = 1.05f;
    static final float FEATURE_DETECT_ACCURACY = 1.03f;
 
    static final String FACE_HAAR_CASCADE_NAME  
            = "haarcascades/haarcascade_frontalface.xml";
    static final String RIGHT_EYE_HAAR_CASCADE_NAME  
            = "haarcascades/haarcascade_rightEye.xml";
    static final String LEFT_EYE_HAAR_CASCADE_NAME  
            = "haarcascades/haarcascade_leftEye.xml";
    static final String MOUTH_HAAR_CASCADE_NAME 
            = "haarcascades/haarcascade_mouth.xml";
    static final String NOSE_HAAR_CASCADE_NAME  
            = "haarcascades/haarcascade_nose.xml";

    private CascadeClassifier faceHaarCascade;
    private CascadeClassifier rightEyeHaarCascade;
    private CascadeClassifier leftEyeHaarCascade;
    private CascadeClassifier mouthHaarCascade;
    private CascadeClassifier noseHaarCascade;
    
    public FaceDetector() throws Exception {
        faceHaarCascade = new CascadeClassifier(
                ResourceManager.getPath(FACE_HAAR_CASCADE_NAME)
        );
        leftEyeHaarCascade = new CascadeClassifier(
                ResourceManager.getPath(LEFT_EYE_HAAR_CASCADE_NAME)
        );
        rightEyeHaarCascade = new CascadeClassifier(
                ResourceManager.getPath(RIGHT_EYE_HAAR_CASCADE_NAME)
        );
        mouthHaarCascade = new CascadeClassifier(
                ResourceManager.getPath(MOUTH_HAAR_CASCADE_NAME)
        );
        noseHaarCascade = new CascadeClassifier(
                ResourceManager.getPath(NOSE_HAAR_CASCADE_NAME)
        );
    }
    
    public ArrayList<Face> detectFaces(Mat image, boolean withFeatures) {
        final Mat imageGray = new Mat();
        cvtColor(image, imageGray, CV_BGR2GRAY);
        equalizeHist(imageGray, imageGray);
        RectVector faceRects = new RectVector();
        faceHaarCascade.detectMultiScale(imageGray, faceRects,
                FACE_DETECT_ACCURACY, 8, CV_HAAR_DO_CANNY_PRUNING,
                new Size(150, 150), new Size(imageGray.cols(), imageGray.rows()));
        faceRects = deleteInnerRects(faceRects);
        final ArrayList<Face> result = new ArrayList<Face>();
        for (int i = 0; i < faceRects.size(); i++) {
            Rect faceRect = new Rect(new Point(faceRects.get(i).x(), 
                    faceRects.get(i).y() - faceRects.get(i).height() / 5),
                    new Point(faceRects.get(i).x() + faceRects.get(i).width(),
                    faceRects.get(i).y() + (faceRects.get(i).height() * 6) / 5));
            faceRect = cutRect(faceRect, image);
            final Face newFace = new Face(image, imageGray, faceRect);
            if (withFeatures) {
                detectEyesOnFace(newFace);
                detectMouthOnFace(newFace);
                detectNoseOnFace(newFace);
                detectSkin(newFace);
            }
            result.add(newFace);
        }
        return result;
    } 
    
    public void detectEyesOnFace(Face face) {
        final Rect leftEyesPositionRect = new Rect(new Point(0, face.getPos().height() / 7),
                new Point(face.getPos().width() * 3 / 5, face.getPos().height() * 9 / 14));
        final Rect rightEyesPositionRect = new Rect(new Point(face.getPos().width() * 1 / 5, 
                face.getPos().height() / 7), new Point(face.getPos().width(),
                face.getPos().height() * 9 / 14));
        final Mat leftGrayFace = new Mat(face.getGrayImage(), leftEyesPositionRect);
        final Mat rightGrayFace = new Mat(face.getGrayImage(), rightEyesPositionRect);
        
        final Size minFeatureSize = new Size(face.getPos().width() / 5, face.getPos().height() / 7 / 2);
        final Size maxFeatureSize = new Size(face.getPos().width() / 2, face.getPos().height() / 2);
        
        final RectVector eyes = new RectVector();
        RectVector leftEyeRects = new RectVector();
        leftEyeHaarCascade.detectMultiScale(leftGrayFace, leftEyeRects,
                FEATURE_DETECT_ACCURACY, 2, CV_HAAR_DO_CANNY_PRUNING,
                minFeatureSize, maxFeatureSize);
        leftEyeRects = deleteInnerRects(leftEyeRects);
        Rect leftEye = null;
        if (leftEyeRects.size() > 0) {
            leftEye = leftEyeRects.get(0);
            for (int i = 1; i < leftEyeRects.size(); i++) {
                if (leftEyeRects.get(i).y() > leftEye.y()) {
                    leftEye = leftEyeRects.get(i);
                }
            }
            eyes.resize(eyes.size() + 1);
            eyes.put(eyes.size() - 1, new Rect(new Point(leftEye.x(),
                leftEye.y() + face.getPos().height() / 7), new Point(
                leftEye.x() + leftEye.width(), leftEye.y() + leftEye.height()
                + face.getPos().height() / 7)));
        }
        RectVector rightEyeRects = new RectVector();
        rightEyeHaarCascade.detectMultiScale(rightGrayFace, rightEyeRects,
                FEATURE_DETECT_ACCURACY, 2, CV_HAAR_DO_CANNY_PRUNING,
                minFeatureSize, maxFeatureSize);
        rightEyeRects = deleteInnerRects(rightEyeRects);
        Rect rightEye = null;
        if (rightEyeRects.size() > 0) {
            rightEye = rightEyeRects.get(0);
            for (int i = 1; i < rightEyeRects.size(); i++) {
                if (rightEyeRects.get(i).y() > rightEye.y()) {
                    rightEye = rightEyeRects.get(i);
                }
            } 
            eyes.resize(eyes.size() + 1);
            eyes.put(eyes.size() - 1, new Rect(new Point(face.getPos().width() * 1 / 3 + rightEye.x(),
                rightEye.y() + face.getPos().height() / 7), new Point(
                face.getPos().width() * 1 / 3 + rightEye.x() + rightEye.width(),
                rightEye.y() + rightEye.height() + face.getPos().height() / 7)));
        }
        face.setEyes(eyes); 
    }
    
    public void detectNoseOnFace(Face face) {
        final Rect nosePositionRect = new Rect(new Point(0, face.getPos().height() * 3 / 7), 
                new Point(face.getPos().width(), face.getPos().height() * 5 / 7));
        final Mat grayFace = new Mat(face.getGrayImage(), nosePositionRect);
        final Size minFeatureSize = new Size(face.getPos().width() / 5, face.getPos().height() / 7);
        final Size maxFeatureSize = new Size(face.getPos().width() / 2, face.getPos().height() / 2);
        RectVector noseRects = new RectVector();
        noseHaarCascade.detectMultiScale(grayFace, noseRects,
                FEATURE_DETECT_ACCURACY, 3, CV_HAAR_DO_CANNY_PRUNING,
                minFeatureSize, maxFeatureSize);
        noseRects = deleteInnerRects(noseRects);
        if (noseRects.size() > 0) {
            Rect result = noseRects.get(0);
            for (int i = 1; i < noseRects.size(); i++) {
                if (noseRects.get(i).y() < result.y()) {
                    result = noseRects.get(i);
                }
            }
            result = new Rect(new Point(result.x(), 
                    result.y() + face.getPos().height() * 3 / 7), new Point(
                    result.x() + result.width(),
                    result.y() + result.height() + face.getPos().height() * 3 / 7));
            face.setNose(cutRect(result, face.getImage()));
        }
    }
    
    public void detectMouthOnFace(Face face) {
        final Rect mouthPositionRect = new Rect(new Point(0, face.getPos().height() * 9 / 14),
                new Point(face.getPos().width(), face.getPos().height()));
        final Mat grayFace = new Mat(face.getGrayImage(), mouthPositionRect);
        final Size minFeatureSize = new Size(face.getPos().width() / 5, face.getPos().height() / 7);
        final Size maxFeatureSize = new Size(face.getPos().width() / 2, face.getPos().height());
        RectVector mouthRects = new RectVector();
        mouthHaarCascade.detectMultiScale(grayFace, mouthRects,
                FEATURE_DETECT_ACCURACY, 2, CV_HAAR_DO_CANNY_PRUNING,
                minFeatureSize, maxFeatureSize);
        mouthRects = deleteInnerRects(mouthRects);
        if (mouthRects.size() > 0) {
            Rect result = mouthRects.get(0);
            for (int i = 1; i < mouthRects.size(); i++) {
                if (mouthRects.get(i).y() < result.y()) {
                    result = mouthRects.get(i);
                }
            }
            result = new Rect(new Point(result.x(), 
                    result.y() + face.getPos().height() * 9 / 14), new Point(
                    result.x() + result.width(),
                    result.y() + result.height() + face.getPos().height() * 9 / 14));
            face.setMouth(cutRect(result, face.getImage()));
        }
    }
    
    public void detectSkin(Face face) {
        //Not implemented
    }
    
    private Rect cutRect(Rect r, Mat m) {
        final Point p1 = new Point(-Math.min(-r.x(), 0), -Math.min(-r.y(), 0));
        final Point p2 = new Point(Math.min(r.x() + r.width(), m.cols() - 1),
                Math.min(r.y() + r.height(), m.rows() - 1));
        return new Rect(p1, p2);
    }

    private RectVector deleteInnerRects(RectVector vect) {
        final RectVector result = new RectVector();
        for (int i = 0; i < vect.size(); i++) {
            boolean putIt = true;
            for (int j = 0; j < vect.size(); j++) {
                if (j != i && innerRect(vect.get(j), vect.get(i))) {
                    putIt = false;
                    break;
                }
            }
            if (putIt) {
                result.resize(result.size() + 1);
                result.put(result.size() - 1, vect.get(i));
            }
        }
        return result;
    }
    
    private boolean innerRect(Rect a, Rect b) {
        if (a.x() > b.x() && a.y() > b.y() && (a.x() + a.width()) < (b.x() + b.width())
                && (a.y() + a.height()) < (b.y() + b.height())) {
            return true;
        }
        return false;
    }
}
