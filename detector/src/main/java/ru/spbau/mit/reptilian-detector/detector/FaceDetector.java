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


public class FaceDetector {
    static final double faceScaleConst = 0;
    
    static final String faceHaarCascadeName  
            = "haarcascades/haarcascade_frontalface.xml";
    static final String rightEyeHaarCascadeName  
            = "haarcascades/haarcascade_rightEye.xml";
    static final String leftEyeHaarCascadeName  
            = "haarcascades/haarcascade_leftEye.xml";
    static final String mouthHaarCascadeName  
            = "haarcascades/haarcascade_mouth.xml";
    static final String noseHaarCascadeName  
            = "haarcascades/haarcascade_nose.xml";

    CascadeClassifier faceHaarCascade;
    CascadeClassifier rightEyeHaarCascade;
    CascadeClassifier leftEyeHaarCascade;
    CascadeClassifier mouthHaarCascade;
    CascadeClassifier noseHaarCascade;
    
    //Constructors
    
    public FaceDetector () throws Exception {
        faceHaarCascade = new CascadeClassifier(
                ResourceManager.getPath(faceHaarCascadeName)
        );
        leftEyeHaarCascade = new CascadeClassifier(
                ResourceManager.getPath(leftEyeHaarCascadeName)
        );
        rightEyeHaarCascade = new CascadeClassifier(
                ResourceManager.getPath(rightEyeHaarCascadeName)
        );
        mouthHaarCascade = new CascadeClassifier(
                ResourceManager.getPath(mouthHaarCascadeName)
        );
        noseHaarCascade = new CascadeClassifier(
                ResourceManager.getPath(noseHaarCascadeName)
        );
    }
    
    //Other methods
    
    public ArrayList<Face> detectFaces(Mat image, boolean withFeatures) {
        Mat imageGray = new Mat();
        cvtColor(image, imageGray, CV_BGR2GRAY);
        equalizeHist(imageGray,imageGray);
        RectVector faceRects = new RectVector();
        faceHaarCascade.detectMultiScale(imageGray, faceRects,
                1.05, 8, CV_HAAR_DO_CANNY_PRUNING,
                new Size(150, 150), new Size(imageGray.cols(),imageGray.rows()));
        faceRects = deleteInnerRects(faceRects);
        ArrayList<Face> result = new ArrayList<Face>();
        for (int i = 0; i < faceRects.size(); i++) {
            Rect faceRect = new Rect(new Point(faceRects.get(i).x(), 
                    faceRects.get(i).y() - faceRects.get(i).height() / 5),
                    new Point(faceRects.get(i).x() + faceRects.get(i).width(),
                    faceRects.get(i).y() + (faceRects.get(i).height() * 6) / 5));
            faceRect = cutRect(faceRect, image);
            Face newFace = new Face(image,imageGray,faceRect);
            if (withFeatures) {
                detectEyesOnFace(newFace);
                detectMouthOnFace(newFace);
                detectNoseOnFace(newFace);
            }
            result.add(newFace);
        }
        return result;
    } 
    
    public void detectEyesOnFace(Face face) {
        Rect leftEyesPositionRect = new Rect( new Point(0,face.position.height() / 7),
                new Point(face.position.width() / 2, face.position.height() * 4 / 7));
        Rect rightEyesPositionRect = new Rect( new Point(face.position.width() / 2, 
                face.position.height() / 7), new Point(face.position.width(),
                face.position.height() * 4 / 7));
        Mat leftGrayFace = new Mat(face.grayImage, leftEyesPositionRect);
        Mat rightGrayFace = new Mat(face.grayImage, rightEyesPositionRect);
        
        Size minFeatureSize = new Size(face.position.width()/5, face.position.height()/7 /2);
        Size maxFeatureSize = new Size(face.position.width()/2, face.position.height()/2);
        
        RectVector eyes = new RectVector();
        RectVector leftEyeRects = new RectVector();
        leftEyeHaarCascade.detectMultiScale(leftGrayFace, leftEyeRects,
                1.05, 2, CV_HAAR_DO_CANNY_PRUNING,
                minFeatureSize, maxFeatureSize);
        leftEyeRects = deleteInnerRects(leftEyeRects);
        Rect leftEye = null;
        if (leftEyeRects.size() > 0) {
            leftEye = leftEyeRects.get(0);
            for (int i = 1; i < leftEyeRects.size(); i++ ) {
                if (leftEyeRects.get(i).y() > leftEye.y()) {
                    leftEye = leftEyeRects.get(i);
                }
            }
            eyes.resize(eyes.size() + 1);
            eyes.put(eyes.size() - 1, new Rect(new Point(leftEye.x(),
                leftEye.y() + face.position.height() / 7), new Point(
                leftEye.x() + leftEye.width(), leftEye.y()+ leftEye.height() +
                face.position.height() / 7)));
        }
        RectVector rightEyeRects = new RectVector();
        rightEyeHaarCascade.detectMultiScale(rightGrayFace, rightEyeRects,
                1.05, 2, CV_HAAR_DO_CANNY_PRUNING,
                minFeatureSize, maxFeatureSize);
        rightEyeRects = deleteInnerRects(rightEyeRects);
        Rect rightEye = null;
        if (rightEyeRects.size() > 0) {
            rightEye = rightEyeRects.get(0);
            for (int i = 1; i < rightEyeRects.size(); i++ ) {
                if (rightEyeRects.get(i).y() > rightEye.y()) {
                    rightEye = rightEyeRects.get(i);
                }
            } 
            eyes.resize(eyes.size() + 1);
            eyes.put(eyes.size() - 1, new Rect(new Point(face.position.width() / 2 + rightEye.x(),
                rightEye.y() + face.position.height() / 7), new Point(
                face.position.width() / 2 + rightEye.x() + rightEye.width(),
                rightEye.y()+ rightEye.height() + face.position.height() / 7)));
        }
        face.eyes = eyes; 
    }
    
    public void detectNoseOnFace(Face face) {
        Rect nosePositionRect = new Rect( new Point(0,face.position.height() * 3 / 7), 
                new Point(face.position.width(), face.position.height() * 5 / 7));
        Mat grayFace = new Mat(face.grayImage, nosePositionRect);
        Size minFeatureSize = new Size(face.position.width()/5, face.position.height()/7);
        Size maxFeatureSize = new Size(face.position.width()/2, face.position.height()/2);
        RectVector noseRects = new RectVector();
        noseHaarCascade.detectMultiScale(grayFace, noseRects,
                1.05, 3, CV_HAAR_DO_CANNY_PRUNING,
                minFeatureSize, maxFeatureSize);
        noseRects = deleteInnerRects(noseRects);
        if (noseRects.size() > 0) {
            face.nose = noseRects.get(0);
            for (int i = 1; i < noseRects.size(); i++ ) {
                if (noseRects.get(i).y() < face.nose.y()) {
                    face.nose = noseRects.get(i);
                }
            }
            face.nose = new Rect( new Point(face.nose.x(), 
                    face.nose.y() + face.position.height() * 3 / 7), new Point(
                    face.nose.x() + face.nose.width(),
                    face.nose.y() + face.nose.height() + face.position.height() * 3 / 7));
            face.nose = cutRect(face.nose, face.image);
        }
    }
    
    public void detectMouthOnFace(Face face) {
        Rect mouthPositionRect = new Rect( new Point(0,face.position.height() * 9 / 14),
                new Point(face.position.width(), face.position.height()));
        Mat grayFace = new Mat (face.grayImage, mouthPositionRect);
        Size minFeatureSize = new Size(face.position.width()/5, face.position.height()/7);
        Size maxFeatureSize = new Size(face.position.width()/2, face.position.height());
        RectVector mouthRects = new RectVector();
        mouthHaarCascade.detectMultiScale(grayFace, mouthRects,
                1.05, 2, CV_HAAR_DO_CANNY_PRUNING,
                minFeatureSize, maxFeatureSize);
        mouthRects = deleteInnerRects(mouthRects);
        if (mouthRects.size() > 0) {
            face.mouth = mouthRects.get(0);
            for (int i = 1; i < mouthRects.size(); i++ ) {
                if (mouthRects.get(i).y() < face.mouth.y()) {
                    face.mouth = mouthRects.get(i);
                }
            }
            face.mouth = new Rect( new Point(face.mouth.x(), 
                    face.mouth.y() + face.position.height() * 9 / 14), new Point(
                    face.mouth.x() + face.mouth.width(),
                    face.mouth.y() + face.mouth.height() + face.position.height() * 9 / 14));
            face.mouth = cutRect(face.mouth, face.image);
        }
    }
    //Inner Methods
    
    //WHY I NEED TO DO THIS!? STUPID JAVA XC
    int min(int a, int b) {
        if (a < b) return a;
        return b;
    }
    
    Rect cutRect(Rect r, Mat m) {
        Point p1 = new Point(- min(-r.x(),0), - min(-r.y(),0));
        Point p2 = new Point(min(r.x() + r.width(), m.cols() - 1),
                min(r.y() + r.height(), m.rows() - 1));
        return new Rect(p1, p2);
    }

    
    //remove unnecessery rects
    RectVector deleteInnerRects(RectVector vect) {
        RectVector result = new RectVector();
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
    
    //Check is rectangle a situated in rectangle b?
    boolean innerRect(Rect a, Rect b) {
        if (a.x() > b.x() && a.y() > b.y() && (a.x() + a.width()) < (b.x() + b.width()) &&
                (a.y() + a.height()) < (b.y() + b.height())) {
            
            return true;
        }
        return false;
    }
}