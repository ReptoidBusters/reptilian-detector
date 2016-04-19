package ru.spbau.mit.reptilian_detector.detector;

import org.bytedeco.javacv.*;
import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_imgproc.*;
import static org.bytedeco.javacpp.opencv_imgcodecs.*;

public class Face {
    private RectVector eyes;
    private Rect nose;
    private Rect mouth;
    //!Not all image, face rect only.
    private Mat image; 
    private Mat grayImage;
    private Rect position;
    
    //Constructors
    //im - full image
    //pos - face position on image
    public Face() {
        iniFace(null, null, null, null, null, null);
    }
    
    public Face(Mat im, Rect pos) {
        final Mat tim = new Mat(im, pos);
        final Mat gim = new Mat();
        cvtColor(tim, gim, CV_BGR2GRAY);
        equalizeHist(gim, gim);
        iniFace(tim, gim, pos, null, null, null);
    }
    
    public Face(Mat im, Mat gim, Rect pos) {
        final Mat tim = new Mat(im, pos);
        final Mat tgim = new Mat(gim, pos);
        iniFace(tim, tgim, pos, null, null, null);
    }
    
    public Face(Mat im, Rect pos, RectVector eyesIni, Rect noseIni, Rect mouthIni) {
        final Mat tim = new Mat(im, pos); 
        final Mat gim = new Mat();
        cvtColor(tim, gim, CV_BGR2GRAY);
        equalizeHist(gim, gim);   
        iniFace(tim, gim, pos, eyesIni, noseIni, mouthIni);
    }
    
    public Face(Mat im, Mat gim, Rect pos, RectVector eyesIni, Rect noseIni, Rect mouthIni) {
        final Mat tim = new Mat(im, pos);
        final Mat tgim = new Mat(gim, pos);
        iniFace(tim, tgim, pos, eyesIni, noseIni, mouthIni);
    }
    
    //Field access methods
    ////Get methods
    
    public Rect getPos() {
        return position;
    }
    
    public RectVector getEyes() {
        return eyes;
    }
    
    public Rect getMouth() {
        return mouth;
    }
    
    public Rect getNose() {
        return nose;
    }
    
    public Mat getImage() {
        return image;
    }
    
    public Mat getGrayImage() {
        return grayImage;
    }
    
    ////Set methods    
    public void setEyes(RectVector newEyes) {
        eyes = newEyes;
    }
    
    public void setNose(Rect newNose) {
        nose = newNose;
    }
    
    public void setMouth(Rect newMouth) {
        mouth = newMouth;
    }
    
    //Other methods
    
    public void applyFilter(IFilter f) {
        if (image != null) {
            f.applyFaceFilter(image);
            if (eyes != null) {
                for (int eye = 0; eye < eyes.size(); eye++) {
                    f.applyEyeFilter(image, eyes.get(eye));
                }
            }
            if (mouth != null) {
                f.applyMouthFilter(image, mouth);
            }
            if (nose != null) {
                f.applyNoseFilter(image, nose);
            }
        }
    }
    
    //Inner methods
    
    void iniFace(Mat im, Mat gim, Rect pos, RectVector eyesIni, Rect noseIni, Rect mouthIni) {
        image = im;
        grayImage = gim;
        nose = noseIni;
        mouth = mouthIni;
        eyes = eyesIni;
        position = pos;
    }
}
