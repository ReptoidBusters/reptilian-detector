package ru.spbau.mit.reptilian_detector.detector;

import java.util.ArrayList;

import org.bytedeco.javacv.*;
import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_imgproc.*;
import static org.bytedeco.javacpp.opencv_imgcodecs.*;

public class Face {
    public RectVector eyes;
    public Rect nose;
    public Rect mouth;
    public Mat image; //Not all image, face rect only.
    public Mat grayImage;
    public Rect position;
    
        
    //Inner methods
    
    void iniFace (Mat im, Mat gim, Rect pos, RectVector eyesIni, Rect noseIni, Rect mouthIni) {
        image = im;
        grayImage = gim;
        nose = noseIni;
        mouth = mouthIni;
        eyes = eyesIni;
        position = pos;
    }
    
    //Constructors
    //im - full image
    //pos - face position on image
    
    public Face (Mat im, Rect pos) {
        im = new Mat(im, pos);
        Mat gim = new Mat();
        cvtColor(im, gim, CV_BGR2GRAY);
        equalizeHist(gim,gim);
        iniFace(im, gim, pos, null, null, null);
    }
    
    public Face (Mat im, Mat gim, Rect pos) {
        im = new Mat(im, pos);
        gim = new Mat(gim, pos);
        iniFace(im, gim, pos, null, null, null);
    }
    
    public Face (Mat im, Rect pos, RectVector eyesIni, Rect noseIni, Rect mouthIni) {
        im = new Mat(im, pos); 
        Mat gim = new Mat();
        cvtColor(im, gim, CV_BGR2GRAY);
        equalizeHist(gim,gim);   
        iniFace(im, gim , pos, eyesIni, noseIni, mouthIni);
    }
    
    public Face (Mat im, Mat gim, Rect pos, RectVector eyesIni, Rect noseIni, Rect mouthIni) {
        im = new Mat(im, pos);
        gim = new Mat(gim, pos);
        iniFace(im, gim, pos, eyesIni, noseIni, mouthIni);
    }
    
    public Face() {
        iniFace(null, null, null, null, null, null);
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
                f.applyMouthFilter(image,mouth);
            }
            if (nose != null) {
                f.applyNoseFilter(image,nose);
            }
        }
    }
}