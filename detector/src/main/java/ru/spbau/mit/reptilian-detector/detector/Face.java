package ru.spbau.mit.reptilian_detector.detector;

import java.lang.Math;

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
    private Mat skinMask;
    private Rect position;
    private Mat perspectivePoints;
    private Mat affinePoints;
    private Mat altAffinePoints;
    
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
    
    public Mat getSkinMask() {
        return skinMask;
    }
    
    public Mat getPerspectivePoints() {
        return perspectivePoints;
    }
    
    public Mat getAffinePoints() {
        return affinePoints;
    }
    
    ////Set methods    
    public void setEyes(RectVector newEyes) {
        eyes = newEyes;
        iniOrientation();
    }
    
    public void setNose(Rect newNose) {
        nose = newNose;
        iniOrientation();
    }
    
    public void setMouth(Rect newMouth) {
        mouth = newMouth;
        iniOrientation();
    }
    
    public void setSkinMask(Mat mask) {
        skinMask = mask;
    }
    
    //Other methods
    
    public void applyFilter(IFilter f) {
        if (image != null && perspectivePoints != null) {
            f.applyFaceFilter(image, skinMask, perspectivePoints, affinePoints, altAffinePoints);
            if (eyes != null) {
                for (int eye = 0; eye < eyes.size(); eye++) {
                    f.applyEyeFilter(image, skinMask, perspectivePoints, affinePoints, altAffinePoints, eyes.get(eye));
                }
            }
            if (mouth != null) {
                f.applyMouthFilter(image, skinMask, perspectivePoints, affinePoints, altAffinePoints, mouth);
            }
            if (nose != null) {
                f.applyNoseFilter(image, skinMask, perspectivePoints, affinePoints, altAffinePoints, nose);
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
        skinMask = null;
        iniOrientation();
    }
    
    void iniOrientation() { //I am not sure that it works correctly
        Segment horiz = null;
        Segment vert = null;
        if (eyes != null && eyes.size() == 2) {
            horiz = new Segment(middleOfRect(eyes.get(0)), middleOfRect(eyes.get(1)));
        }
        if (mouth != null && nose != null) {
            vert = new Segment(middleOfRect(mouth), middleOfRect(nose));
        }
        if (horiz == null) {
            if (vert != null && eyes != null && eyes.size() == 1) {
                Segment tmp = vert.getOrto();
                tmp.moveTo(middleOfRect(eyes.get(0)));
                horiz = new Segment(middleOfRect(eyes.get(0)), tmp.intersect(vert));
                horiz.resize(2d);
                if (horiz.getFirst().x() > horiz.getSecond().x()) {
                    horiz.reverse();
                }
            }
        }
        if (vert == null) {
            Point someFeature = null;
            if (nose != null) {
                someFeature = middleOfRect(nose);
            }
            if (mouth != null) {
                someFeature = middleOfRect(mouth);
            }
            if (horiz != null && someFeature != null) {
                Segment tmp = horiz.getOrto();
                tmp.moveTo(someFeature);
                vert = new Segment(someFeature, tmp.intersect(horiz));
                if (nose != null) {
                    vert.resize(-1d);
                    vert.reverse();
                }
                if (mouth != null) {
                    vert.resize(0.5d);
                }
            }
        }
        perspectivePoints = null;
        affinePoints = null;
        if (vert != null && horiz != null) {
            perspectivePoints = new Mat( 
                    (float)horiz.getFirst().x(), (float)horiz.getFirst().y(),
                    (float)horiz.getSecond().x(), (float)horiz.getSecond().y(),
                    (float)vert.getFirst().x(), (float)vert.getFirst().y(), 
                    (float)vert.getSecond().x(), (float)vert.getSecond().y()
                    ).reshape(2, 4);     
            affinePoints = new Mat( 
                    (float)horiz.getFirst().x(), (float)horiz.getFirst().y(),
                    (float)horiz.getSecond().x(), (float)horiz.getSecond().y(),
                    (float)vert.getSecond().x(), (float)vert.getSecond().y()
                    ).reshape(2, 3);
            altAffinePoints = new Mat( 
                    (float)horiz.getFirst().x(), (float)horiz.getFirst().y(),
                    (float)horiz.getSecond().x(), (float)horiz.getSecond().y(),
                    (float)vert.getFirst().x(), (float)vert.getFirst().y()
                    ).reshape(2, 3);
        }
    }
    
    Point middleOfRect(Rect r) {
        return new Point(r.x() + r.width() / 2, r.y() + r.height() / 2);
    }
}

class Segment {
    private Point first, second;
    
    Segment(Point f, Point s) {
        first = new Point(f);
        second = new Point(s);
    }
    
    void reverse() {
        final Point tmp = first;
        first = second;
        second = tmp;
    }
    
    void moveTo(Point p) {
        final Point f = new Point(p);
        final Point s = new Point(second.x() - first.x() + p.x(),
            second.y() - first.y() + p.y());
        first = f;
        second = s;
    }
    
    void resize(double koef) {
        final Point f = new Point(first);
        final Point s = new Point((int)(first.x() + (second.x() - first.x()) * koef),
                (int)(first.y() + (second.y() - first.y()) * koef));
        first = f;
        second = s;
    }
    
    void setSize(double size) {
        final Point f = new Point(first);
        final Point s = new Point((int)(first.x() + (second.x() - first.x()) * size / length()),
                (int)(first.y() + (second.y() - first.y()) * size / length()));
        first = f;
    }
    
    Segment getOrto() {
        final Point f = new Point(first);
        final Point s = new Point(-(second.y() - first.y()) + first.x(),
                (second.x() - first.x()) + first.y());
        return new Segment(f, s);
    }
    
    Point intersect(Segment s) {
        final int A1 = second.y() - first.y();
        final int B1 = first.x() - second.x();
        final int C1 = first.y() * (second.x() - first.x()) 
                - first.x() * (second.y() - first.y());
        final int A2 = s.second.y() - s.first.y();
        final int B2 = s.first.x() - s.second.x();
        final int C2 = s.first.y() * (s.second.x() - s.first.x()) 
                - s.first.x() * (s.second.y() - s.first.y());
        if (A1 * B2 - A2 * B1 != 0) {
            final int x = (C2 * B1 - C1 * B2) / (A1 * B2 - A2 * B1);
            final int y = (C1 * A2 - C2 * A1) / (A1 * B2 - A2 * B1);
            return new Point(x, y);
        }
        return null;
    }
    
    Point getCenter() {
        return new Point(first.x() + (second.x() - first.x()) / 2,
            first.y() + (second.y() - first.y()) / 2);
    }
    
    Point getFirst() {
        return first;
    }
    
    Point getSecond() {
        return second;
    }
    
    double length() {
        return Math.sqrt((second.x() - first.x()) * (second.x() - first.x()) + (second.y() - first.y()) * (second.y() - first.y()));
    }
}
