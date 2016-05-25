package ru.spbau.mit.reptilian_detector.detector;

import org.bytedeco.javacv.*;
import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_imgproc.*;
import static org.bytedeco.javacpp.opencv_imgcodecs.*;

@SuppressWarnings({"JavadocType", "PMD"})

public class Face {
    private RectVector eyes;
    private Rect nose;
    private Rect mouth;
    //!Not all image, face rect only.
    private Mat image; 
    private Mat grayImage;
    private Mat skinMask;
    private Rect position;
    private TransformationFacePointsCollection transformationPoints;
    
    //img - full image
    //pos - face position on image
    public Face() {
        initFace(null, null, null, null, null, null);
    }
    
    public Face(Mat img, Rect pos) {
        final Mat timg = new Mat(img, pos);
        final Mat gimg = new Mat();
        cvtColor(timg, gimg, CV_BGR2GRAY);
        equalizeHist(gimg, gimg);
        initFace(timg, gimg, pos, null, null, null);
    }
    
    public Face(Mat img, Mat gimg, Rect pos) {
        final Mat timg = new Mat(img, pos);
        final Mat tgimg = new Mat(gimg, pos);
        initFace(timg, tgimg, pos, null, null, null);
    }
    
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
    
    public void setEyes(RectVector newEyes) {
        eyes = newEyes;
        initOrientation();
    }
    
    public void setNose(Rect newNose) {
        nose = newNose;
        initOrientation();
    }
    
    public void setMouth(Rect newMouth) {
        mouth = newMouth;
        initOrientation();
    }
    
    public void setSkinMask(Mat mask) {
        skinMask = mask;
    }
    
    
    public void applyFilter(IFilter f) {
        if (image != null && transformationPoints != null) {
            f.applyFaceFilter(image, skinMask, transformationPoints);
            if (eyes != null) {
                for (int eye = 0; eye < eyes.size(); eye++) {
                    f.applyEyeFilter(image, skinMask, transformationPoints, eyes.get(eye));
                }
            }
            if (mouth != null) {
                f.applyMouthFilter(image, skinMask, transformationPoints, mouth);
            }
            if (nose != null) {
                f.applyNoseFilter(image, skinMask, transformationPoints, nose);
            }
        }
    }
    
    
    private void initFace(Mat im, Mat gim, Rect pos, RectVector eyesIni, Rect noseIni, Rect mouthIni) {
        image = im;
        grayImage = gim;
        nose = noseIni;
        mouth = mouthIni;
        eyes = eyesIni;
        position = pos;
        skinMask = null;
        initOrientation();
    }
    
    private void initOrientation() {
        Segment horiz = null;
        Segment vert = null;
        if (eyes != null && eyes.size() == 2) {
            horiz = new Segment(middleOfRect(eyes.get(0)), middleOfRect(eyes.get(1)));
        }
        if (mouth != null && nose != null) {
            vert = new Segment(middleOfRect(mouth), middleOfRect(nose));
        }
        if (horiz == null) {
            horiz = dedicateHorizontal(vert);
        }
        if (vert == null) {
            vert = dedicateVertical(horiz);
        }
        transformationPoints = null;
        if (vert != null && horiz != null) {
            initTransformMat(vert, horiz);
        }
    }
    
    private Segment dedicateVertical(Segment horiz) {
        Segment vert = null;
        Point someFeature = null;
        if (nose != null) {
            someFeature = middleOfRect(nose);
        }
        if (mouth != null) {
            someFeature = middleOfRect(mouth);
        }
        if (horiz != null && someFeature != null) {
            final Segment tmp = horiz.getOrtho();
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
        return vert;
    }
    
    private Segment dedicateHorizontal(Segment vert) {
        Segment horiz = null;
        if (vert != null && eyes != null && eyes.size() == 1) {
            final Segment tmp = vert.getOrtho();
            tmp.moveTo(middleOfRect(eyes.get(0)));
            horiz = new Segment(middleOfRect(eyes.get(0)), tmp.intersect(vert));
            horiz.resize(2d);
            if (horiz.getFirst().x() > horiz.getSecond().x()) {
                horiz.reverse();
            }
        }
        return horiz;
    }
    
    private void initTransformMat(Segment verts, Segment horiz) {
        transformationPoints = new TransformationFacePointsCollection(
                horiz.getFirst(), horiz.getSecond(), verts.getFirst(), verts.getSecond());
    }
    
    private Point middleOfRect(Rect r) {
        return new Point(r.x() + r.width() / 2, r.y() + r.height() / 2);
    }

    class Segment {
        private Point first;
        private Point second;
    
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
            final Point s = new Point((int) (first.x() + (second.x() - first.x()) * koef),
                    (int) (first.y() + (second.y() - first.y()) * koef));
            first = f;
            second = s;
        }
    
        void setSize(double size) {
            final Point f = new Point(first);
            final Point s = new Point((int) (first.x() + (second.x() - first.x()) * size / length()),
                    (int) (first.y() + (second.y() - first.y()) * size / length()));
            first = f;
            second = s;
        }
    
        Segment getOrtho() {
            final Point f = new Point(first);
            final Point s = new Point(-(second.y() - first.y()) + first.x(),
                    (second.x() - first.x()) + first.y());
            return new Segment(f, s);
        }
    
        Point intersect(Segment s) {
            final int a1 = second.y() - first.y();
            final int b1 = first.x() - second.x();
            final int c1 = first.y() * (second.x() - first.x()) 
                    - first.x() * (second.y() - first.y());
            final int a2 = s.second.y() - s.first.y();
            final int b2 = s.first.x() - s.second.x();
            final int c2 = s.first.y() * (s.second.x() - s.first.x()) 
                    - s.first.x() * (s.second.y() - s.first.y());
            if (a1 * b2 - a2 * b1 != 0) {
                final int x = (c2 * b1 - c1 * b2) / (a1 * b2 - a2 * b1);
                final int y = (c1 * a2 - c2 * a1) / (a1 * b2 - a2 * b1);
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
            return Math.sqrt((second.x() - first.x()) * (second.x() - first.x()) 
                    + (second.y() - first.y()) * (second.y() - first.y()));
        }
    }
}
