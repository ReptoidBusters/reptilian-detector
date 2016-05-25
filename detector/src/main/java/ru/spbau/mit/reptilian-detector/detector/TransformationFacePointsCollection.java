package ru.spbau.mit.reptilian_detector.detector;

import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_imgproc.*;
import static org.bytedeco.javacpp.opencv_imgcodecs.*;

@SuppressWarnings({"JavadocType", "PMD"})

public class TransformationFacePointsCollection {
    private Point eye1;
    private Point eye2;
    private Point nose;
    private Point mouth;
    
    TransformationFacePointsCollection(Point eye1, Point eye2, Point nose, Point mouth) {
        this.eye1 = eye1;
        this.eye2 = eye2;
        this.nose = nose;
        this.mouth = mouth;
    }
    
    Mat getAffinePoints() {
        return new Mat(
                (float) eye1.x(), (float) eye1.y(),
                (float) eye2.x(), (float) eye2.y(),
                (float) nose.x(), (float) nose.y()
                ).reshape(2, 3);
    }
    
    Mat getAltAffinePoints() {
        return new Mat(
                (float) eye1.x(), (float) eye1.y(),
                (float) eye2.x(), (float) eye2.y(),
                (float) mouth.x(), (float) mouth.y()
                ).reshape(2, 3);
    }
    
    Mat getPerspectivePoints() {
        return new Mat(
                (float) eye1.x(), (float) eye1.y(),
                (float) eye2.x(), (float) eye2.y(),
                (float) nose.x(), (float) nose.y(), 
                (float) mouth.x(), (float) mouth.y()
                ).reshape(2, 4);
    }
};
