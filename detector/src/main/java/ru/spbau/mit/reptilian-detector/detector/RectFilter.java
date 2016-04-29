package ru.spbau.mit.reptilian_detector.detector;

import org.bytedeco.javacv.*;
import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_imgproc.*;
import static org.bytedeco.javacpp.opencv_imgcodecs.*;

public class RectFilter implements IFilter {
    @Override
    public void applyEyeFilter(Mat face, Mat skinMask, Mat perspectivePoints, Mat affinePoints, Mat altAffinePoints, Rect eye) {
        rectangle(face, new Point(eye.x(), eye.y()),
                new Point(eye.x() + eye.width(), eye.y() + eye.height()), Scalar.GREEN,
                2, CV_AA, 0);
    }
    @Override
    public void applyNoseFilter(Mat face, Mat skinMask, Mat perspectivePoints, Mat affinePoints, Mat altAffinePoints, Rect nose) {
        rectangle(face, new Point(nose.x(), nose.y()),
                new Point(nose.x() + nose.width(), nose.y() + nose.height()), Scalar.BLUE,
                2, CV_AA, 0);
    }
    @Override
    public void applyMouthFilter(Mat face, Mat skinMask, Mat perspectivePoints, Mat affinePoints, Mat altAffinePoints, Rect mouth) {
        rectangle(face, new Point(mouth.x(), mouth.y()),
                new Point(mouth.x() + mouth.width(), mouth.y() + mouth.height()), Scalar.RED,
                2, CV_AA, 0);        
    }
    @Override
    public void applyFaceFilter(Mat face, Mat skinMask, Mat perspectivePoints, Mat affinePoints, Mat altAffinePoints) {
        rectangle(face, new Point(2, 2), new Point(face.cols() - 2, face.rows() - 2),
                Scalar.YELLOW, 2, CV_AA, 0);
    }
}
