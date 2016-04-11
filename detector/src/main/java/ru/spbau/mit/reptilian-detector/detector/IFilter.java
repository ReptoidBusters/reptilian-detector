package ru.spbau.mit.reptilian_detector.detector;

import java.util.ArrayList;

import org.bytedeco.javacv.*;
import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_imgproc.*;
import static org.bytedeco.javacpp.opencv_imgcodecs.*;

public interface IFilter {
    public void applyEyeFilter(Mat face, Rect eye);
    public void applyNoseFilter(Mat face, Rect nose);
    public void applyMouthFilter(Mat face, Rect mouth);
    public void applyFaceFilter(Mat face);
}