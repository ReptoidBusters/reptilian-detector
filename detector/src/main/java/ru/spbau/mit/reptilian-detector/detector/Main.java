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
import static org.bytedeco.javacpp.opencv_videoio.*;


public class Main {

    public static void main(String[] args) throws Exception {
        if (!args[0].equals("%Cam")) {
            System.out.println(args[0]);
            final Mat image = imread(args[0]);
            if (image != null) {
                final FaceDetector detector = new FaceDetector();
                final RectFilter filter = new RectFilter();
                System.out.println("Go detect");
                final ArrayList<Face> faces = detector.detectFaces(image, true);
                for (Face i : faces) {
                    i.applyFilter(filter);
                }
                namedWindow("Result", WINDOW_NORMAL);
                imshow("Result", image);
                cvWaitKey(0);
                System.out.println("OK");
            }
        } else {
            Mat image;
            final FaceDetector detector = new FaceDetector();
            final RectFilter filter = new RectFilter();
            final VideoCapture camera = new VideoCapture(0);
            camera.open(0);
            while (true) {
                image = new Mat();
                camera.read(image);
                final ArrayList<Face> faces = detector.detectFaces(image, true);
                for (Face i : faces) {
                    i.applyFilter(filter);
                }
                imshow("Result", image);
                cvWaitKey(1);
            }
        }
    }
}
