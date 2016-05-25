package ru.spbau.mit.reptilian_detector.detector;

import org.bytedeco.javacpp.indexer.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacv.*;
import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_imgproc.*;
import static org.bytedeco.javacpp.opencv_imgcodecs.*;
import static org.bytedeco.javacpp.opencv_objdetect.*;
import static org.bytedeco.javacpp.opencv_highgui.*;
import static org.bytedeco.javacpp.opencv_videoio.*;

@SuppressWarnings({"JavadocType", "PMD"})

final class Main {
    private Main() { }

    public static void main(String[] args) throws Exception {
        final String resultWindowName = "RESULT";
        if (!args[0].equals("%Cam")) {
            System.out.println(args[0]);
            final Mat image = imread(args[0]);
            if (image != null) {
                final FaceDetector detector = new FaceDetector();
                final IFilter filter = new FaceOrientationTestFilter();
                System.out.println("Go detect");
                filter.applyFilter(image, detector.detectFaces(image, true));
                namedWindow(resultWindowName, WINDOW_NORMAL);
                imshow(resultWindowName, image);
                cvWaitKey(0);
                System.out.println("OK");
            }
        } else {
            Mat image;
            final FaceDetector detector = new FaceDetector();
            final IFilter filter = new FaceOrientationTestFilter();
            final VideoCapture camera = new VideoCapture(0);
            camera.open(0);
            while (true) {
                image = new Mat();
                camera.read(image);
                filter.applyFilter(image, detector.detectFaces(image, true));
                imshow(resultWindowName, image);
                cvWaitKey(1);
            }
        }
    }
}
