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
        if ((args.length == 1) && (!args[0].equals("%Cam"))) {
            System.out.println(args[0]);
            final Mat image = imread(args[0]);
            if (image != null) {
                final FaceDetector detector = new FaceDetector();
                final IFilter filter = new LightReptilianFilter();
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
            final VideoCapture camera = new VideoCapture(0);
            camera.open(0);
            while (true) {
                IFilter filter = new LightReptilianFilter();
                if (args.length == 2) {
                    if (args[1] == "%Dark") {
                        filter = new DarkReptilianFilter();
                    }
                    if (args[1] == "%Pain") {
                        filter = new PainFilter();
                    }
                }
                image = new Mat();
                ///Preventing frames delay
                for (int i = 0; i < 13; i++) {
                    camera.read(image);
                }
                filter.applyFilter(image, detector.detectFaces(image, true));
                imshow(resultWindowName, image);
                cvWaitKey(0);
            }
        }
    }
}
