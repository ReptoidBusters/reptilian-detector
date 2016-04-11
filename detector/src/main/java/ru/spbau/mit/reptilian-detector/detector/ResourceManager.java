package ru.spbau.mit.reptilian_detector.detector;

import java.io.File;
import java.net.URL;

import org.bytedeco.javacpp.indexer.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacv.*;

public class ResourceManager {
    static String getPath(String resourceName) throws Exception {
        int nameStartIndex = resourceName.indexOf("/") + 1;
        String rName = resourceName.substring(nameStartIndex);
        URL url = ResourceManager.class.getClassLoader().getResource(resourceName);
        File file = Loader.extractResource(url, null, rName, rName);
        file.deleteOnExit();
        return file.getAbsolutePath();  
    }
}