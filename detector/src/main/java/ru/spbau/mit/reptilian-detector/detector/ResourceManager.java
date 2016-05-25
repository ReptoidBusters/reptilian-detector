package ru.spbau.mit.reptilian_detector.detector;

import java.io.File;
import java.net.URL;

import org.bytedeco.javacpp.indexer.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacv.*;

@SuppressWarnings({"JavadocType", "PMD"})

final class ResourceManager {
    private ResourceManager() { }
    
    static String getPath(String resourceName) throws Exception {
        final int nameStartIndex = resourceName.lastIndexOf("/") + 1;
        final String rName = resourceName.substring(nameStartIndex);
        final URL url = ResourceManager.class.getClassLoader().getResource(resourceName);
        final File file = Loader.extractResource(url, null, rName, rName);
        file.deleteOnExit();
        return file.getAbsolutePath();  
    }
}
