package ru.spbau.mit.reptilian_detector.app;

import android.app.Activity;
import android.os.Bundle;
import android.widget.TextView;

public class App extends Activity {
    
    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.app_layout);
    }

    @Override
    public void onStart() {
        super.onStart();
        TextView textView = (TextView) findViewById(R.id.text_view);
        textView.setText("Hello, world !!!");
    }

}

