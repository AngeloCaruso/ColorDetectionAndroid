package com.example.opencv_detection;

import android.graphics.Color;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceView;
import android.view.View;
import android.widget.Button;
import android.widget.Toast;

import com.rtugeek.android.colorseekbar.ColorSeekBar;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCamera2View;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    JavaCameraView javaCameraView;
    Mat cameraFrame, imgGray, imgCanny, colorMat, blurredImg, hsvImg, mask, morphMat;
    ColorSeekBar colorSeekBar;
    Scalar lowerColor; //= new Scalar(90,100,20);
    Scalar uppercolor; //= new Scalar(100,255,255);
    float[] hsv = new float[3];
    BaseLoaderCallback mLoaderCallBack = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch(status){
                case BaseLoaderCallback.SUCCESS:
                    javaCameraView.enableView();
                    break;
                default:
                    super.onManagerConnected(status);
                    break;
            }
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        colorSeekBar = (ColorSeekBar)findViewById(R.id.colorSeekBar);

        colorSeekBar.setOnColorChangeListener(new ColorSeekBar.OnColorChangeListener() {
            @Override
            public void onColorChangeListener(int colorBarPosition, int alphaBarPosition, int color) {
                Log.d("color", String.valueOf(color));
                Color.colorToHSV(color,hsv);
                lowerColor = new Scalar(hsv[0]/2, 40, 40);
                uppercolor = new Scalar((hsv[0]/2) + 10 , 255, 255);
                Log.d("color", String.valueOf(hsv[0]/2) + ", " + String.valueOf(hsv[1]) + ", "+ String.valueOf(hsv[2]));
            }
        });

        if (OpenCVLoader.initDebug()) {
            Toast.makeText(this, "OpenCV loaded", Toast.LENGTH_LONG);
        } else {
            Toast.makeText(this, "Couldn't load OpenCV", Toast.LENGTH_LONG);
        }
        javaCameraView = (JavaCameraView)findViewById(R.id.java_camera_view);
        javaCameraView.setVisibility(SurfaceView.VISIBLE);
        javaCameraView.setCvCameraViewListener(this);
    }

    @Override
    protected void onPause(){
        super.onPause();
        if(javaCameraView!=null){
            javaCameraView.disableView();
        }
    }
    @Override
    protected void onDestroy(){
        super.onDestroy();
        if(javaCameraView!=null){
            javaCameraView.disableView();
        }
    }
    @Override
    protected void onResume(){
        super.onResume();
        if (OpenCVLoader.initDebug()) {
            Toast.makeText(this, "OpenCV loaded", Toast.LENGTH_LONG);
            mLoaderCallBack.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        } else {
            Toast.makeText(this, "Couldn't load OpenCV", Toast.LENGTH_LONG);
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION,this,mLoaderCallBack);
        }
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        cameraFrame = new Mat(height,width, CvType.CV_8UC1);
        imgGray = new Mat(height,width, CvType.CV_8UC1);
        imgCanny = new Mat(height,width, CvType.CV_8UC1);
        colorMat = new Mat(height,width, CvType.CV_8UC4);
        blurredImg = new Mat();
        hsvImg = new Mat();
        mask = new Mat();
        morphMat = new Mat();
    }

    @Override
    public void onCameraViewStopped() {
        cameraFrame.release();
        imgGray.release();
        imgCanny.release();
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        cameraFrame = inputFrame.rgba();
        //apply blur filter
        Imgproc.blur(cameraFrame, blurredImg, new Size(7,7));
        //convert to hsv
        Imgproc.cvtColor(blurredImg, hsvImg, Imgproc.COLOR_RGB2HSV);
        //Define thresholds
        Core.inRange(hsvImg, lowerColor, uppercolor, mask);
        //Morphological filters
        Mat dilateElement = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(24,24));
        Mat erodeElement = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(12,12));

        Imgproc.erode(mask, morphMat, erodeElement);
        Imgproc.erode(mask, morphMat, erodeElement);

        Imgproc.dilate(mask, morphMat, dilateElement);
        Imgproc.dilate(mask, morphMat, dilateElement);

        //Imgproc.cvtColor(cameraFrame, imgGray,Imgproc.COLOR_RGB2GRAY);
        //Imgproc.Canny(morphMat,imgCanny,50,150);
        cameraFrame = this.findColorAndLabel(morphMat,cameraFrame);
        return cameraFrame;
    }

    private Mat findColorAndLabel(Mat mask, Mat frame){
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();

        Imgproc.findContours(mask, contours, hierarchy, Imgproc.RETR_CCOMP, Imgproc.CHAIN_APPROX_SIMPLE);

        if(hierarchy.size().height > 0 && hierarchy.size().width > 0){
            for (int i = 0; i >= 0 ; i = (int) hierarchy.get(0, i)[0]) {
                Imgproc.drawContours(frame,contours,i, new Scalar(250,0,0));
            }
        }
        return frame;
    }

}
