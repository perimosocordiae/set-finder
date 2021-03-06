package com.perimosocordiae.setfinder;

import java.util.*;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.*;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.imgproc.Imgproc;

import android.app.Activity;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.WindowManager;

public class SetFinderActivity extends Activity implements CvCameraViewListener2 {
    private static final String  TAG = "SetFinder";

    private MenuItem             mShowDebug;
    private CameraBridgeViewBase mOpenCvCameraView;
    // Generic temporary matrices.
    private Mat fr_tmp1, fr_tmp2;
    // Storage for card outlines.
    private List<MatOfPoint> rects;
    // Storage for card information.
    private List<SetCard> cards;
    // Cropping target, used for generative perspective transforms.
    // Must be initialized after OpenCV loads.
    private Mat cropTarget;

    // Color constants.
    private static final Scalar cardOutlineColor = new Scalar(0, 255, 0);
    private static final Scalar noSetsColor = new Scalar(255, 0, 0);
    private static final Scalar blackColor = Scalar.all(0);
    private static final Scalar whiteColor = Scalar.all(255);
    // Upper and lower bounds for card/shape thresholding.
    private static final Scalar cardHSVlb = new Scalar(0,0,220);
    private static final Scalar cardHSVub = new Scalar(255,70,255);
    // Size of cropped cards.
    private static final Size cardSize = new Size(450, 450);

    public static boolean debugMode = false;
    // findCards params
    public double sideErrorScale = 0.02;
    public double maxCornerAngleCos = 0.3;
    public int minRectArea = 1000;
    public int maxRectArea = 100000;
    // fillAttributes params
    public int minShapeArea = 100;

    private BaseLoaderCallback  mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");
                    mOpenCvCameraView.enableView();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    public SetFinderActivity() {
        Log.i(TAG, "Instantiated new SetFinderActivity");
    }

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.set_finder_surface_view);

        mOpenCvCameraView = (CameraBridgeViewBase)
            findViewById(R.id.set_finder_activity_surface_view);
        mOpenCvCameraView.setCvCameraViewListener(this);
    }

    @Override
    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume()
    {
        super.onResume();
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_3, this, mLoaderCallback);
    }

    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        mShowDebug = menu.add("Show Debug Info");
        menu.add("Find Sets");
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        debugMode = (item == mShowDebug);
        return true;
    }

    public void onCameraViewStarted(int width, int height) {
        fr_tmp1 = new Mat();
        fr_tmp2 = new Mat();
        rects = new ArrayList<MatOfPoint>();
        cards = new ArrayList<SetCard>();
        cropTarget = new Mat(4, 2, CvType.CV_32F);
        cropTarget.put(0, 0,
           0.0, 0.0,
           cardSize.width-1, 0.0,
           cardSize.width-1, cardSize.height-1,
           0.0, cardSize.height-1);
    }

    public void onCameraViewStopped() {
        // Explicitly deallocate Mats
        if (fr_tmp1 != null) { fr_tmp1.release(); fr_tmp1 = null; }
        if (fr_tmp2 != null) { fr_tmp2.release(); fr_tmp2 = null; }
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        Mat rgba = inputFrame.rgba();
        // scale down a bit
        //TODO: choose an absolute size and scale to that.
        Size origSize = rgba.size();
        Imgproc.resize(rgba, rgba, new Size(), 0.5, 0.5, Imgproc.INTER_AREA);

        // Find any cards in the image, and store them in this.rects
        findCards(rgba);
        Log.i(TAG, "found " + rects.size() + " cards.");

        // Fill in the four attributes for each card, and stores them in this.cards
        fillAttributes(rgba);

        if (debugMode) {
            showDebugInfo(rgba);
        } else {
            showSet(rgba);
        }
        // scale it back up
        Imgproc.resize(rgba, rgba, origSize, 0, 0, Imgproc.INTER_LINEAR);
        return rgba;
    }

    private void showSet(Mat rgba) {
        int n = cards.size();
        for (int i = 0; i < n-2; i++) {
            for (int j = i+1; j < n-1; j++) {
                for (int k = j+1; k < n; k++) {
                    if (cards.get(i).setWith(cards.get(j), cards.get(k))) {
                        List<MatOfPoint> setOutlines = Arrays.asList(
                            rects.get(i), rects.get(j), rects.get(k));
                        Imgproc.drawContours(rgba, setOutlines, -1, cardOutlineColor, 3);
                        return;
                    }
                }
            }
        }
        // no sets found
        Size sizeRgba = rgba.size();
        int rows = (int) sizeRgba.height;
        int cols = (int) sizeRgba.width;
        Point pos = new Point(rows/2 - 2, cols/2 - 98);
        String text = getString(R.string.no_sets);
        Core.putText(rgba, text, pos, Core.FONT_HERSHEY_SIMPLEX, 1, blackColor, 2);
        pos.x += 2;
        pos.y += 2;
        Core.putText(rgba, text, pos, Core.FONT_HERSHEY_SIMPLEX, 1, noSetsColor, 2);
    }

    private void showDebugInfo(Mat rgba) {
        Imgproc.drawContours(rgba, rects, -1, cardOutlineColor, 3);
        for (int i = 0; i < cards.size(); i++) {
            String label = cards.get(i).debugString();
            MatOfPoint rect = rects.get(i);
            Point pos = rect.toArray()[0];
            Core.putText(rgba, label, pos, Core.FONT_HERSHEY_SIMPLEX, 0.5, blackColor);
            pos.x--;
            pos.y--;
            Core.putText(rgba, label, pos, Core.FONT_HERSHEY_SIMPLEX, 0.5, whiteColor);
        }
    }

    private int[] findSet() {
        int n = cards.size();
        for (int i = 0; i < n-2; i++) {
            for (int j = i+1; j < n-1; j++) {
                for (int k = j+1; k < n; k++) {
                    if (cards.get(i).setWith(cards.get(j), cards.get(k))) {
                        return new int[] {i, j, k};
                    }
                }
            }
        }
        return null;
    }

    static class SetCard {
        int[] attributes = new int[4];
        private static final int NUMBER = 0, FILLING = 1, COLOR = 2, SHAPE = 3;
        private static final int SOLID = 0, OPEN = 1, STRIPED = 2;
        private static final int RED = 0, PURPLE = 1, GREEN = 2;
        private static final int OVAL = 0, DIAMOND = 1, ESS = 2;
        private static final String[][] attrNames = {
            {"1","2","3"},
            {"Solid","Open","Striped"},
            {"Red","Purple","Green"},
            {"Oval","Diamond","Ess"}
        };
        // Depends on the color we choose.
        Scalar shapeHSVlb, shapeHSVub;

        public void setNumber(List<MatOfPoint> contours) {
            int number = contours.size();
            if (number > 3) {
                Log.w(TAG, "Got too many contours, setting number to 3.");
                number = 3;
            } else if (number == 0) {
                Log.w(TAG, "Got no contours, setting number to 1.");
                number = 1;
            }
            attributes[NUMBER] = number - 1;
        }

        public void setColor(Mat hsv) {
            Mat hueThresh = new Mat();
            // green
            Scalar lb = new Scalar(36, 0, 0);
            Scalar ub = new Scalar(110, 255, 255);
            Core.inRange(hsv, lb, ub, hueThresh);
            double green = Core.sumElems(hueThresh).val[0];
            // purple
            lb.val[0] = 120;  // min_hue
            ub.val[0] = 255;  // max_hue
            Core.inRange(hsv, lb, ub, hueThresh);
            double purple = Core.sumElems(hueThresh).val[0];
            // red
            lb.val[0] = 0;  // min_hue
            ub.val[0] = 10;  // max_hue
            lb.val[1] = 60;  // min_sat
            Core.inRange(hsv, lb, ub, hueThresh);
            double red = Core.sumElems(hueThresh).val[0];

            if (green > purple && green > red) {
                attributes[COLOR] = GREEN;
                shapeHSVlb = new Scalar(36, 0, 0);
                shapeHSVub = new Scalar(110, 255, 255);
            } else if (purple > green && purple > red) {
                attributes[COLOR] = PURPLE;
                shapeHSVlb = new Scalar(120, 0, 0);
                shapeHSVub = new Scalar(255, 255, 255);
            } else {
                attributes[COLOR] = RED;
                shapeHSVlb = new Scalar(0, 60, 0);
                shapeHSVub = new Scalar(10, 255, 255);
            }
        }

        public void setFilling(Mat hsv, Mat thresh) {
            Scalar meanHSV = Core.mean(hsv, thresh);
            double meanSat = meanHSV.val[1];
            double meanVal = meanHSV.val[2];
            if (meanSat > meanVal) {
                attributes[FILLING] = SOLID;
                return;
            }
            if (meanVal - meanSat < 60) {
                attributes[FILLING] = OPEN;
            } else {
                attributes[FILLING] = STRIPED;
            }
        }

        public void setShape(List<MatOfPoint> contours, double sideErrorScale) {
            MatOfPoint2f approx = new MatOfPoint2f();
            for (MatOfPoint contour : contours) {
                contour.convertTo(approx, CvType.CV_32FC2);
                double arcLength = Imgproc.arcLength(approx, true);
                Rect br = Imgproc.boundingRect(contour);
                double arcLengthRect = (br.width + br.height) * 2;
                if (arcLength > arcLengthRect) {
                    continue;  // very concave!
                }
                Imgproc.approxPolyDP(approx, approx, sideErrorScale * arcLength, true);
                long numEdges = approx.total();
                if (4 <= numEdges && numEdges <= 8) {
                    attributes[SHAPE] = DIAMOND;  // a little wiggle room
                    return;
                }
                approx.convertTo(contour, CvType.CV_32S);
                if (Imgproc.isContourConvex(contour)) {
                    attributes[SHAPE] = OVAL;
                } else {
                    attributes[SHAPE] = ESS;
                }
                return;
            }
            // all contours are very convex: try something else!
            MatOfInt hullIdx = new MatOfInt();
            int[] point = new int[2];
            for (MatOfPoint contour : contours) {
                Imgproc.convexHull(contour, hullIdx);
                approx.create(hullIdx.rows(), 1, CvType.CV_32FC2);
                int hi = 0;
                for (int idx : hullIdx.toArray()) {
                    contour.get(idx, 0, point);
                    approx.put(hi++, 0, (float)point[0], (float)point[1]);
                }
                double arcLength = Imgproc.arcLength(approx, true);
                Imgproc.approxPolyDP(approx, approx, sideErrorScale * arcLength, true);
                long numEdges = approx.total();
                if (4 <= numEdges && numEdges <= 6) {
                    attributes[SHAPE] = DIAMOND;
                    return;
                }
            }
            // no clue, just guess oval
            attributes[SHAPE] = OVAL;
        }

        public String debugString() {
            StringBuilder sb = new StringBuilder();
            for (int i=0; i<4; i++) {
                sb.append(attrNames[i][attributes[i]]).append(" ");
            }
            return sb.toString();
        }

        public boolean setWith(SetCard bCard, SetCard cCard) {
            for (int i=0; i<4; i++) {
                int a = attributes[i];
                int b = bCard.attributes[i];
                int c = cCard.attributes[i];
                if (a==b) {
                    if (b!=c) return false;
                } else if (a==c || b==c) {
                    return false;
                }
            }
            return true;
        }
    }

    private void fillAttributes(Mat rgba) {
        cards.clear();  // TODO: use prior information about these cards?
        Mat bbox = new Mat(4, 2, CvType.CV_32F);
        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        Mat hierarchy = new Mat();
        int[] hierData = new int[4];
        for (MatOfPoint rect : rects) {
            // TODO: avoid allocating a new SetCard every time.
            SetCard sc = new SetCard();
            // Crop out the card from the full image.
            rect.convertTo(bbox, CvType.CV_32FC2);
            Mat transform = Imgproc.getPerspectiveTransform(bbox, cropTarget);
            Imgproc.warpPerspective(rgba, fr_tmp1, transform, cardSize);
            // Convert the cropped card to HSV.
            Imgproc.cvtColor(fr_tmp1, fr_tmp1, Imgproc.COLOR_RGB2HSV);
            // Set the color attribute.
            sc.setColor(fr_tmp1);
            // Threshold out the card shapes.
            Core.inRange(fr_tmp1, sc.shapeHSVlb, sc.shapeHSVub, fr_tmp2);
            // Set the filling attribute
            sc.setFilling(fr_tmp1, fr_tmp2);
            // Find contours in the thresholded card image.
            Imgproc.findContours(fr_tmp2, contours, hierarchy, Imgproc.RETR_TREE,
                                 Imgproc.CHAIN_APPROX_SIMPLE);
            // Filter out any interior contours.
            int i = 0;
            for (Iterator<MatOfPoint> it = contours.iterator(); it.hasNext(); i++) {
                hierarchy.get(i, 0, hierData);
                if (hierData[3] >= 0 || Imgproc.contourArea(it.next()) < minShapeArea) {
                    it.remove();
                }
            }
            // Set the remaining attributes: shape and number.
            sc.setShape(contours, 0.01);
            sc.setNumber(contours);
            cards.add(sc);
        }
    }

    private void findCards(Mat rgba) {
        rects.clear();  // store found cards here. TODO: use old positions as priors?
        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        MatOfPoint2f approx = new MatOfPoint2f();

        // convert to HSV space, threshold, and find contours
        Imgproc.cvtColor(rgba, fr_tmp1, Imgproc.COLOR_RGB2HSV);
        Core.inRange(fr_tmp1, cardHSVlb, cardHSVub, fr_tmp1);
        Imgproc.findContours(fr_tmp1, contours, new Mat(), Imgproc.RETR_LIST,
                             Imgproc.CHAIN_APPROX_SIMPLE);

        for (MatOfPoint cnt : contours) {
            // approximate contours to get more regular shapes
            cnt.convertTo(approx, CvType.CV_32FC2);
            double sideErrorThresh = sideErrorScale * Imgproc.arcLength(approx, true);
            Imgproc.approxPolyDP(approx, approx, sideErrorThresh, true);
            // only take contours with 4 sides
            if (approx.total() != 4) continue;
            approx.convertTo(cnt, CvType.CV_32S);
            // apply area, convexity, and right-angle filters
            double cArea = Imgproc.contourArea(approx);
            if (
                    cArea > minRectArea && cArea < maxRectArea &&
                    Imgproc.isContourConvex(cnt) &&
                    maxAngleCos(approx.toArray()) < maxCornerAngleCos
               ) {
                rects.add(cnt);
            }
        }
    }

    /*
        Helpers for findCards
    */
    static double maxAngleCos(Point[] cnt) {
        double maxCos = 0.0;
        for (int i = 2; i < 5; i++) {
            double cosine = Math.abs(angle(cnt[i%4], cnt[i-2], cnt[i-1]));
            if (cosine > maxCos) {
                maxCos = cosine;
            }
        }
        return maxCos;
    }
    static double angle(Point pt1, Point pt2, Point pt0) {
        double dx1 = pt1.x - pt0.x;
        double dy1 = pt1.y - pt0.y;
        double dx2 = pt2.x - pt0.x;
        double dy2 = pt2.y - pt0.y;
        return (dx1*dx2 + dy1*dy2) / Math.sqrt((dx1*dx1 + dy1*dy1) * (dx2*dx2 + dy2*dy2) + 1e-10);
    }
}
