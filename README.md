set-finder
==========

Android app for detecting sets in the card game Set.

### Android app

The following instructions assume that you have the Android SDK and tools set up already.

**Eclipse + ADT instructions**

 1. Import `set-finder/android_app` as an Android Application project.
 2. Follow the [OpenCV for Android instructions](http://docs.opencv.org/doc/tutorials/introduction/android_binary_package/O4A_SDK.html#manual-opencv4android-sdk-setup).
 3. Add OpenCV as a dependency of SetFinder.

**Command-line instructions**

 1. Download [OpenCV for Android](http://sourceforge.net/projects/opencvlibrary/files/opencv-android/) (*should look like `OpenCV-$version-android-sdk.zip`*).
 2. Unzip it, then either:
    * `mv $path_to_opencv/sdk set-finder/android_app/lib/opencv`, or
    * `cat 'android.library.reference.1=$path_to_opencv/sdk/java' >> set-finder/android_app/local.properties`
 3. Build with `ant debug` or `ant release`, from `set-finder/android_app`.

### Python prototype

 1. Make sure [numpy](http://www.numpy.org/) and (desktop) [OpenCV](http://opencv.org/) are installed.
 2. Run `python prototype.py`.
