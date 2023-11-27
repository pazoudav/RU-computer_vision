Assignment 1: OpenCV setup and point operations
===============================================

Install OpenCV on your computer from binaries or compile it from the GitHub repository.

Use OpenCV and Python (or C++) to construct and demonstrate real-time detection of a bright spot in the image.

1. Capture an image from the laptop video camera.
1. Display the image.  Repeat steps 1. - 2. continuously in a loop to show a real-time video.
1. Measure the time spent capturing and processing each frame and display this in the image as frames per second (FPS).
1. Locate the brightest spot and mark the brightest spot in the image, using OpenCV functions.
1. Locate and mark the reddest spot in the image.  How do you define "reddest"?
1. Repeat 4. and 5. above, but now search for the brightest spot in the image by going through each pixel in the image in a double for-loop.
1. Run your code on live video streamed to a computer from a mobile phone (IP camera). 


For testing, check and write down answers to the following questions:
- The processing time for one video frame or image.
- How does the processing time change when you add the bright spot detection?
- Is the processing time identical when you do not display the image?
- How does your for-loop implementation compare to the built-in function?
- Moving your hand in front of the camera, estimate the latency between image capture and display.
- Is the latency different when capturing from a mobile phone?

Notes:
- You may get better performance when you compile a Release rather than Debug version of the code (C++).
- If you have extracted the official OpenCV binaries, you may find that python does not find cv2.  You can run opencv\build\setup_vars_opencv4.cmd or use the Windows Environment Variables dialog to add the location of opencv\build\python to the PYTHONPATH.
- Demonstrate the running code on your computer and submit a link to a github repo containing the source file(s) and a PDF file with your observations.