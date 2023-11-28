Use OpenCV and Python to construct and demonstrate real-time detection and localization of a prominent line in live video.
----
Detection and localization of a prominent straight edge in live video

1. Capture an image from a video camera.
1. Use an edge detector (such as Canny) to create an edge image.
1. Write the (x,y) coordinates of (all) edge pixels into an array.
1. Use RANSAC to fit a straight line with the greatest support to the extracted edge pixels.
1. Display the line in the live image.
1. Adjust the parameters of the edge detector and the RANSAC algorithm for best results at video rate.

-----

For testing, you can check:

- How well your straight line detector follows the edge of a sheet of paper moved across the camera's field of view.
- How well does it detect other straight lines in your environment?
- The processing time for one video frame or image.

----
NOTE:
The following measures can be used to increase the frame rate:
- Reducing the size of the input image.
- Adjusting the Canny parameters to reduce the ratio of outliers to inliers.
- Checking a subset of the detected edge points (i.e., every k-th point) for support in RANSAC.

Demonstrate the running code on your computer in class.

Submit a git-link to the code or upload it using the submit button.

This is an individual assignment.  You are encouraged to solicit help from your colleagues, but not to copy code.