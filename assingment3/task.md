Assignment 3: Real-time slide document viewer
=============================================

Use OpenCV and Python to construct and demonstrate real-time detection and rectification of a prominent rectangular shape in live video.

Part I - Detecting the object boundary
-------------------
Detection and localization of a prominent straight edge in live video
1. Capture an image from a video camera.
1. Use an edge detector (such as Canny) to create an edge image.
1. Use the Hough Transform (in OpenCV) to locate four prominent lines in the image.
1. Display the lines in the live image.
1. Adjust the parameters of the edge detector and the Hough Transform for best results at video rate.

Part II - Rectification
-----------------------
1. Compute and enumerate the intersections of the lines defining the four corners of the quadrangle.
1. Using the four corner locations, create a perspective transformation that maps to the corners of a new image, and warp the image content to the new image.
1. Display the rectified image.

For testing you can check:
- How well your straight line detector follows the edge of a sheet of paper moved across the field of view of the camera.
- How well it detects other straight lines in your environment.
- The processing time for one video frame or image.

Notes:
- From the lecture slides (L2.3), we have that the cross product of two lines gives the point of intersection in homogeneous coordinates. Divide by the last coordinate to obtain the pixel coordinates.
- There is an example of how to use OpenCV functions to compute the homography and warp the images here: https://www.learnopencv.com/homography-examples-using-opencv-python-c/ (Links to an external site.)

Demonstrate the running code on your computer in class.

Submit a git-link to the code or upload it using the submit button.

This is an individual assignment.  You are encouraged to solicit help from your colleagues but not to copy code.