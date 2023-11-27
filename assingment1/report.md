Assignment 1: OpenCV setup and point operations
===============================================

David Pažout
------------

For testing, check and write down answers to the following questions:
- The processing time for one video frame or image?
- How does the processing time change when you add the bright spot detection?
- Is the processing time identical when you do not display the image?
- How does your for-loop implementation compare to the built-in function?
- Moving your hand in front of the camera, estimate the latency between image capture and display.
- Is the latency different when capturing from a mobile phone?

------



The processing time for one video frame is around 120 FPS. 
![alt text](images/average_FPS.png)

Adding the bright spot detection increases the processing time for the video frame to 114 FPS.
![alt text](images/average_FPS_bsd.png)

When not displaying the image the processing time increases to 12 ms or 80 FPS. The increase in time needed to process a frame is likely caused by printing of the FPS number to the console, thus adding additional overhead.
![alt text](images/average_FPS_no_display.png)

Moving the processing to a double for loop decreases the FPS to 0.
![alt text](images/average_FPS_for_loop.png)

I estimate the latency between image capture and display to be around 0.1 second but I have no process of validating my estimate.

The latency from a mobile phone is around 1 second.