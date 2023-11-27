import cv2 as cv
import time
import numpy as np
import matplotlib.pyplot as plt

font = cv.FONT_HERSHEY_SIMPLEX 
now = time.time()

# returns elapsed time from last tick
def tick():
    global now
    delta = time.time() - now
    now = time.time()
    if delta <  0.001:
        delta = 0.001
    return delta

def add_fps(img):
    fps = 1/tick() # int(1/tick()) # 
    cv.putText(img, f'{fps:.2f}', (10,30), font, 1, (0,0,255), 2, cv.LINE_AA)
    return fps

def mark_point(img, point, color=(0,255,0)):
    markerType = cv.MARKER_CROSS
    markerSize = 15
    thickness = 2
    cv.drawMarker(img, point, color, markerType, markerSize, thickness)

def redness(b,g,r):
    return r - (b+g)/2

def with_opencv(img):
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, _, _, point = cv.minMaxLoc(gray_img)
    mark_point(img, point)
    
    # blue, green, red = cv.split(img)
    # diff = redness(blue, green, red.astype(int)) 
    # _, _, _, point = cv.minMaxLoc(diff)
    # mark_point(img, point, color=(0,0,255))


def with_for_loop(img):
    max_brightness = 0
    bright_point = (0,0)
    
    max_red = 0
    red_point = (0,0)
    
    for y,row in enumerate(img):
        for x, pixel in enumerate(row):
            brightness = np.sum(pixel) 
            if brightness > max_brightness:
                max_brightness = brightness
                bright_point = (x,y) 
            
            red = redness(*pixel.astype(int))
            if red > max_red:
                max_red = red
                red_point = (x,y) 
                
    mark_point(img, bright_point)
    mark_point(img, red_point, color=(0,0,255))
          
          
def make_plots(raw_fps):
    #create x axis
    x = [0]
    for idx, fps in enumerate(raw_fps):
        x.append(x[idx]+1/max(fps,1))
    x = x[1:]

    # get rolling average
    rolling_fps = []
    window_size = 16
    for idx, fps in enumerate(raw_fps):
        avg = []
        for offset in range(-window_size//2,window_size//2):
            if idx+offset >= 0 and idx+offset < len(raw_fps):
                avg.append(raw_fps[idx+offset])
        rolling_fps.append(np.average(avg))

    # total average fps
    avg_fps = np.average(raw_fps)
    print(avg_fps)
    plt.plot(x, raw_fps, linewidth=0.8)
    plt.plot(x, rolling_fps)
    plt.plot([x[0],x[-1]], [avg_fps, avg_fps])
    
    plt.xlabel("time [s]")
    plt.ylabel("FPS")
    # plt.title("average FPS of a frame with bright spot detection")

    plt.legend(['FPS', f'rolling average FPS', 'total average FPS'])

    plt.show()      

#local webcam connection
cap = cv.VideoCapture(0)

# IP cam connection
# cap = cv.VideoCapture('https://192.168.1.16:8080/video')

raw_fps = []
total = 0
while(True):
    ret, img = cap.read()
    
    # with_opencv(img)
    with_for_loop(img)
    
    fps = add_fps(img)
    raw_fps.append(fps)
    cv.imshow('frame', img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    
    total += 1/max(1,fps)
    if total > 10:
        break

cap.release()
cv.destroyAllWindows()

make_plots(raw_fps)
