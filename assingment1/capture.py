import cv2 as cv
import time
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--plot', '-p', action='store_true', 
                        help='show plot of FPS')
parser.add_argument('--timeout', '-t', type=int, 
                        help='set program runtime length in seconds')
parser.add_argument('--source', '-s', type=str, choices=['webcam', 'ip'], default='webcam', 
                        help='select the source device')
parser.add_argument('--no_display', action='store_true', 
                        help='turn of display window')
parser.add_argument('--processing', type=str, nargs='*', choices=['brightness', 'redness'], default=['brightness', 'redness'], 
                        help='choose postprocessing steps')
parser.add_argument('--for_loop', action='store_true', 
                        help='process in for loop instead of openCV')

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
    if 'brightness' in args.processing:
        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        _, _, _, point = cv.minMaxLoc(gray_img)
        mark_point(img, point)
    if 'redness' in args.processing:
        blue, green, red = cv.split(img)
        diff = redness(blue, green, red.astype(int)) 
        _, _, _, point = cv.minMaxLoc(diff)
        mark_point(img, point, color=(0,0,255))


def with_for_loop(img):
    max_brightness = 0
    bright_point = (0,0)
    
    max_red = 0
    red_point = (0,0)
    
    for y,row in enumerate(img):
        for x, pixel in enumerate(row):
            if 'brightness' in args.processing:
                brightness = np.sum(pixel) 
                if brightness > max_brightness:
                    max_brightness = brightness
                    bright_point = (x,y) 
            if 'redness' in args.processing:
                red = redness(*pixel.astype(int))
                if red > max_red:
                    max_red = red
                    red_point = (x,y) 
    if 'brightness' in args.processing:      
        mark_point(img, bright_point)
    if 'redness' in args.processing:
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
    print(f'{avg_fps:.2f}')
    
    plt.plot(x, raw_fps, linewidth=0.8)
    plt.plot(x, rolling_fps)
    plt.plot([x[0],x[-1]], [avg_fps, avg_fps])
    
    plt.xlabel("time [s]")
    plt.ylabel("FPS")
    # plt.title("average FPS of a frame with bright spot detection")
    plt.legend(['FPS', f'rolling average FPS', 'total average FPS'])
    plt.show()      

if __name__ == '__main__':
    args = parser.parse_args()
    
    if args.source == 'webcam':
        cap = cv.VideoCapture(0)
    elif args.source == 'ip':
        cap = cv.VideoCapture('https://192.168.1.16:8080/video')
    
    raw_fps = []
    total = 0
    while(True):
        ret, img = cap.read()
        
        
        if args.for_loop:
            with_for_loop(img)
        else:
            with_opencv(img)
        
        fps = add_fps(img)
        raw_fps.append(fps)
        
        if not args.no_display:
            cv.imshow('frame', img)
            
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        
        total += 1/max(1,fps)
        if args.timeout is not None and total > args.timeout:
            print(f'reached timeout of {args.timeout} seconds')
            break

    cap.release()
    cv.destroyAllWindows()
    
    if args.plot:
        make_plots(raw_fps)
