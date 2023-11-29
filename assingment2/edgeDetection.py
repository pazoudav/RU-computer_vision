import cv2 as cv
import numpy as np
import time 
import random
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--plot', '-p', action='store_true', 
                        help='show plot of FPS')
parser.add_argument('-show', type=str, nargs='*', choices=['edges', 'inliers', 'line'], default=['edges', 'line'], 
                        help='sets parts of post-processing to be displayed')
parser.add_argument('-img_size', type=int, nargs=2, default=[640,480], metavar=('WIDTH', 'HEIGHT'),
                        help='sets image size')
parser.add_argument('-canny', type=int, nargs=2, default=[100, 200], metavar=('LOW_THRESHOLD', 'HIGH_THRESHOLD'),
                        help='low and hight threshold for Canny edge detection')
parser.add_argument('-RANSAC', type=int, nargs=2, default=[512, 4], metavar=('ITERATIONS', 'DELTA'),
                        help='number of iterations and delta for RANSAC line fitting')

font = cv.FONT_HERSHEY_SIMPLEX 
now = time.time()
rng = np.random.default_rng()

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
    print(f'average FPS={avg_fps:.2f}')
    
    plt.plot(x, raw_fps, linewidth=0.8)
    plt.plot(x, rolling_fps)
    plt.plot([x[0],x[-1]], [avg_fps, avg_fps])
    
    plt.xlabel("time [s]")
    plt.ylabel("FPS")
    plt.title("FPS plot")
    plt.legend(['FPS', f'rolling average FPS', 'total average FPS'])
    plt.show()  


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
    cv.putText(img, f'FPS={fps:.2f}', (10,30), font, 1, (0,0,255), 2, cv.LINE_AA)
    return fps


def RANSAC(points, iterations, delta):
    max_hit_count = 0
    best_fit = [0,0,0]
    best_hit_points = np.array([[0,0]])
    if len(points) == 0:
        return best_fit, best_hit_points

    rnd_idxs1 = rng.integers(low=0, high=len(points), size=iterations)  
    rnd_idxs2 = rng.integers(low=0, high=len(points), size=iterations)
    rnd_p1 = points[rnd_idxs1]  
    rnd_p2 = points[rnd_idxs2] 
    lines = np.cross(rnd_p1, rnd_p2)
    norm_factors = np.linalg.norm(lines[:,:2],axis=1)
    lines = lines/norm_factors.reshape((norm_factors.shape[0],1))
    distances = np.abs(points @ lines.T)
    hit_points = distances < delta
    hit_count = np.sum(hit_points,axis=0)
    idx = np.argmax(hit_count)
    best_fit = lines[idx]
    best_hit_points = points[hit_points.T[idx]][:,:2]
    return best_fit, best_hit_points
    
    # for k in range(iterations):
    #     p1, p2 = points[rnd_idxs1[k]], points[rnd_idxs2[k]]
    #     a, b, c = np.cross(p1,p2)
    #     norm_factor = (a**2+b**2)**0.5
    #     distances = np.abs(points @ [a,b,c])/norm_factor
    #     hit_points = distances < delta
    #     hit_count = np.sum(hit_points)
        
    #     if hit_count > max_hit_count:
    #         max_hit_count = hit_count
    #         best_fit = np.array([a,b,c])/norm_factor
    #         best_hit_points = points[hit_points]
    # return best_fit, best_hit_points[:,:2]


def draw_line(img, line_parameters):
    b,a,c = line_parameters
    if abs(a) < 0.0001:
        a = 0.0001
    y0 = int(0)
    y1 = int(img.shape[1]) 
    x0 = int(-c/a)
    x1 = int(-y1*b/a-c/a)
    cv.line(img, [x0,y0], [x1,y1], (0,0,255), 1) 



cap = cv.VideoCapture(0)
args = parser.parse_args()
fps_a = []
idx_array = np.array([[[y,x,1] for x in range(args.img_size[0])] for y in range(args.img_size[1])])

while(True):
    ret, img = cap.read() 
    if img.shape[0] != args.img_size[1] or img.shape[1] != args.img_size[0]:
        img = cv.resize(img, args.img_size)
    
    grey_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    edge_img = cv.Canny(grey_img, args.canny[0], args.canny[1])
    edge_pixels = idx_array[edge_img == 255]
    line_parameters, hit_points = RANSAC(edge_pixels, iterations=args.RANSAC[0], delta=args.RANSAC[1])
    
    if 'edges' in args.show:
        img[:,:,1] = np.maximum(img[:,:,1], edge_img)
        cv.putText(img, 'Canny edges', (10,60), font, 0.8, (0,255,0), 1, cv.LINE_AA)
    
    if 'inliers' in args.show:
        img[*hit_points.T] = [255,0,0]
        cv.putText(img, 'RANSAC inliers', (10,90), font, 0.8, (255,0,0), 1, cv.LINE_AA)
    
    if 'line' in args.show:
        draw_line(img, line_parameters)
        cv.putText(img, 'best line', (10,120), font, 0.8, (0,0,255), 1, cv.LINE_AA)
    
    fps = add_fps(img)
    fps_a.append(fps)
    
    cv.imshow('frame', img)  
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
   
cap.release()
cv.destroyAllWindows()

if args.plot:
    make_plots(fps_a)
    