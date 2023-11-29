import cv2 as cv
import numpy as np
import time 
import random
import argparse
import matplotlib.pyplot as plt
from math import tan, atan

parser = argparse.ArgumentParser()
parser.add_argument('-show', type=str, nargs='*', choices=['edges', 'lines', 'rect'], default=['rect'], 
                        help='sets parts of post-processing to be displayed')
parser.add_argument('-img_size', type=int, nargs=2, default=[640,480], metavar=('WIDTH', 'HEIGHT'),
                        help='sets image size')
parser.add_argument('-canny', type=int, nargs=2, default=[100, 200], metavar=('LOW_THRESHOLD', 'HIGH_THRESHOLD'),
                        help='low and hight threshold for Canny edge detection')


font = cv.FONT_HERSHEY_SIMPLEX 
now = time.time()
# rng = np.random.default_rng()

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


def draw_line(img, line_parameters):
    a,b,c = line_parameters
    if abs(a) < 0.0001:
        a = 0.0001
    y0 = int(0)
    y1 = int(img.shape[1]) 
    x0 = int(-c/a)
    x1 = int(-y1*b/a-c/a)
    cv.line(img, [x0,y0], [x1,y1], (255,0,0), 1) 


def get_lines(edge_img):
    lines = cv.HoughLines(edge_img, rho=1, theta=np.pi/180 , threshold=80)
    line_factors = []
    if lines is None or len(lines) < 4:
        return None
        
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))    
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        a,b,c = np.cross([x1,y1,1], [x2,y2,1])
        norm_factor = np.linalg.norm([a,b])
        line_factors.append(np.array([a,b,c])/norm_factor)
    
    return line_factors
    
    
def remove_close_lines(lines):
    if lines is None:
        return None
    new_lines = [lines[0]]
    for line in lines:
        accept = True
        for line_ in new_lines:
            dot = np.dot(line[:2], line_[:2])
            if abs(dot) > 0.9:
                c = line[2] if dot > 0 else -line[2]
                c_ = line_[2]
                if abs(c-c_) < 20:
                    accept = False
        if accept:
            new_lines.append(line)
    return new_lines

def get_intersections(lines):
    if len(lines) < 4:
        return None
    lines = lines[:4]
    similarity = np.zeros((4,4))
    for row,line in enumerate(lines):
        for col,line_ in enumerate(lines):
            similarity[row,col] = abs(np.dot(line[:2], line_[:2]))
    order = np.argsort(similarity)
    intersection_points = []
    intersection_points_set = set()
    for row in range(4):
        line = lines[row]
        for col in [0,1]:
            point = np.cross(line, lines[order[row][col]])
            point = point/point[2]
            point = point[:2].astype(int)
            if tuple(point) not in intersection_points_set:  
                intersection_points_set.add(tuple(point)) 
                intersection_points.append(point)
    intersection_points = np.array(intersection_points)
    if len(intersection_points) == 4:
        return intersection_points
    else:
        return None
    
def warp_image(img, points, img_size):
    points = sorted(points, key=lambda x: x[0])
    if points[0][1] > points[1][1]:
        points[0], points[1] = points[1], points[0]
    if points[2][1] > points[3][1]:
        points[2], points[3] = points[3], points[2]
    resmatrix = cv.getPerspectiveTransform(np.float32(points), np.float32([[0,0],[0,480],[640,0],[640,480]]))
    warp_img = cv.warpPerspective(img, resmatrix, img_size)
    if 'rect' in args.show:
        cv.line(img, points[0], points[1], (0,0,255), 2)
        cv.line(img, points[0], points[2], (0,0,255), 2)
        cv.line(img, points[1], points[3], (0,0,255), 2)
        cv.line(img, points[2], points[3], (0,0,255), 2)
    return warp_img



cap = cv.VideoCapture(0)
args = parser.parse_args()
fps_a = []
warped_img = np.zeros((args.img_size[1],args.img_size[0],3), dtype=np.uint8)

while(True):
    ret, img = cap.read() 
    if img.shape[0] != args.img_size[1] or img.shape[1] != args.img_size[0]:
        img = cv.resize(img, args.img_size)
    
    grey_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    edge_img = cv.Canny(grey_img, args.canny[0], args.canny[1])
    if 'edges' in args.show:
        img[:,:,1] = np.maximum(img[:,:,1], edge_img)   
    
    lines = get_lines(edge_img)
    lines = remove_close_lines(lines)
    if lines is not None:
        if 'lines' in args.show:
            for line in lines:
                draw_line(img, line)
        points = get_intersections(lines)
        if points is not None:
            warped_img = warp_image(img, points, args.img_size)
            
    img = np.concatenate((img, warped_img), axis=1)
    
    fps = add_fps(img)
    fps_a.append(fps)
    cv.imshow('frame', img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
   
cap.release()
cv.destroyAllWindows()

    