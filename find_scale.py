import cv2
import numpy as np
from matplotlib import pyplot as plt
import time

def searchImage(img, query):
    alg = eval("cv2.TM_CCOEFF")
    res = cv2.matchTemplate(img, query, alg)
    start_pt = cv2.minMaxLoc(res)[3]
    return start_pt

img_path = "/Users/cbmonk/Downloads/searched4.png"
img = cv2.imread(img_path, 0)
bgr_img = cv2.imread( img_path )
query = cv2.imread("/Users/cbmonk/Downloads/query2.png",0)
w, h = query.shape[::-1]
# All the 6 algorithms for comparison in a list
time0 = time.time()
# Look for matches
top_left = searchImage(img, query)
flip_img = cv2.flip(img, 1)
top_right = searchImage(flip_img, query)
#print(min_val, max_val, min_loc, top_left)
bottom_right = (img.shape[0] - top_right[0], top_left[1] + h)
cv2.rectangle(bgr_img, top_left, bottom_right, (255,0,0), 2)
cv2.circle(flip_img, (img.shape[0]*2-top_left[0], top_left[1]), 2, (0,255,0), 7)
cv2.circle(flip_img, top_right, 2, (0,255,0), 7)
cv2.imshow("Flipped", flip_img)
print("Time: " + str(time.time()-time0))
cv2.imshow("Detected Points", bgr_img)
cv2.waitKey(0)
