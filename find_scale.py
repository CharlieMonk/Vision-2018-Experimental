import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
img_path = "/Users/cbmonk/Downloads/searched4.png"
img = cv2.imread(img_path, 0)
bgr_img = cv2.imread( img_path )
query = cv2.imread("/Users/cbmonk/Downloads/query2.png",0)
w, h = query.shape[::-1]
# All the 6 algorithms for comparison in a list
time0 = time.time()
alg = eval("cv2.TM_CCOEFF")
# Look for matches
res = cv2.matchTemplate(img, query, alg)
print("Res: ", res)
top_left = cv2.minMaxLoc(res)[3]
#print(min_val, max_val, min_loc, top_left)
bottom_right = (top_left[0] + w, top_left[1] + h)
cv2.rectangle(bgr_img, top_left, bottom_right, (255,0,0), 2)
#cv2.circle(bgr_img, min_loc, 2, (0,255,0), 7)
cv2.circle(bgr_img, top_left, 2, (0,255,0), 7)
print("Time: " + str(time.time()-time0))
cv2.imshow("Detected Points", bgr_img)
cv2.waitKey(0)
