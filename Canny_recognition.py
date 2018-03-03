import numpy as np
import cv2

img = cv2.imread("/Users/cbmonk/Downloads/searched4.png")
canny = cv2.Canny(img, 100, 150)
cv2.imshow("Canny", canny)

cv2.waitKey(0)
