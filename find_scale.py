import numpy as np
import cv2
from matplotlib import pyplot as plt
import time

def findSIFT(img1, img2):
    # Set up SIFT
    detector = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    keypts1, descriptors1 = detector.detectAndCompute(img1, None)
    keypts2, descriptors2 = detector.detectAndCompute(img2, None)
    return keypts1, keypts2, descriptors1, descriptors2

def findSIFTMatches(descriptors1, descriptors2):
    kdtree = 1
    queryParams = dict(algorithm=kdtree, trees=5)
    inputParams = dict(checks=50)
    matcher = cv2.FlannBasedMatcher(queryParams, inputParams)
    matches = matcher.knnMatch(descriptors1, descriptors2, k=2)


    good_matches = []
    for i,j in matches:
        if(i.distance < 0.8*j.distance):
            good_matches.append(i)
    return good_matches



# Time Section 1----------------------------------------------------------------
time0 = time.time()
# Define the minimum match count to be 10
# Read the images
img1 = cv2.imread('/Users/cbmonk/Downloads/query2.png',0)          # queryImage
img2 = cv2.imread('/Users/cbmonk/Downloads/searched4.png',0) # trainImage
img2_bgr = cv2.imread('/Users/cbmonk/Downloads/searched4.png')

keypts1, keypts2, descriptors1, descriptors2 = findSIFT(img1, img2)

time1 = time.time()
print("Time check 1:" + str(time1-time0))
# Time Section 2----------------------------------------------------------------

good_matches = findSIFTMatches(descriptors1, descriptors2)

time2 = time.time()
print("Time check 2:" + str(time2-time1))
# Time Section 3----------------------------------------------------------------
queryPts = np.float32([keypts1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
inputPts = np.float32([keypts2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)
retval, mask = cv2.findHomography(queryPts, inputPts, cv2.RANSAC,5.0)
matchesMask = mask.ravel().tolist()
h,w = img1.shape
pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
dst = cv2.perspectiveTransform(pts,retval)
cv2.rectangle(img2, (100,100), (200,200), 255, 3)
img2_bgr = cv2.polylines(img2_bgr,[np.int32(dst)],True,(0,0,255),3, cv2.LINE_AA)

time3 = time.time()
print("Time check 2:" + str(time3-time2))
# Time Section 4----------------------------------------------------------------

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

img3 = cv2.drawMatches(img1,keypts1,img2,keypts2,good_matches,None,**draw_params)

time4 = time.time()
print("Time check 2:" + str(time4-time3))
print("Total time: " + str(time4-time0))

cv2.imshow("BGR Searched Image", img2_bgr)
plt.imshow(img3, 'gray'),plt.show()
