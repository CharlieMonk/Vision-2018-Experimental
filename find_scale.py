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
img2 = cv2.imread('/Users/cbmonk/Downloads/searched5.png',0) # trainImage
img2_bgr = cv2.imread('/Users/cbmonk/Downloads/searched5.png')

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
retval, mask = cv2.findHomography(queryPts, inputPts, cv2.LMEDS, 5.0)
matchesMask = mask.ravel().tolist()
maskedPts = np.array([-1,-1])
for i in range(len(inputPts)):
    if(matchesMask[i] == 1):
        #cv2.circle(img2_bgr, tuple(inputPts[i][0]), 5, (0,0,255), -1)
        maskedPts = np.vstack((maskedPts,inputPts[i][0]))
maskedPts = maskedPts[1:]
rectPt1_arr = np.nanmin(maskedPts, axis=0)
rectPt1 = (int(rectPt1_arr[0]), int(rectPt1_arr[1]))
rectPt2_arr = np.nanmax(maskedPts, axis=0)
rectPt2 = (int(rectPt2_arr[0]), int(rectPt2_arr[1]))
#print(maskedPts)
# for pt in inputPts:
#     cv2.circle(img2_bgr, tuple(pt[0]), 5, (0,0,255), -1)
#print("Mask: " + str(mask.shape) + ", inputPts: " + str(inputPts.shape))
#print(inputPts[0])

h,w = img1.shape
print("h: " + str(h) + " w: " + str(w))
pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
dst = cv2.perspectiveTransform(pts,retval)
cv2.rectangle(img2_bgr, rectPt1, rectPt2, (0, 255, 0), 3)

time3 = time.time()
print("Time check 3:" + str(time3-time2))
# Time Section 4----------------------------------------------------------------

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

img3 = cv2.drawMatches(img1,keypts1,img2,keypts2,good_matches,None,**draw_params)

time4 = time.time()
print("Time check 4:" + str(time4-time3))
print("Total time: " + str(time4-time0))

plt.imshow(cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2RGB))
plt.show()
plt.imshow(img3, "gray")
plt.show()
