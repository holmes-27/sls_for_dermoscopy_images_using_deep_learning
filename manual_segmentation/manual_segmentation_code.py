import cv2
import numpy as np
import os

def nothing(x):
    pass

def getContours(edge,imgCpy):
    contours, hierarchy = cv2.findContours(edge,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        cv2.drawContours(imgCpy, cnt, -1, (0,255,0))


cv2.namedWindow("trackbar")

cv2.createTrackbar("h min", "trackbar", 0, 179, nothing)
cv2.createTrackbar("h max", "trackbar", 255, 255, nothing)
cv2.createTrackbar("s min", "trackbar", 0, 255, nothing)
cv2.createTrackbar("s max", "trackbar", 255, 255, nothing)
cv2.createTrackbar("v min", "trackbar", 0, 255, nothing)
cv2.createTrackbar("v max", "trackbar", 255, 255, nothing)
cv2.createTrackbar("index", "trackbar", 0, 5, nothing)

path = r"C:/Users/Harsha/Downloads/ISIC-images/"
ld = os.listdir(path)

ind = 0

while True:
    ind = cv2.getTrackbarPos("index","trackbar")
    img = cv2.imread(path + ld[ind])
    img = cv2.resize(img,(300,300))
    imgCpy = img.copy()
    
    hmin = cv2.getTrackbarPos("h min","trackbar")
    hmax = cv2.getTrackbarPos("h max","trackbar")
    smin = cv2.getTrackbarPos("s min","trackbar")
    smax = cv2.getTrackbarPos("s max","trackbar")
    vmin = cv2.getTrackbarPos("v min","trackbar")
    vmax = cv2.getTrackbarPos("v max","trackbar")
    
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    lower = np.float32([hmin,smin,vmin])
    upper = np.float32([hmax,smax,vmax])
    
    mask = cv2.inRange(imgHSV, lower, upper)
    maskCpy = cv2.merge([mask,mask,mask])
    
    segmented = cv2.bitwise_and(img, img,mask=mask)
    
    imgCanny = cv2.Canny(mask, 50, 50)
    getContours(imgCanny, imgCpy)
    
    output = np.hstack((img,segmented,imgCpy,maskCpy))
    
    cv2.imshow("Output",output)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()