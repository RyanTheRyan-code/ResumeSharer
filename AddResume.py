import cv2
import numpy as np

# cap = cv2.VideoCapture(1)
cap = cv2.VideoCapture('PXL_20241104_031356957.mp4')
# if not cap.isOpened():
#     print("Error opening video stream or file")
#     exit()
imgTarget = cv2.imread('PXL_20241104_031352252.MP~2.jpg')
myVid = cv2.VideoCapture('PXL_20241104_031356957.mp4')
resume = cv2.imread('Resume - Ryan Makela 2024-1.png')

detection = False
frameCounter = 0

success, imgVideo = myVid.read()

hT, wT, cT = imgTarget.shape
imgVideo = cv2.resize(imgVideo, (wT, hT))

orb = cv2.ORB_create(nfeatures = 7500)
kp1, des1 = orb.detectAndCompute(imgTarget, None)
# imgTarget = cv2.drawKeypoints(imgTarget, kp1, None)
scale_factor = 0.5  # Adjust this as needed
# imgTarget = cv2.resize(imgTarget, (0, 0), fx=scale_factor, fy=scale_factor)
# imgVideo = cv2.resize(imgVideo, (0, 0), fx=scale_factor, fy=scale_factor)

while True:
    success,imgWebcam = cap.read()
    imgAug = imgWebcam.copy()
    # if not success:
    #     print("Error reading webcam")
    #     exit()
    kp2, des2 = orb.detectAndCompute(imgWebcam, None)
    # imgWebcam = cv2.drawKeypoints(imgWebcam, kp2, None)



    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)
    # print(len(good))
    imgFeatures = cv2.drawMatches(imgTarget, kp1, imgWebcam, kp2, good, None, flags=2)

    if len(good) > 20:
        detection = True
        srcPts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dstPts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        matrix, mask = cv2.findHomography(srcPts, dstPts, cv2.RANSAC, 5.0)
        # print(matrix)

        pts = np.float32([[0,0],[0,hT],[wT,hT],[wT,0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, matrix)
        img2= cv2.polylines(imgWebcam,[np.int32(dst)],True,(255,0,255),3)

        imgWarp = cv2.warpPerspective(resume, matrix, (imgWebcam.shape[1], imgWebcam.shape[0]))

    maskNew = np.zeros((imgWebcam.shape[0],imgWebcam.shape[1]),np.uint8)
    cv2.fillPoly(maskNew,[np.int32(dst)],(255,255,255))
    maskInv = cv2.bitwise_not(maskNew)
    imgAug = cv2.bitwise_and(imgAug, imgAug, mask=maskInv)
    imgAug = cv2.bitwise_or(imgWarp, imgAug)

    # imgFeatures_resized = cv2.resize(imgFeatures, (0, 0), fx=scale_factor, fy=scale_factor)
    # imgWarp = cv2.resize(imgWarp, (0, 0), fx=scale_factor, fy=scale_factor)
    # imgWebcam = cv2.resize(imgWebcam, (0, 0), fx=scale_factor, fy=scale_factor)
    # maskNew = cv2.resize(maskNew, (0, 0), fx=scale_factor, fy=scale_factor)
    imgAug = cv2.resize(imgAug, (0, 0), fx=scale_factor, fy=scale_factor)

    cv2.imshow('imgAug', imgAug)
    # cv2.imshow('maskNew', maskNew)
    # cv2.imshow('imgWarp', imgWarp)
    # cv2.imshow('img2', img2)
    # cv2.imshow('imgFeatures', imgFeatures_resized)
    # cv2.imshow('ImgTarget', imgTarget)
    # cv2.imshow('myVid', imgVideo)
    # cv2.imshow('Webcam', imgWebcam)
    cv2.waitKey(1)
