#!/usr/bin/env python3.10

import argparse as arg
import numpy as np
import cv2 as cv

def sift(img1, img2):
	detector = cv.SIFT_create()
	
	k1, d1 = detector.detectAndCompute(img1, None)
	k2, d2 = detector.detectAndCompute(img2, None)

	return k1, k2, d1, d2

def flann(d1,d2):
	matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
	knn_matches = matcher.knnMatch(d1, d2, 2)

	ratio_thresh = 0.65
	good_matches = []
	for m,n in knn_matches:
		if m.distance < ratio_thresh * n.distance:
			good_matches.append(m)

	return good_matches

def imgShow(img1,img2, matches, k1,k2,d1,d2):
	img_keypoints = cv.drawMatches(img1,k1,img2,k2,matches,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

	return img_keypoints

target='images/forintok3.jpg'

images=['db/10forint.jpg',
		'db/50forint.png',
		'db/100forint.png',
		'db/100forint2.jpg',
		'db/100forint3.png']

kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])

target = cv.imread(target, cv.IMREAD_GRAYSCALE)
target = cv.GaussianBlur(target, (3,3), 0) 
edge_targ = cv.Canny(image=target, threshold1=100, threshold2=180)

for image in images:
	sample = cv.imread(image,cv.IMREAD_GRAYSCALE)
	sample = cv.resize(sample, (128,int(sample.shape[1]/sample.shape[0]*128)))
	sample = cv.GaussianBlur(sample, (3,3), 0) 
	edge_samp = cv.Canny(image=sample, threshold1=100, threshold2=180)
	
	k1,k2,d1,d2=sift(edge_targ,edge_samp)
	matches = flann(d1,d2)

	print(len(matches))

	result=imgShow(target,sample,matches,k1,k2,d1,d2)
	result = cv.resize(result, (1600,900))
	cv.imshow('', result)
	cv.waitKey()



