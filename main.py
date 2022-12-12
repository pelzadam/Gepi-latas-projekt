#!/usr/bin/env python3.10

import os
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

	ratio_thresh = 0.7
	good_matches = []
	for m,n in knn_matches:
		if m.distance < ratio_thresh * n.distance:
			good_matches.append(m)

	return good_matches

def circles(img,cannyimg):
	circles = cv.HoughCircles(cannyimg,cv.HOUGH_GRADIENT,1,250,param1=75,param2=40,minRadius=100,maxRadius=200)
	circles = np.uint16(np.around(circles))

	good_circles=[]
	good_circles.append(circles[0][0])
	for circle in circles[0,:]:
		i=0
		for i in range(len(good_circles)):
			distance = np.sqrt((int(circle[0])-int(good_circles[i][0]))**2 + (int(circle[1])-int(good_circles[i][1]))**2)
			if (distance < good_circles[i][2]) or (distance < circle[2]):
				print(distance, " ", good_circles[i][2], " ", circle[2])
				break
		
		if i+1 == len(good_circles):
			good_circles.append(circle)	

		cv.waitKey()

	return good_circles

def imgShow(img1,img2, matches, k1,k2,d1,d2):
	img_keypoints = cv.drawMatches(img1,k1,img2,k2,matches[:20],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

	return img_keypoints

forints=('5','10','20','50','100','200')

target='images/forintok3.jpg'
output='results/'

images=[]
matching=[]

for i in range(0,len(forints)):
	j=1
	images.append([])
	while os.path.isfile('db/'+forints[i]+'/'+str(j)):
		images[i].append('db/'+forints[i]+'/'+str(j))
		j=j+1


target = cv.imread(target)
edge_targ = cv.cvtColor(target, cv.COLOR_BGR2GRAY)
edge_targ = cv.GaussianBlur(edge_targ, (11,11), 0) 
edge_targ = cv.Canny(edge_targ, 30, 150, 3)
edge_targ = cv.dilate(edge_targ, (1,1), iterations = 2)

circles = circles(target.copy(),edge_targ)

cv.imwrite(output+"CannyEdge.jpg", edge_targ)
cv.waitKey()

i=0
for circle in circles:
	print(circle[0]," ",circle[1]," ",circle[2])
	mask = np.zeros(edge_targ.shape[:2], dtype="uint8")
	#cv.circle(img,(circle[0],circle[1]),circle[2],(0,255,0),2)
	#cv.circle(img,(circle[0],circle[1]),2,(0,0,255),3)
	cv.circle(mask, (circle[0], circle[1]), circle[2], 255, -1)
	cut = cv.bitwise_and(edge_targ, edge_targ, mask=mask)

	cv.imwrite(output+str(i+1)+". cut.jpg", cut)
	matching.append([circle,[]])

	j=0
	for forint in images:
		value=0
		k=1
		for image in forint:
			print(image)
			sample = cv.imread(image,cv.IMREAD_GRAYSCALE)
			res_sample = cv.resize(sample, (256,int(sample.shape[1]/sample.shape[0]*256)))
			edge_samp = cv.GaussianBlur(res_sample, (5,5), 1) 
			edge_samp = cv.Canny(edge_samp, 30, 150, 3)
			edge_samp = cv.dilate(edge_samp, (1,1), iterations = 2)
			cv.imwrite(output+str(i+1)+'.'+forints[j]+"."+str(k)+". sample.jpg", edge_samp)

			k1,k2,d1,d2=sift(cut,edge_samp)
			matches = flann(d1,d2)

			print(len(matches))
			value += len(matches)

			result=imgShow(target,res_sample,matches,k1,k2,d1,d2)
			result = cv.resize(result, (1600,900))
			cv.imwrite(output+str(i+1)+'.'+forints[j]+"."+str(k)+". keypoints.jpg", result)
			#cv.waitKey()
			k+=1

		value = value / k
		print("Average of ",forint,": ", value)
		matching[i][1].append(value)
		j+=1

	largest_value=max(matching[i][1])
	indexof_largest =forints[matching[i][1].index(largest_value)]

	cv.putText(target,indexof_largest,(circle[0],circle[1]),cv.FONT_HERSHEY_SIMPLEX,1,(255, 255, 0, 0),3)
	cv.circle(target,(circle[0],circle[1]),circle[2],(0,255,0),2)
	#cv.imshow('',target)
	i+=1

cv.imwrite(output+"endresult.jpg", target)

print(matching)
