#!/usr/bin/env python3.10

import os
import argparse as arg
import datetime as dt
import numpy as np
import cv2 as cv

def sift(img1, img2):
	detector = cv.SIFT_create()
	
	k1, d1 = detector.detectAndCompute(img1, None)
	k2, d2 = detector.detectAndCompute(img2, None)

	return k1, k2, d1, d2

def flann(d1,d2):
	index_params = dict(algorithm = 1, trees = 5)
	search_params = dict(checks=50)
	flann = cv.FlannBasedMatcher(index_params,search_params)
	knn_matches = flann.knnMatch(d1, d2, 2)

	ratio_thresh = 0.7
	good_matches = []
	for m,n in knn_matches:
		if m.distance < ratio_thresh * n.distance:
			good_matches.append(m)

	return good_matches

def circles(img,r):
	minrad=r
	maxrad=minrad*2
	distance=int(minrad*2.5)
	circles = cv.HoughCircles(img,cv.HOUGH_GRADIENT,1,distance, \
						   param1=75,param2=40,minRadius=minrad,maxRadius=maxrad)
	circles = np.uint16(np.around(circles))

	return circles[0]

def showMatches(img1,img2, matches, k1,k2,d1,d2):
	img_keypoints = cv.drawMatches(img1,k1,img2,k2,matches[:20],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

	return img_keypoints

parser = arg.ArgumentParser()
parser.add_argument("source",help="Path of the source file.")
parser.add_argument("minrad",help="Minimum radius of the Hough Circles.", type=int)
parser.add_argument("gauss",help="Kernel size of Gaussian blur.", type=int)
args=parser.parse_args()

name=args.source
radius=args.minrad
gauss=args.gauss

timestamp = dt.datetime.now()
forints=('5','10','20','50','100','200')
images=[]
matching=[]

output='results/' + str(timestamp.year) + str(timestamp.month) + str(timestamp.day) + \
	str(timestamp.hour) + str(timestamp.minute) + str(timestamp.second) + '/'
cutpath = output + 'cuts/'
samplepath = output + 'samples/'
matchpath = output + 'matches/'
os.makedirs(output)
os.makedirs(cutpath)
os.makedirs(samplepath)
os.makedirs(matchpath)

log=open(output+'log','a')
log.write("--- Coin detection on {} ---\n\
Date: {}\n\
Minimum radius: {} \n\
Gauss kernel size: {}\n\n\
".format(name,timestamp,radius,gauss))

for i in range(0,len(forints)):
	j=1
	images.append([])
	while os.path.isfile('db/'+forints[i]+'/'+str(j)):
		images[i].append('db/'+forints[i]+'/'+str(j))
		j=j+1


target = cv.imread(name)
blurred = cv.GaussianBlur(target, (gauss,gauss), 0)
edge_targ = cv.Canny(blurred, 30, 150, 2)
edge_targ = cv.dilate(edge_targ, (11,11), iterations = 2)

circles = circles(edge_targ,radius)
print(len(circles)," coins detected.")

cv.imwrite(output+"CannyEdge.jpg", edge_targ)

i=0
count=0
for circle in circles:
	log.write("{}. circle\n".format(i+1))
	log.write("x: {}, y: {}, r: {}\n\n".format(circle[0],circle[1],circle[2]))
	mask = np.zeros(edge_targ.shape[:2], dtype="uint8")
	cv.circle(mask, (circle[0], circle[1]), circle[2], 255, -1)
	cut = cv.bitwise_and(edge_targ, edge_targ, mask=mask)

	cv.imwrite(cutpath+str(i+1)+". cut.jpg", cut)
	matching.append([circle,[]])

	j=0
	for forint in images:
		value=0
		k=0
		for coin in forint:
			sample = cv.imread(coin)
			res_sample = cv.resize(sample, (256,int((sample.shape[1]/sample.shape[0])*256)))
			blur_samp = cv.GaussianBlur(res_sample, (3,3), 2)
			edge_samp = cv.Canny(blur_samp, 30, 150, 2)
			edge_samp = cv.dilate(edge_samp, (1,1), iterations = 2)
			if(i==0):cv.imwrite(samplepath+forints[j]+"."+str(k+1)+". sample.jpg", edge_samp)

			k1,k2,d1,d2=sift(cut,edge_samp)
			matches=flann(d1,d2)

			value += len(matches)

			log.write("Match with {}: {}\n".format(coin,len(matches)))

			result=showMatches(target, res_sample,matches,k1,k2,d1,d2)
			result = cv.resize(result, (1600,900))
			cv.imwrite(matchpath+str(i+1)+'.'+forints[j]+"."+str(k+1)+". keypoints.jpg", result)
			k+=1

		value = value / k
		log.write("Average of {}: {}\n\n".format(forint,value))
		matching[i][1].append(value)
		j+=1

	largest_value=max(matching[i][1])
	indexof_largest =forints[matching[i][1].index(largest_value)]
	count+=int(indexof_largest)

	log.write("The value of the coin is: {}\n\n".format(indexof_largest))

	cv.putText(target,indexof_largest,(circle[0],circle[1]),cv.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0, 0),2)
	cv.circle(target,(circle[0],circle[1]),circle[2],(0,255,0),2)
	print(i+1,". circle done.")
	i+=1

log.write("\nThe overall value of the coins is: {}\n\n".format(count))

cv.putText(target,"The value of the coins is: {}".format(count),(10,30),cv.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0, 0),2)
cv.imwrite(output+"endresult.jpg", target)

log.close()

cv.imshow("Coin counting done",target)
cv.waitKey()
