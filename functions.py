#!/usr/bin/env python3.10

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import filters as flt

def showImg(picArr):
	plt.imshow(picArr)
	plt.show()

def showHist(picArr):
	hist=histogram(picArr)
	plt.plot(hist)
	plt.show()

def histogram(arr):
	hist=np.zeros((256),dtype=int)

	for i in range(0,len(arr)):
		for j in range(0,len(arr[i])):
			for k in range(0,3):
				n=round(arr[i][j][0]*255)
				hist[n]+=1

	print("Successfully generated a histogram.")
	return hist

def comparison(hist1, hist2):
	s=0

	for i in range(0,256):
		s=s+(pow(hist1[i]-hist2[i],2)/hist1[i])

	print("Successfully counted the difference.")
	return s

def grayScale(arr):
	gray=np.zeros((len(arr),len(arr[0]),4),dtype=float)
	for i in range(0,len(arr)):
		for j in range(0,len(arr[i])):
			for k in range(0,3):
				gray[i][j][k]=(arr[i][j][0]+arr[i][j][1]+arr[i][j][2])/3

			if (len(arr[i][j])==4): 
				gray[i][j][3]=arr[i][j][3]
			else: gray[i][j][3]=1

	print("Successfully generated grayscale image.")
	return gray

def colorFilter(arr,flt):
	filtered=np.zeros((len(arr),len(arr[0]),4),dtype=float)
	for i in range(0,len(arr)):
		for j in range(0,len(arr[i])):
			for k in range(0,3):
				filtered[i][j][k]=flt[k]*arr[i][j][k]

			if (len(arr[i][j])==4): 
				filtered[i][j][3]=arr[i][j][3]
			else: filtered[i][j][3]=1

	print("Successfully generated a color-filtered image.")
	return filtered

def contrast(arr):
	contrasted=np.zeros((len(arr),len(arr[0]),4),dtype=float)

	for i in range(0,len(arr)):
		for j in range(0,len(arr[i])):
			for k in range(0,3):
				if(arr[i][j][k]>0.5):
					contrasted[i][j][k]=1
				if(arr[i][j][k]<0.5):
					contrasted[i][j][k]=0

			if (len(arr[i][j])==4): 
				contrasted[i][j][3]=arr[i][j][3]
			else: contrasted[i][j][3]=1

	print("Successfully generated a contrasted image.")
	return contrasted

def convolution(arr, matrix,multiplier=1):
	conved=np.zeros((len(arr),len(arr[0]),4),dtype=float)
	m_mid_i=int(len(matrix)/2)
	m_mid_j=int(len(matrix[0])/2)

	for i in range(0,len(arr)):
		for j in range(0, len(arr[i])):
			for k in range(0,3):

				if ( (i>=m_mid_i and j>=m_mid_j) and (i<=len(arr)-(m_mid_i+1) and j<=len(arr[i])-(m_mid_j+1)) ):
					for mi in range(0,len(matrix)):
						for mj in range(0,len(matrix[mi])):
								conved[i][j][k]=conved[i][j][k] + (arr[i+(mi - m_mid_i)][j+(mj - m_mid_j)][k] * matrix[mi][mj])
					conved[i][j][k]=conved[i][j][k]*multiplier
				else:
					conved[i][j][k]=arr[i][j][k]

				if(conved[i][j][k]<0): conved[i][j][k]=0
				if(conved[i][j][k]>1): conved[i][j][k]=1
			
			if (len(arr[i][j])==4): 
					conved[i][j][3]=arr[i][j][3]
			else: conved[i][j][3]=1

	print("Successfully generated a convoluted image.")
	return conved

