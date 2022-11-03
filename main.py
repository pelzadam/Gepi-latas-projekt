#!/usr/bin/env python3.10

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import functions as f
import filters as flt

images=['images/1forint.png',
		'images/5forint.png',
		'images/20forint.png',
		'images/50forint2.png',
		'images/100forint.png'
]

target=mpimg.imread('images/50forint.png')
hist=f.histogram(f.convolution(target,flt.ridge2))

for img in images:
	image=mpimg.imread(img)
	f.showImg(image)

	imghist=f.histogram(f.convolution(image,flt.ridge2))
	print(f.comparison(hist,imghist))