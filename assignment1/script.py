#!/usr/bin/python

import sys
import math
import numpy
from PIL import Image

image_dimensions = (0,0)
clustering = list()
cl = numpy.empty(image_dimensions, dtype=int)
k_num = 0


# get image dimensions
def getDimensions(image):
    return (image.width, image.height)

# load image
def loadImage(path):
    im = Image.open(path)
    global image_dimensions
    image_dimensions = getDimensions(im)
    global cl
    cl = numpy.empty(image_dimensions, dtype=int)
    return im

# accessor function gets RGB tupel
def getTupel(image, x, y):
    return image.getpixel((x,y))

# distance function, takes two rgb tupels
def eukliddist(pointA,pointB):
    mindim=min(len(pointA),len(pointB))
    sum=0
    for i in range(mindim):
        sum+=math.pow(pointA[i]-pointB[i],2)
    return math.sqrt(sum)

def set_cluster_id(x, y, id):
    cl[x,y] = id


def k_means(image):
    # chose k points as center
    #centers=list()
    #bool changed = True
	#while changed:
    #    pass
        # for each center in centers
		    # for each point in image
                # calculate dist(point, center)
				# set cluster_id as closest cluster
				#
    pass


# main program
if __name__ == "__main__":
    print 'Number of arguments:', len(sys.argv), 'arguments.'
    print 'Argument List:', str(sys.argv)
    im = loadImage(sys.argv[1])
    print(im)
    print(image_dimensions)
    set_cluster_id(499,655,1)
    print(cl)


