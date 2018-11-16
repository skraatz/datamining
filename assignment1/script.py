#!/usr/bin/python

import sys
import math
import numpy
import random
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
    r_a, g_a, b_a = pointA
    (r_b, g_b, b_b), clid = pointB
    
    sum=math.pow(r_a-r_b,2)
    sum+=math.pow(g_a-g_b,2)
    sum+=math.pow(b_a-b_b,2)
    return (math.sqrt(sum), clid)

def set_cluster_id(x, y, id):
    cl[x,y] = id

def get_cluster_id(x,y):
    return cl[x,y]

def k_means(image, k):
    # chose k points as center
    x_w, y_w = image_dimensions
    # centers should be tupels of (RGB tupel, cluster_id)
    centers=list()
    for clid in range(0, k):
        # get random point? may result in equal color points
        x_r = random.randint(0,x_w)
        y_r = random.randint(0,y_w)
        p = getTupel(image, x_r, y_r)
        centers.append((p,k))
    changed = True
    while changed:
        changed = False
        # iterate over all points
        for x in range(0, x_w ):
            for y in range(0, y_w ):
                d_min = float("inf")
                current_cluster = get_cluster_id(x,y)
                # compute distance to all centers
                for center in centers:
                    d, clid = eukliddist(getTupel(image, x, y), center)
                    if d<d_min:
                        d_min = d
			# set cluster_id as closest cluster


# main program
if __name__ == "__main__":
    print ('Number of arguments:', len(sys.argv), 'arguments.')
    print ('Argument List:', str(sys.argv))
    im = loadImage(sys.argv[1])
    print(im)
    print(image_dimensions)
    set_cluster_id(224,282,1)
    print(cl)
    k_means(im, 4)


