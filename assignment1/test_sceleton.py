#!/usr/local/bin/python

import sys, math, numpy, random, os
from PIL import Image

sys.path.append(os.getcwd()+"/module")
from algorithms import k_means, dbscan
from imagefunctions import *

image_dimensions = (0, 0)

# k-means parameter
MAX_CENTEROIDS_DISTANCE = 20
MAX_ITERATIONS = 10
K_PARTITIONS = 8

# dbscan parameters
eps = 10
minpoints = 10
NOISE = 0
UNCLASSIFIED = -1

# general paramaters
outpath = "output/output.bmp"
outformat = "BMP"


########################################################################################################################

# main program
if __name__ == "__main__":
    input_image = load_image(sys.argv[1])
    data = flatten_image(input_image)

    #clustering = k_means(data, K_PARTITIONS, MAX_CENTEROIDS_DISTANCE, MAX_ITERATIONS)
    clustering = dbscan(data, eps, minpoints)
    output_image = unflatten_image(input_image, clustering, K_PARTITIONS)
    save_image(output_image, outpath, outformat)
    print ("image written to " + outpath)

