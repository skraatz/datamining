#!/usr/local/bin/python

import sys, math, numpy, random, os
from PIL import Image

sys.path.append(os.getcwd()+"/module")
from algorithms import * # k_means, dbscan, dbscan_wikipedia
from imagefunctions import *

image_dimensions = (0, 0)

# k-means parameter
MAX_CENTEROIDS_DISTANCE = 5
MAX_ITERATIONS = 20
K_PARTITIONS_MAX = 20

# dbscan parameters
EPSILON_MAX = 20
MIN_PTS_MAX = 30
NOISE = 0
UNCLASSIFIED = -1

# general paramaters
outpath = "output/output.bmp"
outformat = "BMP"


def printhelp():
    helpstring = "script usage: python test_sceleton.py imagefile <algorithm> <alg_parms>\n"
    helpstring = helpstring + "\t\t\t <algorithm>: \teither <d> for dbscan or <k> for k-means\n"
    helpstring = helpstring + "\t\t\t <alg_parms>: \t* dbscan : epsilon: int value, min_pts: int value, distance_function 'e' or 'm'\n"
    helpstring = helpstring + "\t\t\t\t\t* k-means : k: int value\n"
    helpstring = helpstring + "example 1: python test_sceleton.py inputimage.jpg k 5\n"
    helpstring = helpstring + "example 2: python test_sceleton.py inputimage.jpg d 10 20 e\n"

    print(helpstring)

########################################################################################################################


# main program
if __name__ == "__main__":
    clustering = list()
    data = list()
    K_PARTITIONS = 0
    if len(sys.argv) < 4 or sys.argv[1] in ["h", "-h", "-help"]:
        printhelp()
    else:
        input_image = load_image(sys.argv[1])
        data = flatten_image(input_image)
        if sys.argv[2] == "k":
            k = int(sys.argv[3])
            print range(K_PARTITIONS_MAX+1)
            if k in range(K_PARTITIONS_MAX+1) and k > 0:
                clustering = k_means(data, k, MAX_CENTEROIDS_DISTANCE, MAX_ITERATIONS)
                K_PARTITIONS = k
            else:
                print("ERROR: number of clusters should be within 0.." + str(K_PARTITIONS_MAX) + "\n")
                printhelp()
        else:
            if sys.argv[2] == "d":
                if len(sys.argv) != 6:
                    print("ERROR: not enough arguments\n")
                    printhelp()
                else:
                    epsilon = int(sys.argv[3])
                    min_pts = int(sys.argv[4])
                    if (epsilon not in range(EPSILON_MAX+1)) or (min_pts not in range(MIN_PTS_MAX+1)):
                        printhelp()
                    else:
                        df = sys.argv[5]
                        if df not in ['e', 'm']:
                            printhelp()
                        else:
                            if df == 'e':
                                dist_func = euklidian_dist_generic
                            clustering = dbscan(data, epsilon, min_pts)
                            # identify number of martitions
                            print ("counting found partitions")
                            for cluster_id in clustering:
                                K_PARTITIONS = max(K_PARTITIONS, cluster_id)
                            K_PARTITIONS += 1
                            print(str(K_PARTITIONS) + " partitions found")
            else:
                print("ERROR: unknown option:", sys.argv[2], "\n")
    if len(clustering) == len(data) and len(data) > 0:
        print("coloring image")
        output_image = unflatten_image(input_image, clustering, K_PARTITIONS)
        save_image(output_image, outpath, outformat)
        print ("image written to " + outpath)

