#!/usr/local/bin/python

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
def get_dimensions(image):
    return (image.width, image.height)


# load image
def load_image(path):
    im = Image.open(path)
    global image_dimensions
    image_dimensions = get_dimensions(im)
    global cl
    cl = numpy.empty(image_dimensions, dtype=int)
    return im


# accessor function gets RGB tupel
def get_tupel(image, x, y):
    return image.getpixel((x,y))


# distance function, takes two rgb tupels
def euklidian_dist(pointA, pointB):
    r_a, g_a, b_a = pointA
    (r_b, g_b, b_b), clid = pointB
    
    sum = math.pow(r_a-r_b,2)
    sum += math.pow(g_a-g_b,2)
    sum += math.pow(b_a-b_b,2)
    return math.sqrt(sum), clid


def set_cluster_id(x, y, id):
    cl[x,y] = id


def get_cluster_id(x,y):
    return cl[x,y]


def k_means(image, k):
    # chose k points as center
    x_w, y_w = image_dimensions
    # centers should be tupels of (RGB tupel, cluster_id)
    centers = list()
    for clid in range(0, k):
        # get random point? may result in equal color points
        # x_r = random.randint(0,x_w)
        # y_r = random.randint(0,y_w)
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        # p = get_tupel(image, x_r, y_r)
        centers.append(((r, g, b), clid))
    changed = True
    while changed:
        print("assigning pixels to clusters")
        print(centers)
        changed = False
        changes = 0
        # iterate over all points
        for x in range(0, x_w ):
            for y in range(0, y_w ):
                pixel_changed = False
                start_center = centers[0]
                d_min = euklidian_dist(get_tupel(image, x, y), start_center)

                current_cluster = get_cluster_id(x, y)
                # compute distance to all centers
                for center in centers:
                    d, clid = euklidian_dist(get_tupel(image, x, y), center)
                    if d < d_min:
                        d_min = d
                        # set cluster_id as closest cluster
                        if current_cluster != clid:
                            set_cluster_id(x, y, clid)
                            changed = True
                            pixel_changed = True
                if pixel_changed:
                    changes += 1
        # calculate new centeroids
        if changed:
            print (str(changes) + " pixels changed -->")
        print("calculating new centeroids")
        summator = dict()
        for clid in range(0,k):
            sum = dict()
            sum["r"] = 0
            sum["g"] = 0
            sum["b"] = 0
            sum["total"] = 0
            summator[str(clid)] = sum
        #print(summator)
        for x in range(0, x_w):
            for y in range(0, y_w):
                clid = get_cluster_id(x, y)
                r, g, b = get_tupel(image, x, y)
                summator[str(clid)]["r"] += r
                summator[str(clid)]["g"] += g
                summator[str(clid)]["b"] += b
                summator[str(clid)]["total"] += 1

        del centers[:]
        for clid in range(0,k):
            if summator[str(clid)]["total"] > 0:
                r = summator[str(clid)]["r"]/summator[str(clid)]["total"]
                g = summator[str(clid)]["g"]/summator[str(clid)]["total"]
                b = summator[str(clid)]["b"]/summator[str(clid)]["total"]
                print (r, g, b), clid, summator[str(clid)]["total"]
                centers.append(((r, g, b), clid))


# main program
if __name__ == "__main__":
    print ('Number of arguments:', len(sys.argv), 'arguments.')
    print ('Argument List:', str(sys.argv))
    im = load_image(sys.argv[1])
    print(im)
    print(image_dimensions)
    set_cluster_id(224, 282, 1)
    print(cl)
    k_means(im, 8)


