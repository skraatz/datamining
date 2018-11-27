#!/usr/local/bin/python

import sys, math, numpy, random
from PIL import Image

image_dimensions = (0, 0)

# k-means parameter
max_centeroid_distance = 20
max_iterations = 10
k_partitions = 8

# dbscan parameters
eps = 10
minpoints = 10
NOISE = 0
UNCLASSIFIED = -1

# general paramaters
outpath = "output.bmp"
outformat = "BMP"


########################################################################################################################
def load_image(path):
    image = Image.open(path)
    global image_dimensions
    image_dimensions = image.width, image.height
    return image


def flatten_image(image):
    """
    turn image into list of [r,g,b]
    :param image:
    :return:
    """
    x_w, y_w = image.width, image.height
    data = list()
    for x in range(0, x_w):
        for y in range(0, y_w):
            vector_data = list(get_rgb_color(image, x, y))
            data.append(vector_data)
    return data


def unflatten_image(image, clustering, k):
    """
    recolor image based on cluster id
    :param image: Pillow image, that was worked on
    :param clustering:
    :param k: max cluster_id -1
    :return: the colored version of the image
    """
    x_w, y_w = image.width, image.height
    colors = get_spaced_colors(k)
    list_pos = 0
    for x in range(0, x_w):
        for y in range(0, y_w):
            cluster_id = clustering[list_pos]
            set_rgb_color(image, x, y, colors[cluster_id])
            list_pos += 1
    return image


def save_image(image, path):
    image.save(path, outformat)


def get_rgb_color(image, x, y):
    return image.getpixel((x, y))


def set_rgb_color(image, x, y, (r, g, b)):
    image.putpixel((x, y), (r, g, b))


def euklidian_dist(point_a, point_b):
    r_a, g_a, b_a = point_a
    (r_b, g_b, b_b), clid = point_b
    total = math.pow(r_a - r_b, 2)
    total += math.pow(g_a - g_b, 2)
    total += math.pow(b_a - b_b, 2)
    return math.sqrt(total), clid


def euklidian_dist_generic(point_a, point_b):
    """
    takes two equal length lists (data points)
    :param point_a:
    :param point_b:
    :return: metric distance between data points
    """
    assert (len(point_a) == len(point_b))
    sum = 0
    for flag in range(len(point_a)):
        sum += math.pow(point_a[flag] - point_b[flag], 2)
    return math.sqrt(sum)


def get_spaced_colors(n):
    max_value = 16581375  # 255**3
    interval = int(max_value / n)
    colors = [hex(I)[2:].zfill(6) for I in range(0, max_value, interval)]
    return [(int(i[:2], 16), int(i[2:4], 16), int(i[4:], 16)) for i in colors]


def random_centeroid(normalizer, cluster_id):
    random_center = tuple(
        random.randint(0, value) for value in normalizer
    )
    return (random_center, cluster_id)


########################################################################################################################


def k_means_generic(data, k):
    """
    returns clustering in the same order
    as the provided data vectors
    :param data: list of data vector
    :param k: k-means parameter
    :return: list of cluster ids
    """
    clustering = list()  # list of cluster_ids

    normalizer = list()  # store max values for initial center calculation
    for init in range(len(data[0])):
        normalizer.append(0)

    print("calculating max values for initial center vectors")
    for datapoint in data:
        clustering.append(0)  # assign each data point to cluster 0 initially
        for value in range(len(data[0])):
            normalizer[value] = max(normalizer[value], datapoint[value])

    print("choosing initial centers")
    # chose k points as center
    new_centers = list()
    old_centers = list()

    for cluster_id in range(0, k):
        new_centers.append(random_centeroid(normalizer, cluster_id))
    print ("initial centers:")
    for center in new_centers:
        print(center)

    print("starting k-means algorithm")

    running = True
    iterations = 0
    while running:
        running = False
        print("iteration step :" + str(iterations))

        print("assigning closest centeroid")
        # iterate over data points
        for point_index in range(len(data)):
            datapoint = data[point_index]
            current_cluster_id = clustering[point_index]
            current_center, cluster_id = new_centers[current_cluster_id]
            dist_min = euklidian_dist_generic(datapoint, current_center)
            for center, cluster_id in new_centers:
                new_dist = euklidian_dist_generic(datapoint, center)
                if new_dist < dist_min:  # data point is closer to other center
                    clustering[point_index] = cluster_id  # assign new cluster_id
                    dist_min = new_dist  # new minimal distance

        print("recalculating centeroids")
        # calculate new centeroids

        del old_centers[:]
        old_centers += new_centers
        del new_centers[:]

        # finding average for each vector value
        sums = list()
        for i in range(k):
            sums.append(
                (list(
                    0 for value in range(0, len(data[0]))
                ), 0)
            )

        for point_index in range(len(data)):
            datapoint = data[point_index]
            cluster_id = clustering[point_index]
            # vector to sum for center calculation
            sum, counter = sums[cluster_id]
            for i, value in enumerate(datapoint):
                sum[i] += value
            counter += 1
            sums[cluster_id] = sum, counter

        for cluster_id in range(k):
            sum, counter = sums[cluster_id]
            # print (sum, counter)
            if counter == 0:
                new_centers.append(random_centeroid(normalizer, cluster_id))
            else:
                center = tuple(
                    value / counter for value in sum
                )
                # print (center)
            new_centers.append((center, cluster_id))
        print(new_centers)

        print("calculating maximum center distance")
        dist = 0
        for i in range(k):
            oc, ocid = old_centers[i]
            nc, ncid = new_centers[i]
            # print(oc)
            # print(nc)
            old_dist = euklidian_dist_generic(oc, nc)
            dist = max(dist, old_dist)
            print(i, ":", old_dist)

        # check for exit conditions
        if dist > max_centeroid_distance:
            running = True

        iterations += 1
        if iterations >= max_iterations:
            running = False

    return clustering


########################################################################################################################


class Pixel:
    def __init__(self, color, xy, cluster_id=UNCLASSIFIED):
        self.color = color
        x, y = xy
        self.x = x
        self.y = y
        self.cluster_id = cluster_id
        self.r, self.g, self.b = color

    def xy(self):
        return self.x, self.y

    def __dict__(self):
        return self.color, (self.x, self.y), self.cluster_id

    def __str__(self):
        s = str(self.__dict__)
        return s


def default_pixel():
    return Pixel((1, 1, 1), (2, 2), -1)


def next_id(current):
    return current + 1


def neighbourhood(pixel_set, pixel, epsilon):
    neighbours = list()
    for pot_neighbour in pixel_set:
        dist, cl_id = euklidian_dist(pixel.color, (pot_neighbour.color, pot_neighbour.cluster_id))
        if dist <= epsilon:
            neighbours.append(pot_neighbour)
    return neighbours


class Vector:
    def __init__(self):
        self.data = tuple
        self.id = tuple

    def assign(self, data, id):
        self.data = data
        self.id = id


def assign_cluster(pixel, cluster_id):
    pixel.cluster_id = cluster_id


def expand_cluster(pixel_set, start_pixel, cluster_id, epsilon, min_pts):
    print("preparing neighbours")
    seeds = neighbourhood(pixel_set, start_pixel, epsilon)
    if len(seeds) < min_pts:
        print("pixel is noise")
        assign_cluster(pixel_set, start_pixel, NOISE)
        return False
    for pixel in seeds:
        print("assigning")
        assign_cluster(pixel, cluster_id)
        # remove start_pixel from seeds
        while len(seeds) > 0:
            o = seeds.pop(0)  # take first pixel in list
            print(o)
            n = neighbourhood(pixel_set, o, epsilon)
            if len(n) > min_pts:  # o is core object
                for p in n:
                    if p.cluster_id == UNCLASSIFIED or p.cluster_id == NOISE:
                        assign_cluster(p, cluster_id)
                        if p.cluster_id == UNCLASSIFIED:
                            seeds.append(p)
            # remove done by pop operation
    return True


def dbscan(image, epsilon, min_pts):
    x_w, y_w = image_dimensions
    pixels = list()
    # flatten image
    for x in range(0, x_w):
        for y in range(0, y_w):
            pixel = Pixel(get_rgb_color(image, x, y), (x, y))
            pixels.append(pixel)

    cluster_id = next_id(NOISE)
    for pixel in pixels:
        print(pixel)
        if pixel.cluster_id == UNCLASSIFIED:
            if expand_cluster(pixels, pixel, cluster_id, epsilon, min_pts):
                cluster_id = next_id(cluster_id)

    colors = get_spaced_colors(cluster_id)
    for pixel in pixels:
        set_rgb_color(image, pixel.x, pixel.y, colors[pixel.cluster_id])

    return image


########################################################################################################################


# main program
if __name__ == "__main__":
    # print ('Number of arguments:', len(sys.argv), 'arguments.')
    # print ('Argument List:', str(sys.argv))
    input_image = load_image(sys.argv[1])
    print("image dimensions = " + str(image_dimensions))
    data = flatten_image(input_image)
    clustering = k_means_generic(data, k_partitions)
    output_image = unflatten_image(input_image, clustering, k_partitions)
    save_image(output_image, outpath)
    # output_image = k_means(input_image, k_partitions)
    # output_image = dbscan(input_image, 10, 10)
    # save_image(output_image, outpath)
