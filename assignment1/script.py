#!/usr/local/bin/python

import sys, math, numpy, random
from PIL import Image

image_dimensions = (0,0)

# k-means parameter
max_centeroid_distance = 40
max_iterations = 10
k_partitions = 8

# dbscan parameters
eps             = 10
minpoints       = 10
NOISE           = 0
UNCLASSIFIED    = -1

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
            color = colors(cluster_id)
            set_rgb_color(image, x, y, color)
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
    total = math.pow(r_a-r_b,2)
    total += math.pow(g_a-g_b,2)
    total += math.pow(b_a-b_b,2)
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

########################################################################################################################


def k_means(image, k):
    clustering = numpy.empty(image_dimensions, dtype=int)
    x_w, y_w = image_dimensions

    # chose k points as center
    new_centers = list()
    old_centers = list()

    for cluster_id in range(0, k):
        # get random point? may result in equal color points
        random_color = (
                random.randint(0, 255),     # red value
                random.randint(0, 255),     # green value
                random.randint(0, 255)      # blue value
        )
        new_centers.append((random_color, cluster_id))

    changed = True
    iterations = 0
    while changed and iterations < max_iterations:
        iterations += 1
        print("step 1: assigning pixels to clusters")
        changed = False
        # iterate over all points
        for x in range(0, x_w ):
            for y in range(0, y_w ):
                start_center = new_centers[0]
                d_min = euklidian_dist(get_rgb_color(image, x, y), start_center)

                current_cluster = clustering[x, y]
                # compute distance to all centers
                for center in new_centers:
                    d, clid = euklidian_dist(get_rgb_color(image, x, y), center)
                    if d < d_min:
                        d_min = d
                        # set cluster_id as closest cluster
                        if current_cluster != clid:
                            clustering[x, y] = clid
                            changed = True
         # calculate new centeroids
        print("step 2: calculating new centeroids")
        summator = dict()
        for cluster_id in range(0,k):
            sum = dict()
            sum["r"] = 0
            sum["g"] = 0
            sum["b"] = 0
            sum["total"] = 0
            summator[str(cluster_id)] = sum

        for x in range(0, x_w):
            for y in range(0, y_w):
                cluster_id = clustering[x, y]
                r, g, b = get_rgb_color(image, x, y)
                summator[str(cluster_id)]["r"] += r
                summator[str(cluster_id)]["g"] += g
                summator[str(cluster_id)]["b"] += b
                summator[str(cluster_id)]["total"] += 1

        del old_centers[:]
        old_centers += new_centers
        del new_centers[:]
        for cluster_id in range(0,k):
            if summator[str(cluster_id)]["total"] > 0:
                new_center = (
                    summator[str(cluster_id)]["r"]/summator[str(cluster_id)]["total"],
                    summator[str(cluster_id)]["g"]/summator[str(cluster_id)]["total"],
                    summator[str(cluster_id)]["b"]/summator[str(cluster_id)]["total"]
                    )
                new_centers.append((new_center, cluster_id))

        # centroid distance
        # stop criteria: |(old - new)| < max_centeroid_distance
        threshold_exceeded = False
        for old_center, old_cluster_id in old_centers:
            for new_center, new_cluster_id in new_centers:
                if old_cluster_id == new_cluster_id:
                    center_distance, cluster_id = euklidian_dist(old_center, (new_center, new_cluster_id))
                    if center_distance > max_centeroid_distance:
                        threshold_exceeded = True
        if not threshold_exceeded:
            changed = False
        print("Iteration step: " + str(iterations))

    colors = get_spaced_colors(k)
    for x in range(0, x_w):
        for y in range(0, y_w):
            set_rgb_color(image, x, y, colors[clustering[x, y]])

    return image

########################################################################################################################


class Pixel:
    def __init__(self, color, xy, cluster_id = UNCLASSIFIED):
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
        while len(seeds)>0:
            o = seeds.pop(0)        # take first pixel in list
            print(o)
            n = neighbourhood(pixel_set, o, epsilon)
            if len(n) > min_pts:    # o is core object
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
            pixel = Pixel(get_rgb_color(image, x, y),(x, y))
            pixels.append(pixel)

    cluster_id = next_id(NOISE)
    for pixel in pixels:
        print(pixel)
        if pixel.cluster_id == UNCLASSIFIED:
            if expand_cluster(pixels, pixel, cluster_id, epsilon, min_pts):
                cluster_id = next_id(cluster_id)

    colors = get_spaced_colors(k)
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

    #output_image = k_means(input_image, k_partitions)
    output_image = dbscan(input_image, 10, 10)
    save_image(output_image, outpath)


