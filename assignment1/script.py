#!/usr/local/bin/python

import sys, math, numpy, random
from PIL import Image

image_dimensions = (0,0)

# k-means parameter
max_centeroid_distance = 40
max_iterations = 10
k_partitions = 8

# dbscan parameters
eps = 10
minpoints = 10

# general paramaters
outpath = "output.bmp"
outformat = "BMP"


def load_image(path):
    image = Image.open(path)
    global image_dimensions
    image_dimensions = image.width, image.height
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


def get_spaced_colors(n):
    max_value = 16581375  # 255**3
    interval = int(max_value / n)
    colors = [hex(I)[2:].zfill(6) for I in range(0, max_value, interval)]
    return [(int(i[:2], 16), int(i[2:4], 16), int(i[4:], 16)) for i in colors]


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
        for clid in range(0,k):
            sum = dict()
            sum["r"] = 0
            sum["g"] = 0
            sum["b"] = 0
            sum["total"] = 0
            summator[str(clid)] = sum

        for x in range(0, x_w):
            for y in range(0, y_w):
                clid = clustering[x, y]
                r, g, b = get_rgb_color(image, x, y)
                summator[str(clid)]["r"] += r
                summator[str(clid)]["g"] += g
                summator[str(clid)]["b"] += b
                summator[str(clid)]["total"] += 1

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


# main program
if __name__ == "__main__":
    # print ('Number of arguments:', len(sys.argv), 'arguments.')
    # print ('Argument List:', str(sys.argv))
    input_image = load_image(sys.argv[1])
    print(image_dimensions)

    output_image = k_means(input_image, k_partitions)
    save_image(output_image, outpath)

