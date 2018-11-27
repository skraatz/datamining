import math, random


def euklidian_dist_generic(point_a, point_b):
    """
    takes two equal length lists (data points)
    :param point_a:
    :param point_b:
    :return: metric distance between data points
    """
    assert (len(point_a) == len(point_b))
    total = 0
    for flag in range(len(point_a)):
        total += math.pow(point_a[flag] - point_b[flag], 2)
    return math.sqrt(total)


def random_centeroid(normalizer, cluster_id):
    """
    returns a centeroid in the range of max values specified by normalizer
    :param normalizer: a list or tuple of metric maximum values
    :param cluster_id: the cluster id for the centeroid to be returned
    :return: the tuple of centeroid, cluster id
    """
    random_center = tuple(
        random.randint(0, value) for value in normalizer
    )
    return random_center, cluster_id


def k_means(data, k_partitions=8, max_centeroid_distance=20, max_iterations=10):
    """
    returns clustering in the same order
    as the provided data vectors
    :param data: list of data vector
    :param k_partitions: k-means parameter
    :param max_centeroid_distance: maximum distance between old and new centeroid of cluster
    :param max_iterations: maximum number of iterations
    :return: list of cluster ids
    """

    print("starting k-means algorithm")

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

    for cluster_id in range(0, k_partitions):
        new_centers.append(random_centeroid(normalizer, cluster_id))

    running = True
    iterations = 0
    while running:
        print("current centers:")
        print(new_centers)
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
        for i in range(k_partitions):
            zerolist = [0] * len(data[0])
            counter = 0
            sums.append((zerolist, counter))

        for point_index in range(len(data)):
            datapoint = data[point_index]
            cluster_id = clustering[point_index]
            # vector to sum for center calculation
            totals1, counter = sums[cluster_id]
            for i, value in enumerate(datapoint):
                totals1[i] += value
            counter += 1
            sums[cluster_id] = totals1, counter

        for cluster_id in range(k_partitions):
            totals2, counter = sums[cluster_id]
            if counter == 0:
                print("regenerating centroid")
                centeroid = random_centeroid(normalizer, cluster_id)
            else:
                centeroid = (tuple(
                    value / counter for value in totals2
                ), cluster_id)
            new_centers.append(centeroid)

        print("calculating biggest center distance, max = " + str(max_centeroid_distance))
        dist = 0
        dists = list()
        for i in range(k_partitions):
            oc, ocid = old_centers[i]
            nc, ncid = new_centers[i]
            old_dist = euklidian_dist_generic(oc, nc)
            dists.append((i, old_dist))
            dist = max(dist, old_dist)

        print(list(str(i) + ":" + str(round(old_dist,2)) for i, old_dist in dists))

        # check for exit conditions
        if dist > max_centeroid_distance:
            running = True

        iterations += 1
        if iterations >= max_iterations:
            running = False

    return clustering


########################################################################################################################

"""
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
    x_w, y_w = image.width, image.height
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

    return image"""
