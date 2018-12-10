import math, random, sys


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


def max_dist_generic(point_a, point_b):
    """
    takes two equal length lists (data points)
    :param point_a:
    :param point_b:
    :return: metric maximum distance between data points
    """
    assert (len(point_a) == len(point_b))
    dist = 0
    for flag in range(len(point_a)):
        nextdist = abs(point_a[flag] - point_b[flag])
        dist = max(dist, nextdist)
    return dist


# dist_func = max_dist_generic


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

NOISE = 0
UNCLASSIFIED = -1


def next_id(current):
    return current + 1


def neighbourhood(data, data_point_number, epsilon, distfunc = euklidian_dist_generic):
    # print("gathering neighbourhood for ", data[data_point_number])
    neighbours = list()
    for i in range(len(data)):
        if i != data_point_number:
            pot_neighbour = data[i]
            # print("potential neighbour ", pot_neighbour)
            # if euklidian_dist_generic(data[data_point_number], pot_neighbour) < epsilon:
            if distfunc(data[data_point_number], pot_neighbour) < epsilon:
                neighbours.append((data[i], i))
    return neighbours


def expand_cluster(data, clustering, point_counter, cluster_id, epsilon, min_pts, distfunc):
    # print("preparing neighbours")
    seeds = neighbourhood(data, point_counter, epsilon, distfunc)
    print("neighbours: " + str(len(seeds)))
    if len(seeds) < min_pts:
        # print("data point is noise")
        clustering[point_counter] = NOISE
        return False
    # assign all neighbours the same cluster id
    for dp, pos in seeds :
        clustering[pos] = cluster_id
    while len(seeds) > 0:
        o, position = seeds.pop(0)
        n = neighbourhood(data, position, epsilon, distfunc)
        if len(n) > min_pts:  # o is core object
            for p, pos in n:
                if clustering[pos] > NOISE:
                    continue
                clustering[pos] = cluster_id
                if clustering[pos] == UNCLASSIFIED:
                    seeds.append((p, pos))
            # remove done by pop operation
    return True


def dbscan(data, epsilon, min_pts, distfunc = euklidian_dist_generic):
    """
    db scan algorithm
    :param data: list of tupels
    :param epsilon: epsilon distance
    :param min_pts: minimal count of points in epsilon distance
    :return:
    """
    clustering = [UNCLASSIFIED] * len(data)
    cluster_id = next_id(NOISE)

    total = len(data)
    for point_counter in range(len(data)):
        print(str(total-point_counter) + " datapoints remaining")
        if clustering[point_counter] == UNCLASSIFIED:
            if expand_cluster(data, clustering, point_counter, cluster_id, epsilon, min_pts, distfunc):
                cluster_id = next_id(cluster_id)
    return clustering


def dbscan_wikipedia(data, epsilon, min_pts):
    clustering = [UNCLASSIFIED] * len(data)
    cluster_id = next_id(NOISE)
    
    total = len(data)
    for point_counter in range(len(data)):
        if point_counter % 10 == 0:
            print(str(total-point_counter) + " datapoints remaining")
        if clustering[point_counter] == UNCLASSIFIED:               # /* Previously processed in inner loop */
            seeds = neighbourhood(data, point_counter, epsilon)     # /* Find neighbors */
            if len(seeds) < min_pts:                                # /* Density check, expand cluster is false/
                clustering[point_counter] = NOISE                   # /* Label as Noise */
            else:
                cluster_id = next_id(cluster_id)
                clustering[point_counter] = cluster_id
                while len(seeds) < 0:
                    data_point, position = seeds.pop(0)
                    if clustering[position] == NOISE:
                        clustering[position] = cluster_id
                    if clustering[position] == UNCLASSIFIED:
                        clustering[position] = cluster_id
                        neigh = neighbourhood(data, position, epsilon)
                        if len(neigh) >= min_pts:
                            seeds += neigh
    return clustering


