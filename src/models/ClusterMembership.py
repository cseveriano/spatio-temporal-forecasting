import math

def trunc_gaussmf(x, parameters):
    """
    Gaussian Truncated fuzzy membership function
    :param x: data point
    :param parameters: a list with 2 real values (mean and variance) and 2 limits
    :return: the membership value of x given the parameters
    """

    mean = parameters[0]
    var = parameters[1]
    lower_bound = parameters[2]
    upper_bound = parameters[3]
    coordinate_index = parameters[4]

    value = x[coordinate_index]
    if (value >= lower_bound) & (value <= upper_bound) :
        return math.exp((-(value - mean)**2)/(2 * var**2))
    else:
        return 0


def weighted_distance(x, parameters):
    centroids = parameters[0]
    cent_ind = parameters[1]
    m = 2

    n_clusters = len(centroids)

    dists = []

    # Distance between instances and centroids
    for centroid in centroids:
        dists.append(distance(x, centroid))

    # Update membership function
    return calculate_membership(dists[cent_ind], dists, m)


def distance(x, y):
    dist = [(a - b) ** 2 for a, b in zip(x, y)]
    return math.sqrt(sum(dist))

def calculate_membership(dist_inst_c, dists, m):

    # If distance is zero, instance is the centroid
    if dist_inst_c == 0:
        result = 1
    else:
        # instance is the centroid of another group
        if any(x == 0 for x in dists) :
            result = 0
        else:
            exp = 2/(m-1)
            result = 1 / sum([(dist_inst_c / x)**exp for x in dists])

    return result
