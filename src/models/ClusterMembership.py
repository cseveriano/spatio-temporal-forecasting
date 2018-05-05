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
