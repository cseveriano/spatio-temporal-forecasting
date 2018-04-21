import pandas as pd
import numpy as np
import math
import random as rnd
import functools, operator
from pyFTS.common import FuzzySet, Membership
from pyFTS.partitioners import partitioner


# import CMeans

# S. T. Li, Y. C. Cheng, and S. Y. Lin, “A FCM-based deterministic forecasting model for fuzzy time series,”
# Comput. Math. Appl., vol. 56, no. 12, pp. 3052–3063, Dec. 2008. DOI: 10.1016/j.camwa.2008.07.033.

def fuzzy_distance(x, y):
    if isinstance(x, list):
        tmp = functools.reduce(operator.add, [(x[k] - y[k]) ** 2 for k in range(0, len(x))])
    else:
        tmp = (x - y) ** 2
    return math.sqrt(tmp)


def membership(val, vals):
    soma = 0
    for k in vals:
        if k == 0:
            k = 1
        soma = soma + (val / k) ** 2

    return soma


def fuzzy_c_means(k, d, m, deltadist=0.001):

    if isinstance(d, pd.DataFrame):
        df = d
    else:
        df = pd.DataFrame(data=d)

    # Init centroids with random samples from the dataset
    centroids = df.sample(n=k, replace=False)

    # Membership matrix
    U = [[0 for kk in range(0, k)] for xx in range(0, len(df))]

    avg_update = 1000

    iterate = 0

    while iterate < 1000 and avg_update > deltadist:

        # update membership function matrix
        update_membership(U, centroids, df, k, m)

        # update centroids
        old_centroids = centroids.copy()
        update_centroids(centroids, U, m, df)

        #check average updates
        for ind, cent in centroids.iterrows():
            old_centroid = old_centroids.loc[ind]
            avg_update += distance(old_centroid, cent)

        avg_update = avg_update / k
    #        print(avg_update)

        iterate += 1
        print(iterate)

    return [centroids, U]


def update_membership(U, centroids, df, k, m):
    i = 0
    for index, row in df.iterrows():

        dists = []

        # Distance between instances and centroids
        for ind, cent in centroids.iterrows():
            dists.append(distance(row, cent))

        # Update membership function
        for j in range(0, k):
            U[i][j] = calculate_membership(dists[j], dists, m)

        i += 1


def update_centroids(centroids, U, m, data):

    k = 0

    for indc, cent in centroids.iterrows():

        pow_pert = [x[k] ** m for x in U]

        a = np.sum(data.mul(pow_pert, axis=0).values, axis=0)
        b = np.sum(pow_pert)
        update_cent = a / b

        upd = 0
        for c in data.columns:
            centroids.set_value(index=indc, col=c, value=update_cent[upd])
            upd += 1

        k += 1

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

def distance(x,y):
    dist = [(a - b) ** 2 for a, b in zip(x, y)]
    return math.sqrt(sum(dist))

class AugmentedFCMPartitioner(partitioner.Partitioner):
    """

    """

    def __init__(self, **kwargs):
        super(AugmentedFCMPartitioner, self).__init__(name="AugmentedFCM", **kwargs)

    def build(self, data):
        sets = {}

        centroids = fuzzy_c_means(self.partitions, data, 1, 2)
        centroids.append(self.max)
        centroids.append(self.min)
        centroids = list(set(centroids))
        centroids.sort()
        for c in np.arange(1, len(centroids) - 1):
            _name = self.get_name(c)
            if self.membership_function == Membership.trimf:
                sets[_name] = FuzzySet.FuzzySet(_name, Membership.trimf,
                                                [round(centroids[c - 1], 3), round(centroids[c], 3),
                                                 round(centroids[c + 1], 3)],
                                                round(centroids[c], 3))
            elif self.membership_function == Membership.trapmf:
                q1 = (round(centroids[c], 3) - round(centroids[c - 1], 3)) / 2
                q2 = (round(centroids[c + 1], 3) - round(centroids[c], 3)) / 2
                sets[_name] = FuzzySet.FuzzySet(_name, Membership.trimf,
                                                [round(centroids[c - 1], 3), round(centroids[c], 3) - q1,
                                                 round(centroids[c], 3) + q2, round(centroids[c + 1], 3)],
                                                round(centroids[c], 3))

        return sets
