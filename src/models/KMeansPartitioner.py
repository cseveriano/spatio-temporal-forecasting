import numpy as np
import math
import random as rnd
import functools, operator
from pyFTS.common import Composite, FuzzySet, Membership
from pyFTS.partitioners import partitioner
from sklearn.cluster import MiniBatchKMeans
from models import ClusterMembership
from models import ClusterFuzzySet

class KMeansPartitioner(partitioner.Partitioner):

    def __init__(self, **kwargs):
        super(KMeansPartitioner, self).__init__(name="KMeans", preprocess=False, **kwargs)
        self.batch_size = kwargs.get('batch_size',1000)
        self.init_size = kwargs.get('init_size', 1000)

        data = kwargs.get('data', [None])
        self.sets = self.build(data)

        self.ordered_sets = list(self.sets.keys())

    ## OLD COMPOSITE
    # def build(self, data):
    #     sets = {}
    #
    #     clusterer = MiniBatchKMeans(init='k-means++', n_clusters=self.partitions, batch_size=self.batch_size, init_size=self.init_size,
    #                                 n_init=1, verbose=False)
    #     data_labels = clusterer.fit_predict(data)
    #     centroids = clusterer.cluster_centers_
    #
    #
    #     label_ind = 0
    #
    #     for c in centroids:
    #         _name = "C"+str(label_ind)
    #         composite = ClusterFuzzySet.FuzzySet(_name)
    #
    #         for i in np.arange(len(c)):
    #             mean = c[i]
    #             label_values = list(zip(*data[data_labels == label_ind]))[i]
    #             var = np.var(label_values)
    #             lower = min(label_values)
    #             upper = max(label_values)
    #             composite.append(ClusterMembership.trunc_gaussmf, [mean,var,lower,upper,i])
    #
    #
    #         sets[_name] = composite
    #
    #         label_ind += 1
    #
    #     return sets


    def build(self, data):
        sets = {}

        clusterer = MiniBatchKMeans(init='k-means++', n_clusters=self.partitions, batch_size=self.batch_size, init_size=self.init_size,
                                    n_init=1, verbose=False)
        data_labels = clusterer.fit_predict(data)
        centroids = clusterer.cluster_centers_


        label_ind = 0

        for c in centroids:
            _name = "C"+str(label_ind)

            fs = FuzzySet.FuzzySet(_name,ClusterMembership.weighted_distance, [centroids,label_ind],c)
            sets[_name] = fs
            label_ind += 1

        return sets
