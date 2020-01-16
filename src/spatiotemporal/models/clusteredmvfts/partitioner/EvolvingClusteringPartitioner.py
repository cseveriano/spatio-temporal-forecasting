from pyFTS.partitioners import partitioner
from evolving.EvolvingClustering import EvolvingClustering
from . import FuzzySet

class EvolvingClusteringPartitioner(partitioner.Partitioner):

    def __init__(self, **kwargs):
        super(EvolvingClusteringPartitioner, self).__init__(name="EvolvingClustering", preprocess=False, **kwargs)
        self.variance_limit = kwargs.get('variance_limit',0.001)
        self.debug = kwargs.get('debug', False)
        self.clusterer = EvolvingClustering(variance_limit=self.variance_limit, debug=self.debug,
                                                          plot_graph=False)
        self.counter = 0

        data = kwargs.get('data', None)
        if data is not None:
            self.build(data)
        else:
            self.sets = {}
            self.ordered_sets = {}

    # Micro clusters as fuzzy sets version
    def build(self, data):
        update_mc = True

        self.clusterer.fit(data, update_macro_clusters=update_mc, prune_micro_clusters=False)

        if not self.sets:
            micro_clusters = self.clusterer.get_all_active_micro_clusters()
        else:
            micro_clusters = self.clusterer.get_changed_active_micro_clusters()

        for m in micro_clusters:
            _name = "C" + '{:03}'.format(m['id'])
            centroid = m["mean"]

            fs = FuzzySet.FuzzySet(_name, EvolvingClusteringPartitioner.calculate_fuzzyset_membership,
                                       [m], centroid)

            self.sets[_name] = fs
            self.ordered_sets = list(self.sets.keys())
        return self.sets

    @staticmethod
    def calculate_fuzzyset_membership(x, params):
        micro_cluster = params[0]

        (num_samples, mean, variance, norm_ecc) = EvolvingClustering.get_updated_micro_cluster_values(x, micro_cluster["num_samples"],
                                                                                                         micro_cluster["mean"],
                                                                                                         micro_cluster["variance"])
        norm_tip = 1 - norm_ecc

        return norm_tip

    @staticmethod
    def get_macro_cluster_centroid(micro_clusters):

        centroid = [0] * len(micro_clusters[0]["mean"])

        for m in micro_clusters:
            centroid += m["mean"]

        return centroid / len(micro_clusters)