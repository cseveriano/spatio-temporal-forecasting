from pyFTS.partitioners import partitioner
from evolving import EvolvingClustering
from . import FuzzySet
class EvolvingClusteringPartitioner(partitioner.Partitioner):

    def __init__(self, **kwargs):
        super(EvolvingClusteringPartitioner, self).__init__(name="EvolvingClustering", preprocess=False, **kwargs)
        self.variance_limit = kwargs.get('variance_limit',0.001)
        self.debug = kwargs.get('debug', False)
        self.clusterer = EvolvingClustering.EvolvingClustering(variance_limit=self.variance_limit, debug=self.debug,
                                                          plot_graph=False)


        data = kwargs.get('data', None)
        if data is not None:
            self.build(data)
        else:
            self.sets = {}
            self.ordered_sets = {}



            # Micro clusters as fuzzy sets version
    def build(self, data):

        self.clusterer.fit(data)

        if not self.sets:
            micro_clusters = self.clusterer.get_all_active_micro_clusters()
        else:
            micro_clusters = self.clusterer.get_changed_micro_clusters()

        for m in micro_clusters:
            _name = "C" + '{:03}'.format(m['id'])
            centroid = m["mean"]

            fs = FuzzySet.FuzzySet(_name, EvolvingClustering.EvolvingClustering.calculate_micro_membership,
                                       [m, self.clusterer.get_total_density()], centroid)

            self.sets[_name] = fs
            self.ordered_sets = list(self.sets.keys())
        return self.sets

    @staticmethod
    def get_macro_cluster_centroid(micro_clusters):

        centroid = [0] * len(micro_clusters[0]["mean"])

        for m in micro_clusters:
            centroid += m["mean"]

        return centroid / len(micro_clusters)
