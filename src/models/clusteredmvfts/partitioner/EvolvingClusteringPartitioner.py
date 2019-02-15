from pyFTS.partitioners import partitioner
from evolving import EvolvingClustering
from clusteredmvfts.partitioner import FuzzySet
class EvolvingClusteringPartitioner(partitioner.Partitioner):

    def __init__(self, **kwargs):
        super(EvolvingClusteringPartitioner, self).__init__(name="EvolvingClustering", preprocess=False, **kwargs)
        self.variance_limit = kwargs.get('variance_limit',0.001)
        self.debug = kwargs.get('debug', False)



        data = kwargs.get('data', [None])
        self.sets = self.build(data)

        self.ordered_sets = list(self.sets.keys())


    # def build(self, data):
    #     sets = {}
    #     clusterer = EvolvingClustering.EvolvingClustering(variance_limit=self.variance_limit, debug=self.debug, plot_graph=True)
    #
    #     clusterer.fit(data.values)
    #
    #     macro_clusters = clusterer.active_macro_clusters
    #
    #
    #     label_ind = 0
    #
    #     for mc in macro_clusters:
    #         _name = "C"+str(label_ind)
    #
    #         active_micro_clusters = clusterer.get_active_micro_clusters(mc)
    #
    #         centroid = EvolvingClusteringPartitioner.get_macro_cluster_centroid(active_micro_clusters)
    #         fs = FuzzySet.FuzzySet(_name,EvolvingClustering.EvolvingClustering.calculate_membership, active_micro_clusters,centroid)
    #
    #         sets[_name] = fs
    #         label_ind += 1
    #
    #     return sets

    # Micro clusters as fuzzy sets version
    def build(self, data):
        sets = {}
        clusterer = EvolvingClustering.EvolvingClustering(variance_limit=self.variance_limit, debug=self.debug, plot_graph=False)

        clusterer.fit(data.values)

        micro_clusters = clusterer.get_all_active_micro_clusters()


        label_ind = 0

        for m in micro_clusters:
            _name = "C"+str(label_ind)
            centroid =  m["mean"]

            fs = FuzzySet.FuzzySet(_name, EvolvingClustering.EvolvingClustering.calculate_micro_membership, [m, clusterer.get_total_density()], centroid)

            sets[_name] = fs
            label_ind += 1

        return sets

    @staticmethod
    def get_macro_cluster_centroid(micro_clusters):

        centroid = [0] * len(micro_clusters[0]["mean"])

        for m in micro_clusters:
            centroid += m["mean"]

        return centroid / len(micro_clusters)
