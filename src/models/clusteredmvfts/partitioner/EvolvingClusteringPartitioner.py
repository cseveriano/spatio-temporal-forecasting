from pyFTS.common import FuzzySet
from pyFTS.partitioners import partitioner
from evolving import EvolvingClustering

class EvolvingClusteringPartitioner(partitioner.Partitioner):

    def __init__(self, **kwargs):
        super(EvolvingClusteringPartitioner, self).__init__(name="EvolvingClustering", preprocess=False, **kwargs)
        self.variance_limit = kwargs.get('variance_limit',0.001)
        self.debug = kwargs.get('debug', False)



        data = kwargs.get('data', [None])
        self.sets = self.build(data)

        self.ordered_sets = list(self.sets.keys())


    def build(self, data):
        sets = {}
        clusterer = EvolvingClustering.EvolvingClustering(variance_limit=self.variance_limit, debug=self.debug)

        clusterer.fit(data.values)

        macro_clusters = clusterer.macro_clusters


        label_ind = 0

        for mc in macro_clusters:
            _name = "C"+str(label_ind)

            active_micro_clusters = clusterer.get_active_micro_clusters(mc)
            fs = FuzzySet.FuzzySet(_name,EvolvingClustering.calculate_membership, active_micro_clusters,mc)

            sets[_name] = fs
            label_ind += 1

        return sets
