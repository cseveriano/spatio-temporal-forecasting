"""
"""

import numpy as np
from pyFTS.common import fts, flrg, tree
from ..partitioner import EvolvingClusteringPartitioner

class EvolvingClusterFLRG(flrg.FLRG):
    """Conventional High Order Fuzzy Logical Relationship Group"""
    def __init__(self, order, **kwargs):
        super(EvolvingClusterFLRG, self).__init__(order, **kwargs)
        self.LHS = []
        self.RHS = {}
        self.strlhs = ""

    def append_rhs(self, c, **kwargs):
        if c not in self.RHS:
            self.RHS[c] = c

    def append_lhs(self, c):
        self.LHS.append(c)

    def __str__(self):
        tmp = ""
        for c in sorted(self.RHS):
            if len(tmp) > 0:
                tmp = tmp + ","
            tmp = tmp + c
        return self.get_key() + " -> " + tmp


    def __len__(self):
        return len(self.RHS)

    def get_midpoint(self, sets):
        """
        Returns the midpoint value for the RHS fuzzy sets

        :param sets: fuzzy sets
        :return: the midpoint value
        """
        if self.midpoint is None:
            self.midpoint = np.nanmean(self.get_midpoints(sets), axis=0)
        return self.midpoint


class EvolvingClusterFTS(fts.FTS):
    """Conventional High Order Fuzzy Time Series"""
    def __init__(self, **kwargs):

        super(EvolvingClusterFTS, self).__init__( **kwargs)
        self.name = "Evolving Cluster FTS"
        self.shortname = "EvolvingClusterFTS"
        self.detail = "Severiano"
        self.setsDict = {}
        self.is_high_order = True
        self.membership_threshold = kwargs.get('membership_threshold',0.6)
        self.t_norm = kwargs.get('t_norm','threshold')
        self.defuzzy = kwargs.get('defuzzy','mean')
        variance_limit =  kwargs.get('variance_limit',0.001)
        debug =  kwargs.get('debug', False)
        self.partitioner = EvolvingClusteringPartitioner.EvolvingClusteringPartitioner(variance_limit = variance_limit, debug =  debug)


    def generate_lhs_flrg(self, sample):
        lags = {}

        flrgs = []

        for o in np.arange(0, self.order):
            lhs = self.fuzzyfication(sample[o])
            lags[o] = lhs

        root = tree.FLRGTreeNode(None)

        tree.build_tree_without_order(root, lags, 0)

        # Trace the possible paths
        for p in root.paths():
            flrg = EvolvingClusterFLRG(self.order)
            path = list(reversed(list(filter(None.__ne__, p))))

            for lhs in path:
                flrg.append_lhs(lhs)

            flrgs.append(flrg)

        return flrgs

    def generate_flrg(self, data):
        l = len(data)
        for k in np.arange(self.order, l):
            if self.dump: print("FLR: " + str(k))

            sample = data[k - self.order: k]

            rhs_sample = data[k]
            rhs = self.fuzzyfication(rhs_sample)

            flrgs = self.generate_lhs_flrg(sample)

            for flrg in flrgs:
                if flrg.get_key() not in self.flrgs:
                    self.flrgs[flrg.get_key()] = flrg

                for st in rhs:
                    self.flrgs[flrg.get_key()].append_rhs(st)


    def fuzzyfication(self, x):
        memberships = np.zeros(len(self.partitioner.ordered_sets))
        i = 0
        fuzzy_sequence = []
        for key in self.partitioner.ordered_sets:
            memberships[i] = self.sets[key].membership(x)
            i += 1

        if self.t_norm == 'threshold':
            # sorting memberships
            descending = np.argsort(memberships)[::-1]
            total_membership = 0

            for mb in descending:
                if total_membership <= self.membership_threshold:
                    fuzzy_sequence.append(list(self.partitioner.ordered_sets)[mb])
                else:
                    break

                total_membership += memberships[mb]
        elif self.t_norm == 'nonzero':
            # sorting memberships
            descending = np.argsort(memberships)[::-1]
            total_membership = 0

            for mb in descending:
                if memberships[mb] > 0:
                    fuzzy_sequence.append(list(self.partitioner.ordered_sets)[mb])
                else:
                    break

            if not fuzzy_sequence:
                fuzzy_sequence.append(list(self.partitioner.ordered_sets)[descending[0]])

        return fuzzy_sequence

    def train(self, data, **kwargs):

        if self.partitioner is not None:
            self.partitioner.build(data)
            self.sets = self.partitioner.sets
        elif kwargs.get('sets', None) is not None:
            self.sets = kwargs.get('sets', None)

        self.generate_flrg(data)

    def forecast(self, ndata, **kwargs):
        ret = []

        l = len(ndata)

        if l <= self.order:
            return ndata

        for k in np.arange(self.order, l+1):
            sample = ndata[k - self.order: k]
            flrgs = self.generate_lhs_flrg(sample)

            memberships = []
            midpoints = []

            for flrg in flrgs:
                if flrg.get_key() not in self.flrgs:
                    midpoints.append(self.sets[flrg.LHS[-1]].centroid)
                else:
                    f = self.flrgs[flrg.get_key()]

                    if f.midpoint is None:
                        f.midpoint = np.nanmean(f.get_midpoints(self.sets), axis=0)

                    midpoints.append(f.get_midpoint(self.sets))

            if self.defuzzy == 'weighted':
                mvs = []
                for i in np.arange(self.order):
                    mvs.append(self.sets[flrg.LHS[i]].membership(sample[i]))
                memberships.append(np.prod(mvs))
                mv_midps = [x * y for x, y in zip(midpoints, memberships)]
                ret.append(np.sum(mv_midps, axis=0)/np.sum(memberships))
            elif self.defuzzy == 'mean':
                ret.append(np.nanmean(midpoints, axis=0))

        return ret
