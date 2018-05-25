"""
"""

import numpy as np
from pyFTS.common import FuzzySet, FLR, fts, flrg, tree

class SpatioTemporalHighOrderFLRG(flrg.FLRG):
    """Conventional High Order Fuzzy Logical Relationship Group"""
    def __init__(self, order, **kwargs):
        super(SpatioTemporalHighOrderFLRG, self).__init__(order, **kwargs)
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


class SpatioTemporalHighOrderFTS(fts.FTS):
    """Conventional High Order Fuzzy Time Series"""
    def __init__(self, name, **kwargs):

        super(SpatioTemporalHighOrderFTS, self).__init__(kwargs.get('nlags',1), name="STHOFTS" + name, **kwargs)
        self.name = "Spatio Temporal High Order FTS"
        self.shortname = "STHOFTS" + name
        self.detail = "Severiano"
        self.setsDict = {}
        self.is_high_order = True
        self.membership_threshold = kwargs.get('membership_threshold',0.6)

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
            flrg = SpatioTemporalHighOrderFLRG(self.order)
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
                    self.flrgs[flrg.get_key()] = flrg;

                for st in rhs:
                    self.flrgs[flrg.get_key()].append_rhs(st)


    def fuzzyfication(self, x):
        memberships = np.zeros(len(self.partitioner.ordered_sets))
        i = 0
        fuzzy_sequence = []
        for key in self.partitioner.ordered_sets:
            memberships[i] = self.sets[key].membership(x)
            i += 1
        # sorting memberships
        descending = np.argsort(memberships)[::-1]
        total_membership = 0

        for mb in descending:
            if total_membership <= self.membership_threshold:
                fuzzy_sequence.append(list(self.partitioner.ordered_sets)[mb])
            else:
                break

            total_membership += memberships[mb]

        return fuzzy_sequence

    def train(self, data, **kwargs):

        if kwargs.get('sets', None) is not None:
            self.sets = kwargs.get('sets', None)

        self.generate_flrg(data)

    def forecast(self, ndata, **kwargs):

        ret = []

        l = len(ndata)

        if l <= self.order:
            return ndata

        for k in np.arange(self.order, l+1):
            flrgs = self.generate_lhs_flrg(ndata[k - self.order: k])

            tmp = []

            for flrg in flrgs:
                if flrg.get_key() not in self.flrgs:
                    tmp.append(self.sets[flrg.LHS[-1]].centroid)
                else:
                    f = self.flrgs[flrg.get_key()]

                    # TODO: refactoring - include this code in FLRG.py (or a subclass)
                    if f.midpoint is None:
                        f.midpoint = np.nanmean(f.get_midpoints(self.sets), axis=0)

                    tmp.append(f.get_midpoint(self.sets))

            ret.append(np.nanmean(tmp, axis=0))

        return ret
