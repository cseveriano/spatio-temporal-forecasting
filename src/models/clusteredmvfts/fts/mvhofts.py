"""
Multivariate High Order FTS by Lee (2006)

Lee, L. W., Wang, L. H., Chen, S. M., & Leu, Y. H. (2006).
Handling forecasting problems based on two-factors high-order fuzzy time series.
IEEE Transactions on Fuzzy Systems, 14(3), 468–477.
"""

import numpy as np
from pyFTS.common import fts, flrg, tree



class MultivariateHighOrderFLRG(flrg.FLRG):
    """Conventional High Order Fuzzy Logical Relationship Group"""
    def __init__(self, order, **kwargs):
        super(MultivariateHighOrderFLRG, self).__init__(order, **kwargs)
        self.LHS = []
        self.RHS = {}
        self.strlhs = ""

    def appendRHS(self, c):
        if c.name not in self.RHS:
            self.RHS[c.name] = c

    def strLHS(self):
        if len(self.strlhs) == 0:
            for lnd, lag in enumerate(self.LHS):
                if lnd > 0:
                    self.strlhs += ";"
                for fnd, factor in enumerate(lag):
                    if fnd > 0:
                        self.strlhs += "|"
                    for ind, fs in enumerate(factor):
                        if ind > 0:
                            self.strlhs += ","
                        self.strlhs += fs.name
        return self.strlhs

    def appendLHS(self, c):
        self.LHS.append(c)

    # def __str__(self):
    #     tmp = ""
    #     for c in sorted(self.RHS):
    #         if len(tmp) > 0:
    #             tmp = tmp + ","
    #         tmp = tmp + c
    #     return self.strLHS() + " -> " + tmp


    def __len__(self):
        return len(self.RHS)


class MultivariateHighOrderFTS(fts.FTS):
    """Multivariate High Order Fuzzy Time Series"""
    def __init__(self, name, **kwargs):
        super(MultivariateHighOrderFTS, self).__init__(1, name="MVHOFTS" + name, **kwargs)
        self.name = "Multivariate High Order FTS"
        self.shortname = "MVHOFTS" + name
        self.detail = "Lee"
        self.order = 1
        self.fuzzySetsDict = {}
        self.is_high_order = True

    def build_tree(self, node, lags, level):
        if level >= self.order:
            return

        for s in lags[level]:
            node.appendChild(tree.FLRGTreeNode(s))

        for child in node.getChildren():
            self.build_tree(child, lags, level + 1)

    def build_tree_without_order(self, node, lags, level):

        if level not in lags:
            return

        for s in lags[level]:
            node.appendChild(tree.FLRGTreeNode(s))

        for child in node.getChildren():
            self.build_tree_without_order(child, lags, level + 1)

    def generate_flrg(self, data):
        flrgs = {}

        main_factor = data.ix[:, 0].values
        main_key = data.columns[0]
        main_fs = list(self.fuzzySetsDict[main_key].values())

        l = len(main_factor)
        for k in np.arange(self.order, l):

            print("Training: instance "+str(k)+ " of "+str(l))

            if self.dump: print("FLR: " + str(k))

            # Get RHS
            rhs = [set for set in main_fs if set.membership(main_factor[k]) > 0.0]

            lags = []


            for o in range(k - self.order, k):
                lhs = []
                lhs.append([set for set in main_fs if set.membership(main_factor[o]) > 0.0])

                for c in range(1,len(data.columns)):
                    sec_key = data.columns[c]
                    sec_factor = data.ix[:,sec_key].values
                    sec_fs = list(self.fuzzySetsDict[sec_key].values())
                    lhs.append([set for set in sec_fs if set.membership(sec_factor[o]) > 0.0])

                lags.append(lhs)

            flrg = MultivariateHighOrderFLRG(self.order)
            flrg.LHS = lags
            lhs_key = flrg.strLHS()

            if lhs_key not in flrgs:
                flrgs[lhs_key] = flrg

            for st in rhs:
                flrgs[lhs_key].appendRHS(st)

            # root = tree.FLRGTreeNode(None)
            #
            # self.build_tree_without_order(root, lags, 0)
            #
            # # Trace the possible paths
            # for p in root.paths():
            #     flrg = MultivariateHighOrderFTS(self.order)
            #     path = list(reversed(list(filter(None.__ne__, p))))
            #
            #     for lhs in enumerate(path, start=0):
            #         flrg.appendLHS(lhs)
            #
            #     if flrg.strLHS() not in flrgs:
            #         flrgs[flrg.strLHS()] = flrg;
            #
            #     for st in rhs:
            #         flrgs[flrg.strLHS()].appendRHS(st)

        return flrgs

    def train(self, data, sets, order=1,parameters=None):

#        data = self.doTransformations(data, updateUoD=True)

        self.order = order
        columns = data.columns

        i = 0
        for fs in sets:

            setsDict = {}

            for s in fs.sets:
                setsDict[s.name] = s
            self.fuzzySetsDict[columns[i]] = setsDict
            i += 1

        self.flrgs = self.generate_flrg(data)

    def forecast(self, data, **kwargs):

        ret = []
        main_factor = data.ix[:, 0].values
        main_key = data.columns[0]
        main_fs = list(self.fuzzySetsDict[main_key].values())

        l = len(main_factor)

        if l <= self.order:
            return data

 #       ndata = self.doTransformations(data)

        for k in np.arange(self.order, l+1):

            lags = []
            print("Forecasting: instance " + str(k) + " of " + str(l))

            for o in range(k - self.order, k):
                lhs = []
                lhs.append([set for set in main_fs if set.membership(main_factor[o]) > 0.0])
                #lhs.append([FuzzySet.getMaxMembershipFuzzySet(main_factor[o], main_fs)])


                for c in range(1,len(data.columns)):
                    sec_key = data.columns[c]
                    sec_factor = data.ix[:,sec_key].values
                    sec_fs = list(self.fuzzySetsDict[sec_key].values())
                    lhs.append([set for set in sec_fs if set.membership(sec_factor[o]) > 0.0])
                    #lhs.append([FuzzySet.getMaxMembershipFuzzySet(sec_factor[o], sec_fs)])

                lags.append(lhs)

            tmpflrg = MultivariateHighOrderFLRG(self.order)
            tmpflrg.LHS = lags

            if tmpflrg.strLHS() not in self.flrgs:
                ret.append(tmpflrg.LHS[-1][0][0].centroid)
            else:
                flrg = self.flrgs[tmpflrg.strLHS()]
                ret.append(flrg.get_midpoint())

        #ret = self.doInverseTransformations(ret, params=[data[self.order-1:]])

        return ret
