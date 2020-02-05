import numpy as np
import copy
from .utils import *

class Granule:
    def __init__(self):
        self.iGranules = []
        self.oGranules = []
        self.act = 0
        self.points = []
        self.xs = []
        self.ys = []

    def granule_similarity(self, granule):
        """

        :param granule:
        :return:
        """
        term = 0
        for i in range(0, len(self.iGranules)):
            term = term + abs(self.iGranules[i].l - granule.iGranules[i].l) + \
                   abs(self.iGranules[i].lambd - granule.iGranules[i].lambd) + \
                   abs(self.iGranules[i].Lambd - granule.iGranules[i].Lambd) + \
                   abs(self.iGranules[i].L - granule.iGranules[i].L)

        return term

    def calculate_rls(self):
        """
        Calculate least squares
        :return:
        """
        XX = []
        colOut = copy.deepcopy(self.ys)
        col = copy.deepcopy(self.xs)

        for i in range(0, len(col)):
            XX.insert(i, copy.deepcopy(col[i]))
            XX[i].insert(0, 1)

        #try:
        for k in range(0, len(self.oGranules)):
            #result = least_sqr(XX, colOut)
            #result = lstsqr_pinv(XX, colOut)
            result = mldivide_matlab(XX, colOut)
            #result = mldivide(XX, colOut)
            #result = ls_nnls(XX, colOut)
            #result = solve_minnonzero(XX, colOut)
            self.oGranules[k].coef = result
        #except Exception as e:
        #    print(str(e))
        #    pass

    def get_granule_for_3d_plotting(self, only_core=0):
        if len(self.iGranules) > 2: raise Exception("Not possible to plot")
        if len(self.oGranules) > 1: raise Exception("Not possible to plot")

        x = self.iGranules
        y = self.oGranules

        if only_core:
            return np.array([
                [x[0].lambd, x[1].lambd, y[0].ups],
                [x[0].Lambd, x[1].lambd, y[0].ups],
                [x[0].lambd, x[1].Lambd, y[0].ups],
                [x[0].Lambd, x[1].Lambd, y[0].ups],
                [x[0].lambd, x[1].lambd, y[0].Ups],
                [x[0].Lambd, x[1].lambd, y[0].Ups],
                [x[0].lambd, x[1].Lambd, y[0].Ups],
                [x[0].Lambd, x[1].Lambd, y[0].Ups]
            ])

        return np.array([
            [x[0].l, x[1].l, y[0].u],
            [x[0].L, x[1].l, y[0].u],
            [x[0].l, x[1].L, y[0].u],
            [x[0].L, x[1].L, y[0].u],
            [x[0].l, x[1].l, y[0].U],
            [x[0].L, x[1].l, y[0].U],
            [x[0].l, x[1].L, y[0].U],
            [x[0].L, x[1].L, y[0].U]
        ])

    def get_faces_for_3d_plotting(self):
        if len(self.iGranules) > 2: raise Exception("Not possible to plot")
        if len(self.oGranules) > 1: raise Exception("Not possible to plot")

        granule = self.get_granule_for_3d_plotting()

        return [
            [granule[0, :], granule[1, :], granule[3, :], granule[2, :]],
            [granule[1, :], granule[5, :], granule[7, :], granule[3, :]],
            [granule[2, :], granule[3, :], granule[7, :], granule[6, :]],
            [granule[0, :], granule[4, :], granule[6, :], granule[2, :]],
            [granule[6, :], granule[7, :], granule[5, :], granule[4, :]],
            [granule[0, :], granule[4, :], granule[5, :], granule[1, :]]
        ]