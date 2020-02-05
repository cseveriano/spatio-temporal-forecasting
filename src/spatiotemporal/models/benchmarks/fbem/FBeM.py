import numpy as np
import operator as op
import copy

from fbem.Granule import Granule
from fbem.iGranule import iGranule
from fbem.oGranule import oGranule
from fbem.utils import *

class FBeM:
    """
    # FBeM - Fuzzy set-based evolving Model

    FBeM is a data-driven fuzzy set-based method developed by Daniel Leite.
    https://sites.google.com/view/dleite-evolving-ai/algorithms

    Daniel Leite; Rosangela Ballini; Pyramo Costa; Fernando Gomide.
    "Evolving fuzzy granular modeling from nonstationary fuzzy data streams."
    Evolving Systems - Springer, 3(2), pp. 65-79, 2012, https://doi.org/10.1007/s12530-012-9050-9

    """
    def __init__(self):
        """
        Initialization of the object
        """
        self.data = []
        self.c = 0
        self.h = 0
        self.rho = 0.7
        self.n = 2
        self.m = 1
        self.hr = 48
        self.alpha = 0
        self.eta = 0.5
        self.counter = 1

        self.rmse = []
        self.ndei = []
        self.granules = []
        self.ys = []

        self.P = []
        self.PUB = []
        self.PLB = []
        self.store_num_rules = []
        self.vec_rho = []
        self.debug = False

        #debugging variables
        self.merged_granules = 0
        self.created_granules = 0
        self.deleted_granules = 0
        self.file = open("log.txt", "w")

    def create_new_granule(self, index, x, y):
        """
        Create new granule
        :param index:
        :param x:
        :param y:
        :return:
        """
        g = Granule()

        """ Input granules """
        for i in range(0, self.n):
            newGranule = iGranule()
            newGranule.l = x[i]
            newGranule.lambd = x[i]
            newGranule.Lambd = x[i]
            newGranule.L = x[i]

            g.iGranules.insert(i, newGranule)

        """ Output granules """
        for k in range(0, self.m):
            newOGranule = oGranule()
            newOGranule.u = y[k]
            newOGranule.ups = y[k]
            newOGranule.Ups = y[k]
            newOGranule.U = y[k]

            """ Coefficients alpha """
            newOGranule.coef = []
            newOGranule.coef.append(y[0])

            for i in range(0, self.n):
                newOGranule.coef.append(0)

            g.points.append(self.h) #
            g.act = 0
            g.oGranules.insert(k, newOGranule)

        g.xs.append(x)
        g.ys.append(y[0])

        self.granules.insert(index, g)
        self.c += 1

        self.created_granules += 1

        """ Debugging """
        """
        if self.debug:
            ax = create_3d_space()
            ax = plot_granule_3d_space(granule=g, ax=ax, i=3)

            plot_show()
        """
        """ /Debugging """


    def create_check_imaginary_granule(self, granule1, granule2):
        """
        Create new imaginary granule as a junction of two granules
        And check the possibility of this granule to become real
        :param granule1:
        :param granule2:
        :return:
        """
        g = Granule()
        J = 0
        K = 0

        """ Input granules """
        for i in range(0, self.n):
            new_granule = iGranule()
            gran_1 = granule1.iGranules[i]
            gran_2 = granule2.iGranules[i]

            new_granule.l = min([gran_1.l, gran_2.l])
            new_granule.lambd = min([gran_1.lambd, gran_2.lambd])
            new_granule.Lambd = max([gran_1.Lambd, gran_2.Lambd])
            new_granule.L = max([gran_1.L, gran_2.L])

            if new_granule.midpoint() - self.rho / 2 <= new_granule.l and new_granule.midpoint() + self.rho / 2 >= new_granule.L:
                J = J + 1

            g.iGranules.insert(i, new_granule)

        """ Output granules """
        for k in range(0, self.m):
            newOGranule = oGranule()
            gran_1 = granule1.oGranules[k]
            gran_2 = granule2.oGranules[k]

            newOGranule.u = min([gran_1.u, gran_2.u])
            newOGranule.ups = min([gran_1.ups, gran_2.ups])
            newOGranule.Ups = max([gran_1.Ups, gran_2.Ups])
            newOGranule.U = max([gran_1.U, gran_2.U])

            if newOGranule.midpoint() - self.rho / 2 <= newOGranule.u and newOGranule.midpoint() + self.rho / 2 >= newOGranule.U:
                K = K + 1

            """ Coefficients alpha """
            g.points = granule1.points + granule2.points
            g.xs = granule1.xs + granule2.xs
            g.ys = granule1.ys + granule2.ys
            g.act = 0

            newOGranule.coef = [(x + y) / 2 for x, y in zip(granule1.oGranules[k].coef, granule2.oGranules[k].coef)]

            g.oGranules.insert(k, newOGranule)

        """ Check if new imaginary granule can become real """
        become_real = False
        if J + K == self.n + self.m:
            self.c = self.c + 1
            g.calculate_rls()

            self.granules.insert(self.c, g)
            self.merged_granules += 1
            become_real = True


        """ Debugging """
        """
        if self.debug:
            ax = create_3d_space()
            ax = plot_granule_3d_space(granule=granule1,ax=ax)
            ax = plot_granule_3d_space(granule=granule2,ax=ax,i=2)
            ax = plot_granule_3d_space(granule=g, ax=ax, i=3)

            plot_show()
        """
        """ /Debugging """

        return become_real

    def learn(self, x, y):
        """
        Learn as x and y enters
        :param x: observations
        :param y: expected output
        :return:
        """
        # starting from scratch
        if self.h == 0:

            self.ys.append(y[0])

            """ create new granule anyways """
            self.create_new_granule(self.c, x, y)

            self.P.append(np.random.rand())
            self.PUB.append(1)  # Upper bound
            self.PLB.append(0)  # Lower bound
            self.store_num_rules.insert(self.h, self.c)
            self.vec_rho.insert(self.h, self.rho)

            #self.rmse.append(np.sqrt(sum_(power(sub(self.ys, self.P), 2)) / (self.h + 1)))
            self.rmse.append(np.sqrt(sum_(power(sub(self.ys, self.P), 2))))
            self.ndei.append(np.inf)
            self.file.write(str(self.h) + "\t" + str(self.P[self.h]) + "\t1\n"  )

            """
            # Avoiding division by 0
            if np.std(self.ys):
                self.ndei.append(self.rmse[self.h] / np.std(self.ys))
            else:
                self.ndei.append(np.inf)
            """
        else:

            """
            Incremental learning
            Test - y is not available yet
            Check rules that can accommodate x
            """

            I = []
            for i in range(0, self.c):
                J = 0
                for j in range(0, self.n):
                    if self.granules[i].iGranules[j].fits(xj=x[j], rho=self.rho):
                        J += 1

                if J == self.n:
                    """ granule i fits x! """
                    I.append(i)

            """ Check degree of similarity """
            S = []
            for i in range(0, self.c):
                aux = 0
                for j in range(0, self.n):
                    aux += self.granules[i].iGranules[j].com_similarity(x=x[j])

                S.insert(i, 1 - ((1 / (4 * self.n)) * aux))

            """ If no rule encloses x """
            if len(I) == 0:
                I = [S.index(max(S))]

            ## Store points in the granules
            #for i in I:
            #    self.granules[i].xs.append(x)
            #    self.granules[i].points.append(self.h)

            """ Calculate functional consequent """
            if self.h == 562:
                stop = 0

            p = []
            for i in range(0, len(I)):
                for j in range(0, self.m):
                    p_calc = self.granules[I[i]].oGranules[j].p(x=x)
                    p.insert(i, p_calc)
                    ## Store ypoints in the granules
                    #self.granules[I[i]].ys.append(p_calc)

            """ Prediction """
            part_1 = sum_dot(S, I, p)
            part_2 = sum_specific(S, I)
            part_3 = part_1 / part_2

            self.P.insert(self.h, part_3)
            self.PLB.insert(self.h, min_u(granules=self.granules, indices=I, m=self.m))
            self.PUB.insert(self.h, max_U(granules=self.granules, indices=I, m=self.m))

            self.file.write(str(self.h) + '\t' + str(self.P[self.h]) + '\t' + str(I) + '\n')

            """ P must belong to [PLB PUB] """
            if self.P[self.h] < self.PLB[self.h]:
                self.P[self.h] = self.PLB[self.h]

            if self.P[self.h] > self.PUB[self.h]:
                self.P[self.h] = self.PUB[self.h]

            self.store_num_rules.insert(self.h, self.c)  # number of rules

            """ y becomes available // cumulative """
            self.ys.append(y[0])

            # RMSE
            part = sub(self.ys, self.P)
            part = power(part, 2)
            part = sum_(part)
            part = np.sqrt(part / (self.h + 1))

            self.rmse.append(part)
            self.ndei.append(self.rmse[self.h] / np.std(self.ys))

            #self.rmse.append(sqroot(sum_(power(sub(self.ys, self.P), 2)) / (self.h + 1)))
            #self.ndei.insert(self.h, self.rmse[self.h] / np.std(self.ys))

            """
            Train
            Calculate granular consequent
            Check rules that accommodate x
            """
            """
            stop = 0
            if self.h == 24:
                stop = 1
            """

            I = []
            for i in range(0, self.c):
                J = 0
                K = 0

                #
                for j in range(0, self.n):
                    """ xj inside j-th expanded region? """
                    if self.granules[i].iGranules[j].fits(xj=x[j], rho=self.rho):
                        J += 1

                for k in range(0, self.m):
                    """ yk inside k-th expanded region? """
                    if self.granules[i].oGranules[k].fits(y=y[k], rho=self.rho):
                        K += 1

                if J + K == self.n + self.m:
                    I.append(i)

            """ Case 0: no granule fits x """
            if len(I) == 0:
                self.create_new_granule(x=x, y=y, index=self.c)
            else:
                """
                Adaptation of the most qualified granule
                If more than one granule fits the observation
                """
                if len(I) >= 2:
                    S = []
                    for i in range(0, len(I)):
                        aux = 0
                        for j in range(0, self.n):
                            aux += self.granules[I[i]].iGranules[j].com_similarity(x=x[j])

                        S.insert(I[i], 1 - ((1 / (4 * self.n)) * aux))

                    I = I[S.index(max(S))]
                else:
                    I = I[0]

                """ Adapting antecedent of granule I """
                """ Debugging """
                """
                if self.debug and stop:
                    ax = create_3d_space()
                    ax = plot_granule_3d_space(granule=self.granules[I], ax=ax, i=3)
                    ax.scatter(x[0], x[1], self.granules[I].oGranules[0].p(x=x))
                    ax.set_title("Teste")
                    plot_show()
                """
                """ /Debugging """
                for j in range(0, self.n):
                    ig = self.granules[I].iGranules[j]
                    mp = ig.midpoint()
                    if mp - self.rho / 2 < x[j] < ig.l: #1 ok
                        ig.l = x[j]  # Support expansion
                    if mp - self.rho / 2 < x[j] < ig.lambd: #2 ok
                        ig.lambd = x[j] # core expansion
                    if ig.lambd < x[j] < mp: #3 ok
                        ig.lambd = x[j] # core contraction
                    if mp < x[j] < mp + self.rho / 2 : #4 ok
                        ig.lambd = mp # core contraction
                    if mp - self.rho / 2 < x[j] < mp:
                        ig.Lambd = mp  # Core contraction
                    if mp < x[j] < ig.Lambd:
                        ig.Lambd = x[j] # core contraction
                    if ig.Lambd < x[j] < mp + self.rho/2:
                        ig.Lambd = x[j] # core expansion
                    if ig.L < x[j] < mp + self.rho / 2:
                        ig.L = x[j] # support expansion
                    """
                    if ig.l < x[j] < ig.lambd:
                        ig.lambd = x[j]  # Core expansion
                    if mp < x[j] < ig.Lambd:
                        ig.Lambd = x[j]  # Core contraction
                    if ig.Lambd < x[j] < ig.L:
                        ig.Lambd = x[j]  # Core expansion
                    if ig.Lambd < x[j] < mp + self.rho / 2:
                        ig.Lambd = x[j]  # Support expansion
                    if ig.L < x[j] < mp + self.rho / 2:
                        ig.L = x[j]  # Support expansion
                    """

                """ Check if support contraction is needed """
                for j in range(0, self.n):
                    ig = self.granules[I].iGranules[j]
                    mp = self.granules[I].iGranules[j].midpoint()

                    """ Inferior support """
                    if mp - self.rho / 2 > ig.l:
                        ig.l = mp - self.rho / 2
                        if mp - self.rho / 2 > ig.lambd:
                            ig.lambd = mp - self.rho / 2

                    """ Superior Support """
                    if mp + self.rho / 2 < ig.L:
                        ig.L = mp + self.rho / 2
                        if mp + self.rho / 2 < ig.Lambd:
                            ig.Lambd = mp + self.rho / 2

                """ Adapting consequent granule I """
                for k in range(0, self.m):
                    og = self.granules[I].oGranules[k]
                    mp = self.granules[I].oGranules[k].midpoint()

                    if mp - self.rho / 2 < y[k] < og.u: #1 ok
                        og.u = y[k]  # Support expansion
                    if mp - self.rho / 2 < y[k] < og.ups: #2 ok
                        og.ups = y[k] # core expansion
                    if og.ups < y[k] < mp: #3 ok
                        og.ups = y[k] # core contraction
                    if mp < y[k] < mp + self.rho / 2 : #4 ok
                        og.ups = mp # core contraction
                    if mp - self.rho / 2 < y[k] < mp:
                        og.Ups = mp  # Core contraction
                    if mp < y[k] < og.Ups:
                        og.Ups = y[k] # core contraction
                    if og.Ups < y[k] < mp + self.rho/2:
                        og.Ups = y[k] # core expansion
                    if og.U < y[k] < mp + self.rho / 2:
                        og.U = y[k] # support expansion

                    """
                    if mp - self.rho / 2 < y[k] < og.u:
                        og.u = y[k]  # Support expansion
                    if og.u < y[k] < og.ups:
                        og.ups = y[k]  # Core expansion
                    if og.ups < y[k] < mp:
                        og.ups = y[k]  # Core contraction
                    if mp < y[k] < og.Ups:
                        og.Ups = y[k]  # Core contraction
                    if og.Ups < y[k] < og.U:
                        og.Ups = y[k]  # Core expansion
                    if og.U < y[k] < mp + self.rho / 2:
                        og.U = y[k]  # Support expansion
                    """
                """ Check if support contraction is needed """
                for k in range(0, self.m):
                    og = self.granules[I].oGranules[k]
                    mp = self.granules[I].oGranules[k].midpoint()

                    """ Inferior support """
                    if mp - self.rho / 2 > og.u:
                        og.u = mp - self.rho / 2
                        if mp - self.rho / 2 > og.ups:
                            og.ups = mp - self.rho / 2

                    """ Superior support """
                    if mp + self.rho / 2 < og.U:
                        og.U = mp + self.rho / 2
                        if mp + self.rho / 2 < og.Ups:
                            og.Ups = mp + self.rho / 2

                """ Storing sample """
                self.granules[I].points.append(self.h)
                self.granules[I].ys.append(y[0])
                self.granules[I].xs.append(x)
                self.granules[I].act = 0

                """ Least Squares """
                self.granules[I].calculate_rls()  # ate aqui

                """ Debugging """
                """
                if self.debug:
                    ax = create_3d_space()
                    ax = plot_granule_3d_space(granule=self.granules[I], ax=ax, i=3)
                    ax.scatter(x[0], x[1], self.granules[I].oGranules[0].p(x=x), c="green")
                    ax.scatter(x[0], x[1], y[0], c="yellow")
                    ax.set_title("Teste2")
                    plot_show()
                """
                """ /Debugging """

        """ Deleting granules if needed """
        granules_to_delete = []
        for K in range(0, self.c):
            self.granules[K].act += 1
            if self.granules[K].act >= self.hr:
                del self.granules[K]
                self.c -= 1
                break

        """ Coarsening granular structure """
        self.alpha += 1

        if self.alpha == self.hr:
            if self.c >= 3:
                """
                Calculating the similarity between granules
                While choosing two closest granules acording to S
                """
                S = np.zeros((self.c, self.c))
                gra1 = []
                ind1 = -1
                gra2 = []
                ind2 = -1
                aux = -np.inf

                for i1 in range(0, self.c):
                    for i2 in range(i1 + 1, self.c):
                        S[i1][i2] = self.granules[i1].granule_similarity(self.granules[i2])
                        S[i1][i2] = 1 - ((1/(4 * self.n)) * S[i1][i2])
                        if aux < S[i1][i2]:
                            aux = S[i1][i2]
                            gra1 = self.granules[i1]
                            gra2 = self.granules[i2]
                            ind1 = i1
                            ind2 = i2

                res = self.create_check_imaginary_granule(granule1=gra1, granule2=gra2)
                if res:
                    """ deleting granules is possible """
                    del self.granules[ind1]
                    del self.granules[ind2]
                    self.c -= 2

            self.alpha = 0

        """ Adapt granules size """
        if self.counter == 1:
            self.b = self.c

        self.counter += 1

        if self.counter == self.hr:
            self.chi = self.c
            diff = self.chi - self.b

            if diff >= self.eta:
                self.rho *= (1 + diff / self.counter)  # increase rho
            else:
                self.rho *= (1 - (diff + self.eta) / self.counter)  # decrease rho

            self.counter = 1

        self.vec_rho.insert(self.h, self.rho)  # granules size along

        self.h += 1

        """ Check if support contraction is needed """
        for i in range(0, self.c):
            for j in range(0, self.n):

                ig = self.granules[i].iGranules[j]
                mp = self.granules[i].iGranules[j].midpoint()

                """ Inferior support """
                if mp - self.rho / 2 > ig.l:
                    ig.l = mp - self.rho / 2
                    if mp - self.rho / 2 > ig.lambd:
                        ig.lambd = mp - self.rho / 2

                """ Superior Support """
                if mp + self.rho / 2 < ig.L:
                    ig.L = mp + self.rho / 2
                    if mp + self.rho / 2 < ig.Lambd:
                        ig.Lambd = mp + self.rho / 2

    def fit(self, data, order=1):
        self.n = order
        xdata = [x[0] for x in data]
        for k in np.arange(self.n, len(xdata)):
            x = xdata[k-self.n:k]
            y = xdata[k]
            self.learn(x,[y])

    def predict(self, data):
        fcst = []
        xdata = [x[0] for x in data]
        for k in np.arange(self.n, len(xdata)):
            x = xdata[k-self.n:k]
            y = xdata[k]
            self.learn(x,[y])
        fcst.extend(self.P[-len(xdata)+self.n:])
        fcst.append(0)
        return fcst
