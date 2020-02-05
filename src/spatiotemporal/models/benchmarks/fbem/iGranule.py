
class iGranule:

    def __init__(self):
        """
        Initialize object
        """
        self.l = 0
        self.lambd = 0
        self.Lambd = 0
        self.L = 0

    def fits(self, xj, rho):
        """
        Check if xj fits in this granule
        :param xj:
        :param rho:
        :return:
        """
        a = (xj >= (self.midpoint() - rho / 2))
        b = (xj <= (self.midpoint() + rho / 2))

        return a and b

    def midpoint(self):
        """
        Returns the midpoint of the granule
        :return:
        """
        return (self.lambd + self.Lambd) / 2

    def com_similarity(self, x):
        """
        Check similarity with complete observation x
        :param x: complete observation
        :param n:
        :return:
        """

        aux = abs(self.l - x) + abs(self.lambd - x) + abs(self.Lambd - x) + abs(self.L - x)

        return aux