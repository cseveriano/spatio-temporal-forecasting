
class oGranule:

    def __init__(self):
        self.u = 0
        self.ups = 0
        self.Ups = 0
        self.U = 0
        self.coef = []

    def p(self, x):
        """
        Functional consequent with complete data
        :param x:
        :return:
        """

        # Calculate a0
        out = self.coef[0]

        # Calculate other terms
        for i in range(1, len(self.coef)):
            out = out + self.coef[i] * x[i-1]

        return out

    def p_mp(self, x):
        """
        Functional consequent with complete data, taking into account midpoint
        :param x:
        :return:
        """

        # Calculate a0
        out = self.coef[0]

        # Calculate other terms
        for i in range(1, len(self.coef)):
            out = out + self.coef[i] * x[i-1]

        return out

    def midpoint(self):
        """
        Returns the midpoint of the granule
        :return:
        """
        return (self.ups + self.Ups) / 2

    def fits(self, y, rho):
        """
        Check if xj fits in this granule
        :param xj:
        :param rho:
        :return:
        """
        a = (y >= self.midpoint() - rho / 2)
        b = (y <= self.midpoint() + rho / 2)

        return a and b