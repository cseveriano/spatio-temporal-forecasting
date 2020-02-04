from pyFTS.partitioners.Grid import GridPartitioner
from pyFTS.common.Membership import gaussmf, trimf
from pyFTS.models.multivariate import granular
from pyFTS.models.multivariate import variable
import pandas as pd
import numpy as np

class GranularFTS:
    def __init__(self, **kwargs):
        mb = kwargs.pop('membership', 'triangular')
        _membership = None
        if mb == 'triangular':
            _membership = trimf
        elif mb == 'gaussian':
            _membership = gaussmf

        _npartitions = kwargs.pop('npartitions', 10)
        _alpha_cut = kwargs.pop('alpha_cut', 0.3)
        _knn = kwargs.pop('knn', 2)
        data = kwargs.pop('data', None)
        _order = kwargs.pop('order', 1)

        self.columns = data.columns

        exp_variables = []
        for vc in self.columns:
            exp_variables.append(variable.Variable(vc, data_label=vc, alias=vc,
                                                   partitioner=GridPartitioner, npart=_npartitions,
                                                   func=_membership, data=data, alpha_cut=_alpha_cut))

        self.model = granular.GranularWMVFTS(explanatory_variables=exp_variables, target_variable=exp_variables[0], order=_order,
                                knn=_knn)

    def fit(self, data, order=1):
        df = pd.DataFrame(data=data, columns=self.columns)
        self.model.fit(df)

    def predict(self, data, steps=1):
        df = pd.DataFrame(data=data, columns=self.columns)
        fcst = self.model.predict(df, type='multivariate').values
        fcst = np.vstack((fcst, fcst[-1]))
        return fcst
