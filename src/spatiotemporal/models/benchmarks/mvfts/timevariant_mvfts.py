from pyFTS.models.incremental import TimeVariant
from pyFTS.partitioners import Grid
from pyFTS.models.multivariate import mvfts
from pyFTS.models.multivariate import variable
from pyFTS.benchmarks import Measures
from pyFTS.common.Membership import trimf, gaussmf
from pyFTS.benchmarks import Util as bUtil

class TimeVariant_MVFTS:
    def __init__(self, **kwargs):

        mb = kwargs.pop('membership', 'triangular')
        _membership = None
        if mb == 'triangular':
            _membership = trimf
        elif mb == 'gaussian':
            _membership = gaussmf

        _npartitions = kwargs.pop('npartitions', 10)
        data = kwargs.pop('data', None)
        _order = kwargs.pop('order', 1)
        _batch = kwargs.pop('batch', 10)
        _window = kwargs.pop('window', 100)

        self.columns = data.columns

        exp_variables = []
        for vc in self.columns:
            exp_variables.append(variable.Variable(vc, data_label=vc, alias=vc,
                                                   partitioner=Grid.GridPartitioner, npart=_npartitions,
                                                   func=_membership, data=data))

        self.model = mvfts.MVFTS(explanatory_variables=exp_variables, target_variable=exp_variables[0], order=_order)


    def fit(self, data):
        self.model.fit(data)
        rmse, mape, u = Measures.get_point_statistics(test, model)
        row = [key, model.shortname, rmse, mape, u]
        rows.append(row)

        m = {'rmse': rmse, 'mape': mape, 'u': u}

        for k, v in m.items():
            data = (key, 'benchmarks', 'point', model.shortname, None, 1, 'Grid', 35, len(model), 1, None, k, v)

            bUtil.insert_benchmark(data, conn)

resultados = pd.DataFrame(rows, columns=["Dataset", "Model", "RMSE", "MAPE", "U"])