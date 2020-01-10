import pandas as pd
import numpy as np
from fbem.FBeM import FBeM
from fbem.utils import *
from spatiotemporal.models.clusteredmvfts.fts import evolvingclusterfts
import math
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def calculate_rmse(test, forecast, order, step):
    rmse = math.sqrt(mean_squared_error(test[(order):], forecast[:-step]))
    print("RMSE : " + str(rmse))
    return rmse


xls = pd.ExcelFile("../../../data/processed/FBEM/DeathValleyAvg.xls")
sheetx = xls.parse(0)

n = 2

# Preparing x
x = []
for index, row in sheetx.iterrows():
    x = x + row[1:].tolist()
xavg = np.zeros((2, len(x) - n - 1))

# Preparing y
yavg = []
for i in range(0, len(x) - n - 1):
    for j in range(0, n):
        xavg[j][i] = x[i+j]
    yavg.insert(i, x[i+j+1])


fbi = FBeM()
fbi.debug = True
to_normalize = 1

# Normalize data
if to_normalize:
    fbi.rho = 0.3
    min_v = min(yavg)
    max_v = max(yavg)
    xavg = normalize(array=xavg, min=min_v, max=max_v)
    yavg = normalize(array=yavg, min=min_v, max=max_v)
else:
    min_v = min(yavg)
    max_v = max(yavg)
    fbi.rho = 0.2 * (max_v - min_v)

axis_1 = len(yavg)
axis_2 = len(xavg)

x = []
y = []

evolving = evolvingclusterfts.EvolvingClusterFTS(t_norm='nonzero', defuzzy='weighted', variance_limit=0.001)
evolving_yhat = []
evolving_error_list = []
persistence_yhat = []
persistence_error_list = []

for i in range(0, axis_1):
    x = []
    y = []
    for j in range(0, axis_2):
        x.append(xavg[j][i])

    y.append(yavg[i])

    fbi.learn(x=x, y=y)

    de = np.empty((0,2),float)
    for i in x:
        de = np.append(de, np.array([[i] * 2]), axis=0)
    de = np.append(de, np.array([y * 2]), axis=0)
    evolving.fit(de, order=n, num_batches=None)
    y_hat = evolving.predict(x)
    y_hat = y_hat[-1]
    evolving_yhat.append(y_hat)

    part = y - y_hat
    part = power(part, 2)
    part = sum_(part)
    part = np.sqrt(part / (fbi.h + 1))
    evolving_error_list.append(part)

    persistence_yhat.append(x[-1])
    part = y - x[-1]
    part = power(part, 2)
    part = sum_(part)
    part = np.sqrt(part / (fbi.h + 1))
    persistence_error_list.append(part)

## Comparar RMSE
print("FBeM - Average RMSE: ", fbi.rmse[len(fbi.rmse) - 1])
print("Evolving FTS - Average RMSE: ", evolving_error_list[len(fbi.rmse) - 1])
print("Persistence - Average RMSE: ", persistence_error_list[len(fbi.rmse) - 1])
fbi.file.close()

## Comparar graficos
limit = 200
fig, axs = plt.subplots(3, 1, constrained_layout=True)
axs[0].plot(yavg[:limit], 'k-', label="Expected output")
axs[0].plot(fbi.P[:limit], 'b-', label="Predicted output")
axs[0].set_title('FBeM')
fig.suptitle('Death Valley Dataset', fontsize=16)

axs[1].plot(yavg[:limit], 'k-', label="Expected output")
axs[1].plot(evolving_yhat[:limit], 'b-', label="Predicted output")
axs[1].set_title('Evolving FTS')

axs[1].plot(yavg[:limit], 'k-', label="Expected output")
axs[1].plot(persistence_yhat[:limit], 'b-', label="Predicted output")
axs[1].set_title('Persistence')

plt.show()