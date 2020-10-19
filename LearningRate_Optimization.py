import pandas as pd
import random
import numpy as np
# from Data import getPrices
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense  ##fully connected neural network
import keras
from keras.callbacks import CSVLogger
from OptimizationTools import learningRate_optimization

# instrumento = "EUR_USD"
# granularidad = "D"
#
# f_inicio = pd.to_datetime("2019-01-06 17:00:00").tz_localize('GMT')
# f_fin = pd.to_datetime("2019-12-06 17:00:00").tz_localize('GMT')
# token = '40a4858c00646a218d055374c2950239-f520f4a80719d9749cc020ddb5188887'
#
# df_pe= getPrices(p0_fini=f_inicio, p1_ffin=f_fin, p2_gran="H1",
#                             p3_inst=instrumento,p4_oatk=token, p5_ginc=4900)
#
#
#
def trainProcess_min(loss_history):
    min_list = []
    for col in loss_history.columns:
        min = loss_history[col].min()
        min_list.append(min)
    return min_list


def x1p_update(velocidad, c1, x1_pg, x1p, c2, x1_pL):
    x1p_update = list()
    for i in range(0, len(x1p)):
        x1p_update.append(x1p[i] + (velocidad[i] + c1 * np.random.rand() * (x1_pg - x1p[i])
                                    + c2 * np.random.rand() * (x1_pL - x1p[i])))
    return x1p_update


white_wine = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv",
                         sep=';')
red_wine = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
                       sep=';')
red_wine["type"] = 1
white_wine["type"] = 0
wines = [red_wine, white_wine]
wines = pd.concat(wines)
corr = wines.corr()

y = np.ravel(wines.type)

x = wines.ix[:, :12]
y = wines.ix[:, 12]
x_train, x_test, y_train, y_test_ = train_test_split(x, y, test_size=0.33)

model = Sequential()  # initialize network
model.add(Dense(13, activation='relu', input_shape=(12,)))  # first layer, inputs
model.add(Dense(8, activation='relu'))  # hidden layer
model.add(Dense(1, activation='sigmoid'))  # output layer
# n_particles = 10
# x1p = list(random.randrange(start=1, stop=10, step=1) / 10000 for i in range(0, n_particles))  # particulas
# x1pL = x1p
# velocidad_x1 = np.zeros(n_particles)
# x1_pg = 0
# fx_pg = 5
# fx_pL = np.ones(n_particles) * fx_pg
# c1 = 0.75
# c2 = 0.75
# history = pd.DataFrame()
# for i in range(0, 3):
#
#     for j in range(0, n_particles):
#         opt = keras.optimizers.Adam(learning_rate=x1p[j])
#         model.compile(loss='binary_crossentropy', optimizer=opt)
#         csv_logger = CSVLogger('log' + str(j) + '.csv', append=False,
#                                separator=';')
#         model.fit(x_train, y_train, epochs=10, batch_size=1, verbose=1, callbacks=[csv_logger])
#         to_read = 'log' + str(j) + '.csv'
#         fx = (pd.read_csv("C:/Users/anuno/OneDrive/Documents/ITESO/PAP 2/" + to_read,
#                           sep=';', usecols=["loss"]))
#         fx = fx.rename(columns={'loss': 'loss' + str(j)})
#
#         if j == 0:
#             history = fx
#         else:
#
#             history = pd.concat([history, fx], axis=1, sort=False)
#
#     fx = pd.DataFrame(trainProcess_min(history))
#     [val, idx] = fx.min(), fx.idxmin()[0]
#
#     if val.values < float(fx_pg):
#         fx_pg = val
#         x1_pg = x1p[idx]
#
#     for k in range(0, n_particles):
#         if fx.values[k] < fx_pL[k]:
#             fx_pL[k] = fx_pL[k]
#             x1pL = x1p[k]
#
#     x1p = x1p_update(velocidad_x1, c1, x1_pg, x1p, c2, x1pL)

[lr,f_cost]= learningRate_optimization(x_train,y_train,model,50,.75,.75,[1,10],1,1000,5)