import random
import numpy as np
import pandas as pd
import keras
from keras.callbacks import CSVLogger

def learningRate_optimization(x_train,y_train,model,n_particles,c1,c2,search_space,step,scale,iter):

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

    x1p = list(random.randrange(start=search_space[0], stop=search_space[1], step=step) / scale for i in range(0,n_particles))
    x1pL = x1p
    velocidad_x1 = np.zeros(n_particles)
    x1_pg = 0
    fx_pg = 1
    fx_pL = np.ones(n_particles) * fx_pg
    history = pd.DataFrame()


    for i in range(0, iter):

        for j in range(0, n_particles):
            opt = keras.optimizers.Adam(learning_rate=x1p[j])
            model.compile(loss='binary_crossentropy', optimizer=opt)
            csv_logger = CSVLogger('log' + str(j) + '.csv', append=False,
                                   separator=';')
            model.fit(x_train, y_train, epochs=10, batch_size=1, verbose=1, callbacks=[csv_logger])
            to_read = 'log' + str(j) + '.csv'
            fx = (pd.read_csv("C:/Users/anuno/OneDrive/Documents/ITESO/PAP 2/" + to_read,
                              sep=';', usecols=["loss"]))
            fx = fx.rename(columns={'loss': 'loss' + str(j)})

            if j == 0:
                history = fx
            else:

                history = pd.concat([history, fx], axis=1, sort=False)

        fx = pd.DataFrame(trainProcess_min(history))
        [val, idx] = fx.min(), fx.idxmin()[0]

        if val.values < float(fx_pg):
            fx_pg = val
            x1_pg = x1p[idx]

        for k in range(0, n_particles):
            if fx.values[k] < fx_pL[k]:
                fx_pL[k] = fx_pL[k]
                x1pL = x1p[k]

        x1p = x1p_update(velocidad_x1, c1, x1_pg, x1p, c2, x1pL)

        return [x1_pg,fx_pg]