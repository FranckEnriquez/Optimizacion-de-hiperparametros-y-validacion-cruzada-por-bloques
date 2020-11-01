import random
import numpy as np
import pandas as pd
import keras
from keras.callbacks import CSVLogger
import Models

optimizer ={
    'neuron_pctg':(0.5,.1,0.2,0.3,0.4,0.5),
    'lr':(0.0006,.0008,.001,.0012,0.0014,.0016,.0018,.002)

}


optimizer ={
    'learning_rate': {
                    'start':0,
                   'stop':1,
                    'step':1,
                   'scale':1000,
                   },
    'neuron percentage': {
                    'start':1,
                    'end': 5,
                    'step':1,
                    'scale':10
                            }
}


def learningRate_optimization(x_train,y_train,model,lr_searchSpace,step,scale,n_particles,c1,c2,iter):

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
                                        + c2 * np.random.rand() * (x1_pL[i] - x1p[i])))
        return x1p_update


    x1p = list(random.randrange(start=lr_searchSpace[0], stop=lr_searchSpace[1], step=step) / scale for i in range(0,n_particles))

    x1pL = x1p
    velocidad_x1 = np.zeros(n_particles)
    x1_pg = 0
    fx_pg = 1
    fx_pL = np.ones(n_particles) * fx_pg
    history = pd.DataFrame()
    final_model=model

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

def PSO(x_train,y_train,model,optimizer,n_particles,c1,c2,iter):

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

    def createParamRanges(optimizer_dict, n_particles):
        range_dict = {}
        for i in optimizer_dict.keys():
            range_dict[i] = list(random.randrange(start=optimizer_dict[i]["start"], stop=optimizer_dict[i]["end"], step=
            optimizer_dict[i]["step"]) / optimizer_dict[i]["scale"] for j in range(0, n_particles))

        return range_dict

    def localparam_update(x1_pl,x1p,position):
        for i in x1_pl.keys():
            x1_pl[i][position]=x1p[i][position]
        return x1_pl

    def paramspeed_update(x1p,velocidad, c1, x1_pg,  c2, x1_pL):
        x1p_updated={}
        for i in x1p.keys():
            x1p_updated[i]=x1p_update(velocidad, c1, x1_pg, x1p, c2, x1_pL)

        return x1p_updated

    x1p = createParamRanges(optimizer, 50) #diccionario con los valores para cada parámetro


    x1pL = x1p
    velocidad_x1 = np.zeros(n_particles)
    x1_pg = 0
    fx_pg = 1
    fx_pL = np.ones(n_particles) * fx_pg
    history = pd.DataFrame()

    for i in range(0, iter):

        for j in range(0, n_particles):
            # opt = keras.optimizers.Adam(learning_rate=x1p["learning_rate"],)
            # model.compile(loss='binary_crossentropy', optimizer=opt)

            model = Models.createNN(x1p["learning_rate"][j],x1p["neuron percentage"][j])
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
                x1pL[k] = localparam_update(x1pL,x1p,k) #diccionario de parámetros

        x1p = paramspeed_update(x1p,velocidad_x1, c1, x1_pg, c2, x1pL)

    return [x1_pg, fx_pg]

def optimizeNN(param_dict,n_particles, iter,x_train,y_train):

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
                                        + c2 * np.random.rand() * (x1_pL[i] - x1p[i])))
        return x1p_update

    def createParamRanges(optimizer_dict, n_particles):
        range_dict = {}
        for i in optimizer_dict.keys():
            range_dict[i] = list(random.randrange(start=optimizer_dict[i]["start"], stop=optimizer_dict[i]["end"], step=
            optimizer_dict[i]["step"]) / optimizer_dict[i]["scale"] for j in range(0, n_particles))

        return range_dict

    def localparam_update(x1_pl, x1p, position):
        x1_pl = x1_pl[position]
        for i in x1_pl:
            x1_pl[i] = x1p[position][i]
        return x1_pl

    def paramspeed_update(x1p, velocidad, c1, x1_pg, c2, x1_pL):
        x1p_updated = {}
        for i in x1p.keys():
            if type(i) != int:
                x1p[i] = x1p[i]
                x1p_updated[i] = x1p_update(velocidad, c1, x1_pg, x1p[i], c2, x1_pL[i])
            else:
                continue
        return x1p_updated

    def set_keys(optimizer):
        out = {}
        for i in optimizer.keys():
            out[i] = 0
        return out

    c1, c2 = 0.75, 0.75

    x1p = createParamRanges(param_dict, n_particles)  # diccionario con los valores para cada parámetro
    x1pL = x1p

    velocidad_x1 = np.zeros(n_particles)
    x1_pg = 0
    x2_pg = 0  # agregar mas x_pg en caso de mas parámetros
    fx_pg = 1
    fx_pL = np.ones(n_particles) * fx_pg
    history = pd.DataFrame()
    parametros_optimos = 0
    for i in range(0, iter):

        for j in range(0, n_particles):
            model = Models.createNN(lr=x1p["learning_rate"][j], neuron_pctg=x1p["neuron percentage"][j])
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

        if val.values < float(fx_pg):  # extender en caso de mas parametros
            fx_pg = val
            x1_pg = x1p["learning_rate"][idx]
            x2_pg = x1p['neuron percentage'][idx]

        for k in range(0, n_particles):
            if fx[0][k] < fx_pL[k]:
                fx_pL[k] = fx[0][k]
                x1pL['learning_rate'][k] = x1p["learning_rate"][k]  # diccionario de parámetros
                x1pL['neuron percentage'][k] = x1p["neuron percentage"][k]
        x1p = paramspeed_update(x1p, velocidad_x1, c1, x1_pg, c2, x1pL)
        parametros = [x1_pg, x2_pg]

    parametros_optimos = set_keys(optimizer)
    parametros_optimos = dict(zip(parametros_optimos.keys(), parametros))
    parametros_optimos["Funcion de costo"]= fx_pg.values
    return [parametros_optimos,fx_pg]
