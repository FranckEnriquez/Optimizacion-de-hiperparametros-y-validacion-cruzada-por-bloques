from keras.models import Sequential
from keras.layers import Dense
import keras
import numpy as np
from keras.callbacks import CSVLogger
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics

white_wine = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv",
                         sep=';')
red_wine = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
                       sep=';')
red_wine["type"] = 1
white_wine["type"] = 0
wines = [red_wine, white_wine]
wines = pd.concat(wines)
y = np.ravel(wines.type)

x = wines.loc[:,wines.columns!="type"]
y = wines["type"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)
# y_train = np.asarray(train_labels).astype('float32').reshape((-1,1))
# y_test = np.asarray(test_labels).astype('float32').reshape((-1,1))
# def createNN(inputLayer_neurons,hiddenLayer_neurons,outputLayer_neurons,lr):
#     """
#
#     :param inputLayer_neurons: numbers of neurons for the input layer
#     :param hiddenLayer_neurons: number of neurons for the hidden layer
#     :param outLayer_neurons: number of neurons for output layer
#     """
#     model = Sequential()
#     model.add(Dense(inputLayer_neurons,activation="sigmoid",input_shape=(inputLayer_neurons-1,)))
#     model.add(Dense(hiddenLayer_neurons,activation="sigmoid"))
#     model.add(Dense(outputLayer_neurons,activation='sigmoid'))
#     opt = keras.optimizers.Adam(learning_rate=lr)
#     model.compile(loss='binary_crossentropy', optimizer=opt)
#     model.fit(x_train,y_train,verbose=1,epochs=100)
#     print("SIUUU")
#
#     return model


def createNN(neuron_pctg,lr):
    model = Sequential()
    nneurons = neuron_pctg * 100  # maximum 1000 neurons
    layer_counter = 0
    model.add(Dense(13,activation="sigmoid",input_shape=(12,)))  # input layer
    while nneurons <= 10 and layer_counter < 10:  # hidden layer limited to
        model.add(Dense(nneurons, activation="sigmoid"))
        layer_counter += 1
    model.add(Dense(1,activation="sigmoid"))
    opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(loss='binary_crossentropy',optimizer=opt,metrics=["accuracy"])


    return model


def evaluateModel(lr,neuronPctg):
    model = createNN(neuronPctg,lr)
    opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    model.fit(x_train,y_train,verbose=1,epochs=100)
    print(model)
    #pred = model.predict(x_test)
    #pred = model.predict(x_test)
    #y_compare = np.argmax(y_test,axis=1)
    #score = metrics.log_loss(y_compare,pred)

    return 0





