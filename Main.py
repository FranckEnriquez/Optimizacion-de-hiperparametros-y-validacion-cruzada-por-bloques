import pandas as pd
from OptimizationTools import learningRate_optimization
import Data
import Models
from sklearn.model_selection import train_test_split
import numpy as np
from bayes_opt import BayesianOptimization
from keras.wrappers.scikit_learn import  KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense ##fully connected neural network

#model = Models.createNN(13,8,1)


#model = Models.createNN(13,1,.3)

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



parameter_bounds = {
                    'lr':(0.006,0.002),
                    'neuronPctg':(0,0.1)

                            }
optimizer ={
    'neuron_pctg':(0.5,.1,0.2,0.3,0.4,0.5),
    'lr':(0.0006,.0008,.001,.0012,0.0014,.0016,.0018,.002)

}


optimizer.maximize(init_points=10,n_iter=5)
val = Models.evaluateModel(.0001,.10)
model = KerasClassifier(build_fn=Models.createNN,lr=.001,neuron_pctg=.1,epochs=10)
grid = GridSearchCV(estimator=model,param_grid=optimizer,n_jobs=-1)
grid_result = grid.fit(x_train,y_train)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

model= Sequential() #initialize network
model.add(Dense(13,activation='relu',input_shape=(12,))) #first layer, inputs
model.add(Dense(8,activation='relu')) #hidden layer
model.add(Dense(1,activation='sigmoid')) #output layer

from OptimizationTools import learningRate_optimization
[lr,cost]=learningRate_optimization(x_train,y_train,model,50,.75,.75,[7,20],1,10000,2)