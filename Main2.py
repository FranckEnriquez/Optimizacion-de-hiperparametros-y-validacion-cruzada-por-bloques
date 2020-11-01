import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import Models
from keras.wrappers.scikit_learn import  KerasClassifier
import random
import OptimizationTools

from keras.callbacks import CSVLogger

# f_inicio = pd.to_datetime("2017-01-06 17:00:00").tz_localize('GMT')
# f_fin = pd.to_datetime("2020-06-06 17:00:00").tz_localize('GMT')
# token = '40a4858c00646a218d055374c2950239-f520f4a80719d9749cc020ddb5188887'
#
# df_pe= Data.getPrices(p0_fini=f_inicio, p1_ffin=f_fin, p2_gran="D",
#                             p3_inst=instrumento,p4_oatk=token, p5_ginc=4900)
# df_pe= Data.setTag(df_pe)
#
# chunks = BC.createBlocks(df_pe,"M")
# chunks_ = list(pd.DataFrame(chunks[i]) for i in range(len(chunks)))

#model = Models.createNN(13,8,1)


path = 'C:/Users/anuno/OneDrive/Documents/ITESO/PAP 2/'
red_wine = pd.read_csv(path+'winequality-red.csv',sep=',')
white_wine=pd.read_excel(path+'white_wine.xlsx')

red_wine["type"] = 1
white_wine["type"] = 0
wines = [red_wine, white_wine]
wines = pd.concat(wines,sort=False)
y = np.ravel(wines.type)

x = wines.loc[:,wines.columns!="type"]
y = wines["type"]
x_train, x_test, y_train, y_test_ = train_test_split(x, y, test_size=0.33)



param = OptimizationTools.optimizeNN(optimizer,5,3,x_train,y_train)