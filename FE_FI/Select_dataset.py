# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 11:00:53 2020
@author: franck
"""

import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import f_classif
import Create_dataset
import matplotlib.pyplot as plt
pd.options.mode.use_inf_as_na = True

dataset = Create_dataset.create('2012-11-07', '2020-10-28')
dataset = dataset.fillna(dataset.mean())
print(dataset.shape)
#long = int(round(len(df) * 0.80))
#df['Crime_Rate'].fillna(df['Crime_Rate'].max(),inplace=True)

def ANOVA_importance(df,
                     sample:float,
                     VO:str):
    
    '''
    Return the index of the variables with the most
    statistical significance with p-value approach
    There is F statistical approach'''

    long = int(round(len(df) * sample))

    X = df.drop(VO, axis=1)
    y = df[VO]

    #select train and test data
    X_train , X_test , y_train , y_test = X.iloc[:long,:] , X.iloc[long:,:] , \
                                                y.iloc[:long], y.iloc[long:]

    #train model
    constant_filter = VarianceThreshold(threshold=0.01)
    constant_filter.fit(X_train)
    #print(constant_filter)
    X_train_filter = constant_filter.transform(X_train)
    X_test_filter = constant_filter.transform(X_test)
    
    #transpose data
    X_train_T = pd.DataFrame(X_train_filter.T)
    X_test_T = pd.DataFrame(X_test_filter.T)
    
    #eliminate duplicated features
    duplicated_features = X_train_T.duplicated()
    
    #choose features to keep 
    features_to_keep = [not index for index in duplicated_features]
    X_train_unique = X_train_T[features_to_keep].T
    X_test_unique = X_test_T[features_to_keep].T
    #X_test_unique

    #ANOVA SECTION
    sel = f_classif(X_train_unique, y_train)
    
    #choose the p_values < 0.05
    p_values = pd.Series(sel[1])
    p_values.index = X_train_unique.columns
    #p_values.sort_values(ascending=True, inplace=True)
    p_values = p_values[p_values<0.05]

    #data_imp = df[df.index==p_values.index]
    #df.iloc[p_values.index]
    df = pd.concat([df.iloc[:,p_values.index],y,df.Close],axis=1)
    return df

#print(ANOVA_importance(dataset,0.79,'Label'))

df = ANOVA_importance(dataset,0.79,'Label')

print(df.head())
print(df.columns)
print(df.shape)

#df.to_csv("../Feature_Creation/FEFI_1220.csv")

#df.to_csv("../Feature_Creation/FEFI_1220.csv")
