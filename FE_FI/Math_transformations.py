# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 13:28:25 2020

@author: Esteban
"""
import numpy as np

def math_transformations(df):
    original_columns = df.columns
    for col in original_columns:
        df['sin_'+col] = np.sin(df[col].values.astype(float))
        df['cos_'+col] = np.cos(df[col].values.astype(float))
        df['square_'+col] = np.square(df[col].values.astype(float))
        df['sqrt_'+col] = np.sqrt(df[col].values.astype(float))
        df['exp_'+col] = np.exp(df[col].values.astype(float))
        df['exp2_'+col] = np.exp2(df[col].values.astype(float))
        df['tanh_'+col] = np.tanh(df[col].values.astype(float))
        df['arctan_'+col] = np.arctan(df[col].values.astype(float))
        df['log_'+col] = np.log(df[col].values.astype(float))
        df['log2_'+col] = np.log2(df[col].values.astype(float))
        df['log10_'+col] = np.log10(df[col].values.astype(float))
        df['sindiff_' + col] = np.sin(df[col].values.astype(float))**(1/2)
        df['cosdiff_' + col] = np.cos(df[col].values.astype(float))**(1/2)
        df['squarediff_' + col] = np.square(df[col].values.astype(float))**(1/2)
        df['sqrtdiff_' + col] = np.sqrt(df[col].values.astype(float))**(1/2)
        df['expdiff_' + col] = np.exp(df[col].values.astype(float))**(1/2)
        df['exp2diff_' + col] = np.exp2(df[col].values.astype(float))**(1/2)
        df['tanhdiff_' + col] = np.tanh(df[col].values.astype(float))**(1/2)
        df['arctandiff_' + col] = np.arctan(df[col].values.astype(float))**(1/2)
        df['logdiff_' + col] = np.log(df[col].values.astype(float))**(1/2)
        df['log2diff_' + col] = np.log2(df[col].values.astype(float))**(1/2)
        df['log10diff_' + col] = np.log10(df[col].values.astype(float))**(1/2)
    return df
