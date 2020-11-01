# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 11:57:34 2020

@author: Esteban

Creacion de Features para el dataset de entrenamiento de un modelo de ML
que se usara en un sistema de trading diario automatizado
pip install oandapyV20
Se necesita tener Diferenciacion_fraccional en la misma ruta
"""
import pandas as pd
import numpy as np
import Data
from Diferenciacion_fraccional import least_diff
import ruptures as rpt

# Download prices from Oanda into df_pe
instrumento = "EUR_USD"
granularidad = "D"

f_inicio = pd.to_datetime("2016-01-01 17:00:00").tz_localize('GMT')
f_fin = pd.to_datetime("2020-10-23 17:00:00").tz_localize('GMT')
token = '40a4858c00646a218d055374c2950239-f520f4a80719d9749cc020ddb5188887'

df_pe = Data.getPrices(p0_fini=f_inicio, p1_ffin=f_fin, p2_gran = "D",
                      p3_inst = instrumento, p4_oatk = token, p5_ginc = 4900)
df_pe = df_pe.set_index('TimeStamp') #set index to date

# Change everything from string to float
for col in df_pe:
    for i in range(len(df_pe[col])):
        df_pe[col][i] = float(df_pe[col][i])
#%% Add features        

def add_fracdiff_features(df, threshold=1e-4):
    '''
    Takes every column of a DataFrame, fractionally differentiates it to the
    least required order to make it stationary and joins them to the original
    DataFrame
    
    
    Parameters
    ----------
    df : pd.DataFrame
    threshold : float(), optional
        DESCRIPTION. The default is 1e-4.
        If length of df is small, use a bigger threshold, such as 1e-3

    Returns
    -------
    df : Same df as input but with every column duplicated and fractionally
         differentiated to the least required order to make it stationary

    '''
    for col in df.columns:
        _,series = least_diff(df[col], dRange = (0,1), step=0.1, 
                              threshold=threshold, confidence='1%') #threshold menor por ser una serie pequeña
        df[col+'fdiff'] = series
    return df

# Add fracdiff features
df_pe = add_fracdiff_features(df_pe, threshold = 1e-4)

## Technical Indicators

def CCI(data, ndays):
    '''
    Commodity Channel Index
    
    Parameters
    ----------
    data : pd.DataFrame with 3 colums named High, Low and Close
    ndays : int used for moving average and moving std

    Returns
    -------
    CCI : pd.Series containing the CCI
    '''
    TP = (data['High'] + data['Low'] + data['Close']) / 3 
    CCI = pd.Series((TP - TP.rolling(ndays).mean()) / 
                    (0.015 * TP.rolling(ndays).std()), name = 'CCI') 
    return CCI

df_pe['CCI'] = CCI(df_pe, 14) # Add CCI

def SMA(data, ndays): 
    '''Simple Moving Average'''
    SMA = pd.Series(data['Close'].rolling(ndays).mean(), name = 'SMA')
    return SMA

df_pe['SMA_5'] = SMA(df_pe, 5)
df_pe['SMA_10'] = SMA(df_pe, 10)
df_pe['MACD'] = df_pe['SMA_10']-df_pe['SMA_5']

def BBANDS(data, window):
    ''' Bollinger Bands '''
    MA = data.Close.rolling(window).mean()
    SD = data.Close.rolling(window).std()
    return MA + (2 * SD), MA - (2 * SD)

df_pe['Upper_BB'], df_pe['Lower_BB'] = BBANDS(df_pe, 10)
df_pe['Range_BB'] = (df_pe['Close']-df_pe['Lower_BB'])/(df_pe['Upper_BB']-df_pe['Lower_BB'])

def RSI(data, window):
    ''' Relative Strnegth Index'''
    delta = data['Close'].diff().dropna()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    roll_up1 = up.ewm(span=window).mean()
    roll_down1 = down.abs().ewm(span=window).mean()
    RS1 = roll_up1 / roll_down1
    return 100.0 - (100.0 / (1.0 + RS1))

df_pe['RSI'] = RSI(df_pe, 10)

def price_from_max(data, window):
    return data['Close']/data['Close'].rolling(window).max()

df_pe['Max_range'] = price_from_max(df_pe, 20)

def price_from_min(data, window):
    return data['Close']/data['Close'].rolling(window).min() - 1

df_pe['Min_range'] = price_from_min(df_pe, 20)

def price_range(data, window):
    pricerange = (data['Close'] - data['Close'].rolling(window).min()) / \
                 (data['Close'].rolling(window).max() - data['Close'].rolling(window).min())
    return pricerange

df_pe['Price_Range'] = price_range(df_pe, 50)

    
#%% Labeling: 1 for positive next day return, 0 for negative next day return
def next_day_ret(df):
    '''
    Given a DataFrame with one column named 'Close' label each row according to
    the next day's return. If it is positive, label is 1. If negative, label is 0
    Designed to label a dataset used to train a ML model for trading
    
    RETURNS
    next_day_ret: pd.DataFrame
    label: list
    
    Implementation on df_pe:
        _, label = next_day_ret(df_pe)
        df_pe['Label'] = label
    '''
    next_day_ret = df.Close.pct_change().shift(-1)
    label = []
    for i in range(len(next_day_ret)):
        if next_day_ret[i]>0:
            label.append(1)
        else:
            label.append(0)
    return next_day_ret, label

_, label = next_day_ret(df_pe)
df_pe['Label'] = label

#%%
#binary ,returns and accum returns

def ret_div(df):
    
    '''
    Return a logarithm and arithmetic daily returns
    and daily acum daily
    
    '''
    ret_ar = df.Close.pct_change().fillna(0)
    ret_ar_acum = ret_ar.cumsum()
    ret_log = np.log(1+ret_ar_acum).diff()
    ret_log_acum = ret_log.cumsum()
    
    binary = ret_ar
    binary[binary<0] = 0
    binary[binary>0] = 1
    return ret_ar  , ret_ar_acum , ret_log , ret_log_acum, binary

ra , racm , rl , rlacm, binary = ret_div(df_pe)
df_pe['returna'] , df_pe['returna_acums'] , df_pe['returnlog'] , df_pe['returnlog_acum'], df_pe['binary'] = ra , racm , rl , rlacm, binary

#zscore normalization

def z_score(df):
    #zscore 
    mean,std = df.Close.mean() , df.Close.std()
    zscore = (df.Close-mean)/std

    return zscore

df_pe['zscore'] = z_score(df_pe)

# diff integer
def int_diff(df, window: np.arange):
    diff = [
        df.Close.diff(x) for x in window
        ]
    return diff

df_pe['diff1'] , df_pe['diff2'] , df_pe['diff3'] , df_pe['diff4'] , df_pe['diff5'] =int_diff(df_pe,np.arange(1,6))

#moving averages
def mov_averages(df, space: np.arange):
    mov_av = [
        df.Close.rolling(w).mean() for w in space
        ]
    return mov_av

df_pe['mova1'] , df_pe['movaf2'] , df_pe['mova3'] , df_pe['mova4'] , df_pe['mova5'] =mov_averages(df_pe,np.arange(1,6))
    
def quartiles(df,n_bins:int):
    'Assign quartiles to data, depending of position'
    bin_fxn = pd.qcut(df.Close,q=n_bins,labels=range(1,n_bins+1))
    return bin_fxn
df_pe['quartiles'] = quartiles(df_pe,10)


#FUNCIONES de CHANGE POINT DETECTION
def zerolistmaker(n):
    list_zeros = [0] * n #Multiplica 0's por la dimensión 'n'.

    #Regresa una lista de zeros de dimensión n.
    return list_zeros

def boolean_change_point(data, changes):
    
    #Uso de la función de 'zerolistmaker'.
    zero = zerolistmaker(len(data)) #Crea una lista de zeros del tamaño de tus datos.

    change = [int(x) for x in changes] #Cuenta cuantos cambios haras dentro de tu lista.
    
    #For para cambiar los datos en donde haya un cambio.
    for i in range (0, len(change)):
        zero[change[i]] = 1
        
    #Regresa una lista en donde se encuentran los cambios como 1 y los no cambios como 0. 
    return zero


def window(datos):
    '''
    data: Valores del activo EURUSD.
    
    '''
    data = np.array(datos.Close) #De los datos del activo, selecciona la columna Close y la hace un array.
    
    n = len(data) #Tamaño de el array de datos.
    sigma = data.std() #Desviación estandar de los datos.
    p = np.log(n)*sigma**2 #Penalización que tiene el modelo.
    suma = []
    suma1 = []
    #Pasos a realizar para el metodo de window-based.
    for i in range (0,100):
        algo = rpt.Window(width=i+10).fit(data)
        my_bkps = algo.predict(pen=p)
        senal = pd.DataFrame(my_bkps)
        suma.append(my_bkps)
    suma = pd.DataFrame(suma)
    suma = suma.dropna()

    width = list(suma.index)
    width = width[0]

    for i in range (0,100):
        algo = rpt.Window(width=width, jump=i+1).fit(data)
        my_bkps = algo.predict(pen=p)
        senal = pd.DataFrame(my_bkps)
        suma.append(my_bkps)
    suma1 = pd.DataFrame(suma1)
    suma1 = suma.dropna()

    jump = list(suma1.index)
    jump = jump[0] 

    algo = rpt.Window(width=width, jump=jump).fit(data)
    my_bkps = algo.predict(pen=p)
    senal = pd.DataFrame(my_bkps)

    mean = senal.drop(len(my_bkps)-1) #Quitamos de la serie el último valor ya que no es correcto. 
    mean = np.array(mean) #Datos generados del metodo, traidos a un array.
    changes = mean.astype(int) #Hacer que el array contenga solo valores numericos enteros.

    fecha = [] #Lista vacia para introducir fechas donde el cambio ocurrio.
    #For para introducir los valores de la fechas en donde ocurrieron los changepoints.
    for i in range (0, len(my_bkps)-1):
        fecha += datos.index[changes[i]]

    #Esta variable sirve para crear el feature que se utilizará en el modelo.
    feature = boolean_change_point(data, changes)
    
    #La función regresa las fechas y los valores numericos en donde ocurrieron los cambios.
    return fecha, changes, feature

df_pe['Windows'] = window(df_pe)[2]

def binary(data):
    '''
    data: Valores del activo EURUSD.
    
    '''
    datos = np.array(data.Close)

    n = len(datos) #Tamaño de los datos dentro del array.
    sigma = datos.std() #Desviación estandar de los datos.
    p = np.log(n)*sigma**2 #Penalización utilizada dentro del modelo.

    #Pasos a realizar dentro del modelo de Binary segmentation.
    algo = rpt.Binseg().fit(datos)
    my_bkps = algo.predict(pen=p)
    senal = pd.DataFrame(my_bkps)

    mean = senal.drop([len(my_bkps)-1]) #Quitamos de la serie el último valor ya que no es correcto.
    mean = np.array(mean) #Valores obtenidos del modelo traidos a un array.

    changes = mean.astype(int) #Valores del array anterior convertidos a numeros enteros.

    fecha = [] #Lista vacia para introducir fechas deseadas.
    #For para introducir las fechas en donde ocurrio un cambio.
    for i in range (0, len(my_bkps)-1):
        fecha += data.index[changes[i]]
        
    feature = boolean_change_point(datos, changes)
    
    #La función regresa las fechas y los cambios numericos. 
    return fecha, changes, feature

df_pe['binary_c'] = binary(df_pe)[2]

def pelt(data):
    '''
    data: Valores del activo EURUSD.
    
    '''
    datos = np.array(data.Close)

    n = len(datos) #Tamaño de los datos dentro del array.
    sigma = datos.std() #Desviación estandar de los datos.
    p = np.log(n)*sigma**2 #Penalización utilizada dentro del modelo.

    #Pasos a realizar dentro del modelo de Binary segmentation.
    algo = rpt.Pelt().fit(datos)
    my_bkps = algo.predict(pen=p)
    senal = pd.DataFrame(my_bkps)

    mean = senal.drop([len(my_bkps)-1]) #Quitamos de la serie el último valor ya que no es correcto.
    mean = np.array(mean) #Valores obtenidos del modelo traidos a un array.

    changes = mean.astype(int) #Valores del array anterior convertidos a numeros enteros.

    fecha = [] #Lista vacia para introducir fechas deseadas.
    #For para introducir las fechas en donde ocurrio un cambio.
    for i in range (0, len(my_bkps)-1):
        fecha += data.index[changes[i]]
        
    feature = boolean_change_point(datos, changes)
    
    #La función regresa las fechas y los cambios numericos. 
    return fecha, changes, feature

df_pe['pelt'] = pelt(df_pe)[2]
