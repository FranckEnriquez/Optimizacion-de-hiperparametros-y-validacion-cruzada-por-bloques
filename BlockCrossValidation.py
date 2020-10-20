import pandas as pd
from Data import getPrices
import numpy as np

def createBlocks(data,chunkGran):
    """

    :param data: pandas  DataFrame de precios de cierre
    :param chunkGran: periodo para hacer particiiones, M,W,D,H,,MT
    :param to_choose: chunk a elergir
    """
    unique_dates=list()
    # if chunkGran =="M":
    #     dates=(str(data["TimeStamp"][i])[5:7] for i in range(len(data)))
    #     unique_dates = pd.unique(dates)
    # elif chunkGran =='D':
    #     dates = (str(data["TimeStamp"][i])[8:10] for i in range(len(data)))
    # elif chunkGran=='W':
    #
    chunks = list()
    chunk_list=list()
    if chunkGran=='M':
        for i in range(len(data)):
            if i != 0:
                if str(data["TimeStamp"][i])[5:7]==str(data["TimeStamp"][i-1])[5:7]:
                    chunk_list.append(data.iloc[i])

                else :
                    chunks.append(chunk_list)
                    chunk_list=list()
                    chunk_list.append(data.iloc[i])
    return chunks

def splitBlock(block):
    """

    :param block: block of data containing both features and response
    :return: train and test sets for both features and response
    """
    x=block.loc[:,block.columns!='Tag']
    y=block["Tag"]

    x_train=x[0:np.round(3*(len(x)/4))]
    x_test = x[np.round(3*(len(x)/4)):]
    y_train = y[0:np.round(3 * (len(y) / 4))]
    y_test = y[np.round(3 * (len(y) / 4)):]

    return [x_train,x_test,y_train,y_test]






