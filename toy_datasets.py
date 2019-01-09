import numpy as np
import pandas as pd
import sklearn.model_selection
import tensorflow as tf

TOY_DATASET_PATH = "./datasets/toy"

def getDataframe(ds_name):
    path_X = TOY_DATASET_PATH + "/" + ds_name + "_X.csv"
    path_Y = TOY_DATASET_PATH + "/" + ds_name + "_Y.csv"
    
    X = pd.read_csv(path_X,sep=",",index_col=0,header=0)
    Y = pd.read_csv(path_Y,sep=",",index_col=0,header=0)
    return {"X":X.values,"Y":np.reshape(Y.values,(-1))}

def getTFDataset(ds_name):
    df = getDataframe(ds_name)
    return (tf.data.Dataset.from_tensor_slices({"X":df["X"],\
                 "Y":tf.one_hot((-1) + tf.cast(df["Y"],dtype=tf.int32),
                                np.max(df["Y"]))}))


    


