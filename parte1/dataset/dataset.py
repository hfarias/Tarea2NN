from keras.utils.np_utils import to_categorical
from keras.utils import np_utils
import pickle as cPickle
import numpy as np
import os
import random
from sklearn.preprocessing import StandardScaler

def load_CIFAR_one(filename):
    with open(filename, 'rb') as f:
        #datadict = cPickle.load(f, encoding='latin1')
        datadict = cPickle.load(f)
        X = datadict['data']
        Y = datadict['labels']
        #X = np.transpose(np.reshape(X,(-1,32,32,3), order='F'),
        #                axes=(0,2,1,3)) #order batch,x,y,color
        # test
        Y = np.array(Y)
        return X, Y

def load_CIFAR10(PATH):
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(PATH, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_one(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs) #entrenamiento
    Ytr = np.concatenate(ys) #entrenamiento
    del X, Y
    Xt, Yt = load_CIFAR_one(os.path.join(PATH, 'test_batch'))
    limit_sup= 6000
    limit_inf= 2000
    Xu= Xtr[limit_inf:limit_sup,0:3072] 
    Yu = Ytr[limit_inf:limit_sup] 
    return Xtr, Ytr, Xt, Yt, Xu, Yu

def escalar_centrar(X, with_mean=True, with_std=True):
    scaler = StandardScaler(with_mean, with_std).fit(X)
    return scaler.transform(X)

def leer(URL):
    Xtr, Ytr, Xt, Yt, Xv, Yv = load_CIFAR10(URL)
    #Xtr_cs = escalar_centrar(Xtr)
    #Xv_cs = escalar_centrar(Xv)
    #Xt_cs = escalar_centrar(Xt)
    #Ytr_conv = to_categorical(Ytr,10)
    #Yv_conv = to_categorical(Yv,10)
    #Yt_conv = to_categorical(Yt,10)
    Xtr =Xtr/255.0
    Xv  =Xv/255.0
    Xt  =Xt/255.0
    return Xtr,Ytr,Xt,Yt,Xv,Yv
