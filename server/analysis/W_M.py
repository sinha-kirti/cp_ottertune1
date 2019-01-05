#W_M.py

"""
Created on : 5th January, 2019

This code is to test the implementation of self designed ML code in ottertune code

It is an implementation of simple 5 layer classifier based on earlier iris classifier code

"""
import numpy as np
import scipy as sp
import  pandas as pd
from sklearn.preprocessing import StandardScaler
import sklearn
import random 
import time 

from sklearn import preprocessing, model_selection

from keras.models import Sequential 
from keras.layers import Dense 
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
from sklearn.utils import shuffle

from sklearn import datasets
# Initializing the class and data
class W_M(object):
    def __init__(self):
        model = Sequential()
        input_dim = 14
        model.add(Dense(8, input_dim = input_dim , activation = 'relu'))
        model.add(Dense(10, activation = 'relu'))
        model.add(Dense(10, activation = 'relu'))
        model.add(Dense(10, activation = 'relu'))
        model.add(Dense(5, activation = 'softmax'))
        model.compile(loss = 'categorical_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'] )
        self.model = model
        #self.iris_data = datasets.load_iris()
        #print(self.iris_data)
    def fit(self, Zs, labels):
        z = np_utils.to_categorical(labels)
        #print(z)
        #train_x, test_x, train_y, test_y = model_selection.train_test_split(X,Y,test_size = 0.1, random_state = 0)
        train_xz, test_xz, train_yz, test_yz = model_selection.train_test_split(Zs,z,test_size = 0.1, random_state = 0)
        print(train_xz)
        print(test_xz)
        self.model.fit(train_xz, train_yz, epochs = 30, batch_size = 9)
        scores = self.model.evaluate(test_xz, test_yz)
        print("\n%s: %.2f%%" % (self.model.metrics_names[1], scores[1]*100))
        #prediction_ = np.argmax(to_categorical(predictions), axis = 1)
        #prediction_ = encoder.inverse_transform(prediction_)
        #for i, j in zip(prediction_ , predict_species):
            #print( " the nn predict {}, and the species to find is {}".format(i,j))
    
    def predict(self,test_data):
        a = np.array(test_data)
        print(a.shape)
        predicted_label = self.model.predict_classes(a)
        return predicted_label   


