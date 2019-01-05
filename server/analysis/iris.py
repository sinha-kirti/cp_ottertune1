
# Ottertune - iris.py

"""
Created on : 25th December, 2018

This code is to test the implementation of self designed ML code in ottertune code

It is an implementation of iris classifier, in which we classify the 3 species of iris flower based on the petal length, width and sepal length and width

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
class Iris(object):
    def __init__(self):
        self.iris_data = datasets.load_iris()
        #print(self.iris_data)
    def fit(self):
        data2 = self.iris_data
        #del(data1['DESCR'])
        #del(data1['filename'])
        #del(data1['feature_names'])
        data1 = {}
        data1['data'] = data2['data']
        data1['target'] = data2['target']
        #Species = data1['target_names']
        #del(data1['target_names'])
        petal_length = data1['data'][:,0]
        petal_width = data1['data'][:,1]
        sepal_length = data1['data'][:,2]
        sepal_width = data1['data'][:,3]
        del(data1['data'])
        data1['petal_length'] = petal_length
        data1['petal_width'] = petal_width
        data1['sepal_length'] = sepal_length
        data1['sepal_width'] = sepal_width
        data = pd.DataFrame.from_dict(data1)
        data.loc[data['target'] == 0,'target'] = data['target'].apply(lambda x: 'Iris_Setosa')
        data.loc[data['target'] == 1,'target'] = data['target'].apply(lambda x: 'Iris_Versicolor')
        data.loc[data['target'] == 2,'target'] = data['target'].apply(lambda x: 'Iris_Virginica')
        i = 8
        data_to_predict = data[:i].reset_index(drop = True)
        predict_species = data_to_predict.target
        predict_species = np.array(predict_species)
        prediction = np.array(data_to_predict.drop(['target'],axis= 1))
        data = data[i:].reset_index(drop = True)
        X = data.drop(['target'], axis = 1)
        X = np.array(X)
        Y = data['target']
        # Transform name species into numerical values 
        encoder = LabelEncoder()
        encoder.fit(Y)
        Y = encoder.transform(Y)
        Y = np_utils.to_categorical(Y)
        train_x, test_x, train_y, test_y = model_selection.train_test_split(X,Y,test_size = 0.1, random_state = 0)
        input_dim = len(data.columns) - 1
        model = Sequential()
        model.add(Dense(8, input_dim = input_dim , activation = 'relu'))
        model.add(Dense(10, activation = 'relu'))
        model.add(Dense(10, activation = 'relu'))
        model.add(Dense(10, activation = 'relu'))
        model.add(Dense(3, activation = 'softmax'))
        
        model.compile(loss = 'categorical_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'] )
        model.fit(train_x, train_y, epochs = 10, batch_size = 2)
        scores = model.evaluate(test_x, test_y)
        print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        predictions = model.predict_classes(prediction)
        prediction_ = np.argmax(to_categorical(predictions), axis = 1)
        prediction_ = encoder.inverse_transform(prediction_)
        for i, j in zip(prediction_ , predict_species):
            print( " the nn predict {}, and the species to find is {}".format(i,j))

