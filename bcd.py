# -*- coding: utf-8 -*-
import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from bottle import route, run, template

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def saida(parametros):
    #Transformação do parametro em vetor
    entrada = parametros.split(",")

    #Transformação do csv em um dataset de 9 colunas (a última coluna é o resultados da predição)
    dataframe = pandas.read_csv("data-set.csv", header=None)
    dataset = dataframe.values
    X = dataset[:, 0:4].astype(float)
    Y = dataset[:, 4]
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    dummy_y = np_utils.to_categorical(encoded_Y)

    #Treinamento e classificação dos dados pelo algoritmo knn
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X, Y)
    KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                         metric_params=None, n_neighbors=1, p=2, weights='uniform')
    test_input = [entrada]
    classe = knn.predict(test_input)
    return classe


@route('/bcd/<param>')
def index(param):
    classe = saida(param)
    return template('{{classe}}', classe=classe)


# run(host='0.0.0.0', port=9090)
run(host='0.0.0.0', port=8080)
#Exemplo para acessar no browser http://localhost:8080/bcd/7,4,6,4,6,1,4,3,1