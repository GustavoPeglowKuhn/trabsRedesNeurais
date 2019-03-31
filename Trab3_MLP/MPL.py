import pandas as pd
import numpy as np
import math

n = 0.1
eps = 1e-6


nIn  = 3
nOut = 1
camadas = 2         #1 ocultas e 1 de saida
nNeuronios = [10, nOut]



dados = np.array(pd.read_excel('373922-Teste_projeto_1_MLP.xls'))
nAmostras = len(dados)
d = dados[:, 3]
x = np.ones([len(dados), len(dados[0])])
x[:, 0] = x[:, 0]*-1
x[:, 1:] = dados[:, :3]



def g(x):
    return math.tanh(x)

def dg(x):
    return 1-(math.tanh(x)**2)

def Y(_w, _x, nI):
    _l = np.zeros(nI)
    _l = np.dot(_w, _x)
    _y = np.zeros(nI+1)
    _y[0] = -1
    for i in range(len(_l)):
        _y[i+1] = math.tanh(_l[i])
    return _l, _y

############################### treino #
w = [0, np.random.random([nNeuronios[0], nIn+1])*2-1, np.random.random([nOut, nNeuronios[0]+1])*2-1]
l = [0,               np.zeros(nNeuronios[0]),   np.zeros(nNeuronios[1])]
y = [np.zeros(nIn+1), np.zeros(nNeuronios[0]+1), np.zeros(nNeuronios[1]+1)]

wList = []
wList.append(w.copy())

############################## Laço ##
iteracao = 0

############################## Forward ###
y[0] = x[iteracao]

l[1], y[1] = Y(w[1], y[0], len(l[1]))
l[2], y[2] = Y(w[2], y[1], len(l[2]))
#y[2][1:]

############################## Backward ###
sigma2 = (d[iteracao]-y[2][1:]) * dg(l[2])
wAnt = w.copy()
w[2] = wAnt[2] + n*(sigma2*y[2][1:])



















































