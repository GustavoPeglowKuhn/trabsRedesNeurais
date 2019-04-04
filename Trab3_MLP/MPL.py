import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

n = 0.1
eps = 1e-6


nIn  = 3
nOut = 1
camadas = 2         #1 ocultas e 1 de saida
nNeuronios = [10, nOut]



dados = np.array(pd.read_excel('373925-Treinamento_projeto_1_MLP.xls'))
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
        _y[i+1] = g(_l[i])
    return _l, _y

def Em(_d, _y, _n):
	res = 0
	for i in range(_n):
		res = res + ((_d[i]-_y[i])**2)
	#res = res = ((_d[:]-_y[:])**2).sum()
	return .5*res/_n

############################### treino #
w = [0, np.random.random([nNeuronios[0], nIn+1])*2-1, np.random.random([nOut, nNeuronios[0]+1])*2-1]
l = [0,               np.zeros(nNeuronios[0]),   np.zeros(nNeuronios[1])]
y = [np.zeros(nIn+1), np.zeros(nNeuronios[0]+1), np.zeros(nNeuronios[1]+1)]

w0 = w.copy()

saidas = np.zeros(nAmostras)

epocas = 0
continuar = True

E = Eant = 0
Elist = []

while continuar:
	Eant = E
	for amostra in range(nAmostras):
		############################## Forward ###
		y[0] = x[amostra]

		l[1], y[1] = Y(w[1], y[0], len(l[1]))
		l[2], y[2] = Y(w[2], y[1], len(l[2]))
		saidas[amostra] = y[2][1:]

		############################## Backward ###
		wAnt = w.copy()
        
		sigma2 = (d[amostra]-y[2][1]) * dg(l[2])
		w[2][0,:] = wAnt[2][0,:] + n*sigma2*y[1][:]

		#de camada 2 para a 1

		dg1 = np.zeros(nNeuronios[0])
		for i in range(nNeuronios[0]):
			dg1 = dg(l[1][i])
		
		sigma1 = -w[2][0,1:]*sigma2
		
		for i in range(nNeuronios[0]):
			w[1][i,:] = wAnt[1][i,:] + sigma1[i]*n*x[amostra,:]
	
	E = Em(d, saidas, nAmostras)
	Elist.append(E)

	epocas = epocas+1
	continuar = abs(Eant-E)>eps


###############################################################################
############################## Teste #

dadosTeste = np.array(pd.read_excel('373922-Teste_projeto_1_MLP.xls'))
nTeste = len(dadosTeste)
dt = dadosTeste[:, 3]
xt = np.ones([len(dadosTeste), len(dadosTeste[0])])
xt[:, 0] = xt[:, 0]*-1
xt[:, 1:] = dadosTeste[:, :3]

lt = [0,               np.zeros(nNeuronios[0]),   np.zeros(nNeuronios[1])]
yt = [np.zeros(nIn+1), np.zeros(nNeuronios[0]+1), np.zeros(nNeuronios[1]+1)]

res = [dt, np.zeros(nTeste)]

for amostra in range(nTeste):
	yt[0] = x[amostra]

	lt[1], yt[1] = Y(w[1], yt[0], len(lt[1]))
	lt[2], yt[2] = Y(w[2], yt[1], len(lt[2]))
	
	res[1][amostra] = yt[2][1]
	


print(np.matrix(res))

plt.plot(Elist)
plt.ylabel('Elist')
plt.show()









































