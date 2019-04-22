import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


print('###########################################################################')
print('###########################################################################')
print('#                                                                         #')
print('#                            Apenas rascunho                              #')
print('#                                                                         #')
print('###########################################################################')
print('###########################################################################')


def importData(file):
	x = np.array(pd.read_excel(file))
	return x, len(x)

###leituraa dos dados do ex3
xTreino, sTreino = importData('373928-Treinamento_projeto_3_MLP.xls')
xTeste,  sTeste  = importData('373924-Teste_projeto_3_MLP.xls')

###caracteristicas da rede
aprendizado = 0.1
momentum = 0.8
Emax = 0.5e-6
epocasMax = 500		#limite de epocas para o treino

nIn, nOut =  5, 1
camadas = 2         #1 ocultas e 1 de saida
n = [10, nOut]

def g(x):
	return 1. / (1. + np.exp(-x))	

def dg(x):
	return g(x)*(1.-g(x))


x, size = xTreino, sTreino
############################################################
############  Treino
w = [np.random.random([n[0], nIn+1])*2-1, np.random.random([n[1], n[0]+1])*2-1]
w0 = w.copy()
wn = w.copy()
wa1 = w.copy()
wa2 = w.copy()
l = [np.zeros(n[0]),   np.zeros(n[1])]
y = [np.zeros(n[0]+1), np.zeros(n[1])]

epocas = 0

E = 0
Eant = 1
Elist = []

s = [np.ones((n[0])), np.ones((n[1]))]

while (abs(Eant-E)>Emax and epocas < epocasMax):
	Eant = E
	E = 0


	for i in range(size-nIn):	
		wa2 = wa1.copy()
		wa1 =   w.copy()
		
		############################## Forward ###
		l[0] = np.dot(w[0], x[i:i+nIn])
		for j in range(n[0]): 
			y[0][j] = g(l[0][j])
		y[0][n[0]] = -1
		l[1] = np.dot(w[1], y[0])
		for j in range(n[1]): 
			y[1][j] = g(l[1][j])

		############################## Backward ###
		for j in range(n[1]):
			s[1][j] = (d[i][j] - y[1][j]) * dg(l[1][j])
			wn[1][j] = w[1][j] + (aprendizado * s[1][j] * y[0][j])

		for j in range(n[0]):
			s[0][j] = np.dot(s[1], w[1][:, j]) * dg(l[0][j])
			wn[0][j] = w[0][j] + (aprendizado * s[0][j] * x[i])
		
		w[0] = wn[0] + momentum * (wa1[0] - wa2[0])
		w[1] = wn[1] + momentum * (wa1[1] - wa2[1])
		#w = wn
	
	for i in range(size):
		############################## Forward ###
		l[0] = np.dot(w[0], x[i])
		for j in range(n[0]): 
			y[0][j] = g(l[0][j])
		y[0][n[0]] = -1
		l[1] = np.dot(w[1], y[0])
		for j in range(n[1]): 
			y[1][j] = g(l[1][j])
			
		er = 0
		for j in range(nOut):
			er = er + ((d[i][j] - y[1][j])**2)
		E = E + 0.5*er
	E = E/size
	Elist.append(E)
	epocas = epocas +1


############################################################
############  Teste
x, size = xTeste,  sTeste

res = np.zeros([size, 2*nOut])
#res[:, :nOut] = d
E = 0
l = [np.zeros(n[0]),   np.zeros(n[1])]
y = [np.zeros(n[0]+1), np.zeros(n[1])]
for i in range(size):
	############################## Forward ###
	l[0] = np.dot(w[0], x[i])
	for j in range(n[0]): 
		y[0][j] = g(l[0][j])
	y[0][n[0]] = -1
	l[1] = np.dot(w[1], y[0])
	for j in range(n[1]): 
		y[1][j] = g(l[1][j])
	for j in range(n[1]):
		if y[1][j] < 0.5:
			res[i, nOut+j] = 0
		else:
			res[i, nOut+j] = 1
	#res[i, nOut:] = y[1]
	er = 0
	for j in range(n[1]):
		er = er + ((d[i][j] - y[1][j])**2)
		E = E + 0.5*er 
	E = E + 0.5*er
E = E/size


print(np.matrix(res))

print('Eqm: '+str(E)+'    epocas: '+str(epocas))

plt.plot(Elist)
plt.ylabel('Elist')
plt.show()
