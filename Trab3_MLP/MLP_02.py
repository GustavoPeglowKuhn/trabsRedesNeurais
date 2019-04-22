import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def importData(file, _nIn, _nOut):
	dados = np.array(pd.read_excel(file))
	d = dados[:, _nIn:].copy()
	x = dados[:, :_nIn+1].copy()
	x[:, _nIn:] = np.ones([len(dados),1])*-1
	return x, d, len(dados), _nIn, _nOut

###leituraa dos dados do ex1
#xTreino, dTreino, sTreino, nIn, nOut = importData('373925-Treinamento_projeto_1_MLP.xls', 3, 1)
#xTeste,  dTeste,  sTeste,  nIn, nOut = importData('373922-Teste_projeto_1_MLP.xls', 3, 1)

###leituraa dos dados do ex2
xTreino, dTreino, sTreino, nIn, nOut = importData('373926-Treinamento_projeto_2_MLP.xls', 4, 3)
xTeste,  dTeste,  sTeste,  nIn, nOut  = importData('373923-Teste_projeto_2_MLP.xls', 4, 3)

###caracteristicas da rede
aprendizado = 0.1
momentum = 0.9
Emax = 1e-6
epocasMax = 500		#limite de epocas para o treino

camadas = 2         #1 ocultas e 1 de saida
n = [15, nOut]


def g(x):
	return 1. / (1. + np.exp(-x))	

def dg(x):
	return g(x)*(1.-g(x))


x, d, size = xTreino, dTreino, sTreino
############################################################
############  Treino
w = [np.random.random([n[0], nIn+1])*2-1, np.random.random([n[1], n[0]+1])*2-1]
w0 = w.copy()
wn = w
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

	for i in range(size):
		############################## Forward ###
		l[0] = np.dot(w[0], x[i])
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

		w = wn
	
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
x, d, size = xTeste,  dTeste,  sTeste

res = np.zeros([size, 2*nOut])
res[:, :nOut] = d
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
	res[i, nOut:] = y[1]
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


'''res = np.zeros([2, size])
res[0] = d.T
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
	y[1][0] = g(l[1][0])
	res[1, i] = y[1]
	er = 0
	for j in range(n[1]):
		er = er + ((d[i][j] - y[1][j])**2)
		E = E + 0.5*er 
	E = E + 0.5*er
E = E/size'''