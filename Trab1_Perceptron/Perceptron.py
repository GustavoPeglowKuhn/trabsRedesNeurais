import numpy as np
import pandas as pd

treino = np.array(pd.read_excel('370772-dados_treinamento.xls'))
d = treino[:,3]
norma = np.amax(treino)
treino = np.dot(1/norma, treino)
x = np.ones([len(treino), 4])*-1
x[:, 1:4] = treino[:,0:3]

dTreino = d[ 0:25]
xTreino = x[ 0:25]
dTeste  = d[25:30]
xTeste  = x[25:30]

operacao  = np.array(pd.read_excel('370771-dados_operacao.xls'))
operacao = np.dot(1/norma, operacao)

xOperacao = np.ones([len(operacao), 4])*-1
xOperacao[:, 1:4] = operacao[:,0:3]

def G(u):
    if u<0:
        return -1
    else:
        return  1

def Treinar():
    w = np.random.random(4)*2-1
    n = 0.01
    epoca = 0
    
    erro = True
    while erro == True:
        erro = False
        epoca = epoca + 1
        for k in range(len(dTreino)):
            u = np.dot(w, xTreino[k])
            y = G(u)
            
            if y!=dTreino[k]:
                erro = True
                esc = n*(dTreino[k] - y)
                w = w + np.dot(esc, xTreino[k])
    return w, epoca

def Testar(w):
    nerros = 0
    res = np.zeros(len(dTeste))
    for k in range(len(dTeste)):
        u = np.dot(w.T, xTeste[k])
        y = G(u)
        res[k] = y
        
        if y!=dTeste[k]:
            nerros = nerros + 1
    return nerros, res

def Operacao(w):
    res = np.zeros(len(xOperacao))
    for k in range(len(xOperacao)):
        u = np.dot(w.T, xOperacao[k])
        y = G(u)
        res[k] = y
    return res


w, epocas = Treinar()

errTeste, resTeste = Testar(w)

print('\nteste: ('+str(errTeste)+' erros)')
print(resTeste)

resOp = Operacao(w)
print('\nOperacao:')
print(resOp)
