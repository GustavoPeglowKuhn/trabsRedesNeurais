import numpy as np
import pandas as pd


##################### Leitura dos dados de Treino e Teste #####################
df = pd.read_excel('./371889-tarefa02_treino.xls')                  #le o arquivo xls
treino = np.array(df)                                               #cria uma array com os dados lidos do excel

d = treino[:,4]                                                     #copia a terceira coluna de 'treino' para 'd'
norma = np.amax(treino)                                             #pega o maior valor de 'treino'
treino = np.dot(1/norma, treino)                                    #divive o vetor 'treino' pelo seu maior valor, deixando seus valores entre 0 e 1

linhas = len(treino)                                                #numero de linhas no vetor 'treino'
x = np.ones([linhas, 5])                                            #cria uma martiz de uns com tamanho 30x4
x = x * -1                                                          #multiplica a matriz 'x' por -1 (deixa todos os valores iguais a -1)
x[:, 1:5] = treino[:,0:4]                                           #copia para as 3 ultimas colunas de 'x' as 3 primeiras de 'treino'

dTreino = d[ 0:25]                                                  
xTreino = x[ 0:25]                                                  #separa 70% para treino (70% de 30 = 21)
dTeste  = d[25:30]
xTeste  = x[25:30]                                                  #o restante (30%) para teste


##################### Leitura dos dados de Operacao ###########################
df = pd.read_excel('./371890-tarefa02-operacao.xls')                     #le os dados de operacao
operacao  = np.array(df)                                            #cria uma array com os dados lidos do excel
operacao = np.dot(1/norma, operacao)                                #divide pela mesmo valor que os dados de trino foram divididos

linhas = len(operacao)                                              #numero de linhas no vetor 'operacao'
xOperacao = np.ones([linhas, 5])                                    #cria uma martiz de uns com tamanho 10x4
xOperacao = xOperacao * -1                                          #multiplica xOperacao por -1
xOperacao[:, 1:5] = operacao[:,0:4]                                 #


def G(u):
    if u<0:
        return -1
    else:
        return  1

def Eqw(w):
    return 1

def Treinar():                                                      #treina o Perceptron e retorna os pesos e n de epocas
    w = np.random.random(5)*2-1
    n = 0.0025
    eps = 1e-6
    
    epoca = 0
    
    Eant = 0
    Eatual = 0
    
    erro = True
    while erro == True:
        #erro = False
        epoca = epoca + 1
        
        Eant = Eatual
        
        er = 0
        
        for k in range(len(dTreino)):                               #executa 1 vez o treino em todos valores de treino
            u = np.dot(w.T, xTreino[k])                             # u = w0*x0 + w1*x1 + w2*x2 + w3*x3
            
            #w0 = w0 + n*(d0 - y)*x0; w1 = w1 + n*(d1 - y)*x1 ... 
            var = n*(dTreino[k] - u)                                #fator de correcao dos pesos
            w = w + np.dot(var, xTreino[k])                         #multiplica o vetor 'x' por 'var' e soma no vetor 'w'
            
            er = er + ((dTreino[k] - u)**2)
        Eatual = er/len(dTreino)
        
        dE = Eatual - Eant
        if dE < 0:
            dE = -dE                                                #modulo
            
        if dE > eps:
            erro = True
        else:
            erro = False
        
    return w, epoca                                                 #retorna o vetor dos pesos e o numero de epocas de treino


def Testar(w):                                                      #testa o Perceptron com os pesos recebidos
    nerros = 0                                                      #numero de erros
    res = np.zeros(len(dTeste))                                     #cria um vetor para salvar as respostas
    for k in range(len(dTeste)):                                    #executa 1 vez o treino em todos valores de teste
        u = np.dot(w.T, xTeste[k])                                  # u = w0*x0 + w1*x1 + w2*x2 + w3*x3
        y = G(u)
        res[k] = y
        
        if y!=dTeste[k]:
            nerros = nerros + 1
    return nerros, res                                              #retorna o n de erros e os 


def Operacao(w):                                                    #calcula o resultado para os dados de operacao
    res = np.zeros(len(xOperacao))                                  #cria um vetor para salvar as respostas
    for k in range(len(xOperacao)):
        u = np.dot(w.T, xOperacao[k])
        y = G(u)
        res[k] = y
    return res                                                      #retorna os resultados


w, epocas = Treinar()               #treina
errTeste, resTeste = Testar(w)      #testa
resOp = Operacao(w)                 #opera


print('\nnao esta pronto!!!!!\n\n\n')


#imprime os resultados
print('\nteste: ('+str(errTeste)+' erros)')
print('obtido:   '+str(resTeste))
print('desejado: '+str(dTeste))
print('\nOperacao: '+str(resOp))
