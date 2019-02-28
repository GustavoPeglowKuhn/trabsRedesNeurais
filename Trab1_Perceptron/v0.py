import csv
import random
import numpy as np

arqDados = '/sdcard/git/trabsRedesNeurais/Trab1_Perceptron/dados.csv'
arqTrein = '/sdcard/git/trabsRedesNeurais/Trab1_Perceptron/treinamento.csv'

dadosTreino = []
dadosTeste = []

def importCsv(treino, teste):
    csvReader = csv.reader(open(treino))
    for row in csvReader:
        dadosTreino.append(row)
    dadosTreino.pop(0)                            ##remove a 1st linha (cabecalho)
    
    csvReader = csv.reader(open(teste))
    for row in csvReader:
        dadosTeste.append(row)
    dadosTeste.pop(0)

def normaliza():
    maior = 0
    #for row in dadosTeste:
    for i in range(len(dadosTeste)):
        rowMax = max(dadosTeste[i][:2])
        if rowMax > maior:
            maior = rowMax
    
    return maior

class Neuronio():
    def __init__(self, teta, w):
        self.teta=teta
        self.w=w
    
    def out(self, x):
        res=0
        for i in range(0,2):
            res=res + self.w[i]*x[i]
        res=res - self.teta
        return self.G(res)
        
    def G(self, u):
        if u<0:
            return -1
        else:
            return 1

class Perceptron():
    def __init__(self):
        self.n = 0.01             #taxa de aprendizagem
        self.w = [random.randrange(0.1, 1, 0.1), random.randrange(0.1, 1, 0.1), random.randrange(0.1, 1, 0.1)]
        self.teta = -1            #random.randrange(0.1, 1, 0.1)
        self.neuronio = Neuronio(self.teta, self.w)
    
    def treino(dados):
        denovo = True
        count = 0
        
        while (denovo):
            res = self.neuronio.out(dadosTreino[count][:2])     #passa a primeira e a segunda coluna da linha count
            
            err = dadosTreino[count][2] - res
            
            

def main():
    importCsv('dados.csv', 'treinamento.csv')
    
    print(dadosTreino)
    print(dadosTeste)
    
    print("hello")












main()
