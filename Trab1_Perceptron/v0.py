import csv
import random
#import numpy as np

arqDados = '/sdcard/git/trabsRedesNeurais/Trab1_Perceptron/dados.csv'
arqTrein = '/sdcard/git/trabsRedesNeurais/Trab1_Perceptron/treinamento.csv'

dadosTreino = []
dadosTeste = []

def EscVec(vec, escalar):
	for i in range(len(vec)):
		vec[i] = vec[i] * escalar
	return vec

def EscMat(mat, escalar):
	for i in range(len(mat)):
		for j in range(len(mat[i])):
			mat[i][j] = mat[i][j] * escalar
	return mat

def Norma(mat):
	maior = mat[0][0]
	for i in range(len(mat)):
		for j in range(len(mat[i])):
			if mat[i][j] > maior:
				maior = mat[i][j]
	return maior

def ImportCsv(arq):
	dados = []
	csvReader = csv.reader(open(arq))
	for row in csvReader:
		linha = []
		for val in row:
			linha.append( float(val) )
		dados.append(linha)
	#dados.pop(0)                            ##remove a 1st linha (cabecalho)
	return dados

class Neuronio():
	def __init__(self, teta, w):
		self.teta=teta
		self.w=w
	
	def out(self, x):
		#res = self.w[0] * x[0] + self.w[1] * x[1] + self.w[2] * x[2] - self.teta
		#return self.G(res)
		return -1.0
	
	def G(self, u):
		if u<0:
			return -1
		else:
			return 1

class Perceptron():
	def __init__(self, dTreino, result, dTest):
		self.n = 0.1             #taxa de aprendizagem
		self.neuronio = Neuronio(-1, [random.random(), random.random(), random.random()])
		
		self.teino = dTreino
		self.result = result
		self.test = dTest
    
	def Treinar(self):
		denovo = True
		count = 0
		index = 0
		
		nErr = 0
		
		while (denovo):
			res = self.neuronio.out(self.teino[index])
			
			err = self.result[index] - res
			
			if err != 0:
				self.neuronio.w[0] = self.neuronio.w[0] + self.n * err * self.teino[index][0]
				self.neuronio.w[1] = self.neuronio.w[1] + self.n * err * self.teino[index][1]
				
				nErr = 0
			else:
				#denovo = False
				#return count
				
				nErr = nErr + 1
				#if nErr == len(self.teino):
				if nErr == 5:
					denovo = False
					return count
			
			count = count + 1
			index = index + 1
			
			if index == len(self.teino):
				index = 0

	def Testar(self):
		self.acertos = 0
		
		for i in range(len(dadosTeste)):
			if self.neuronio.out(self.test[i]) == self.result:
				self.acertos = self.acertos + 1
		
		return self.acertos

def main():
	dadosTreino = ImportCsv('treinamentoNC.csv')
	dadosTeste = ImportCsv('dadosNC.csv')
	#dadosTreino = ImportCsv(arqDados)
	#dadosTeste = ImportCsv(arqTrein)
	
	norma = Norma(dadosTreino)
	
	treino = []
	result = []
	for i in range(len(dadosTreino)):
		x1, x2, x3 = dadosTreino[i][0], dadosTreino[i][1], dadosTreino[i][2]
		row = [x1, x2, x3]
		treino.append(row)
		result.append(dadosTreino[i][3])
	
	treinoNor = EscMat(treino, 1/norma)
	testeNor = EscMat(dadosTeste, 1/norma)
	
	perceptron = Perceptron(treinoNor, result, testeNor)
	
	print ("ciclos de treino:")
	print (perceptron.Treinar())
	
	print("\n\nn acertos:")
	print(perceptron.Testar())



















dadosTreino = ImportCsv('treinamentoNC.csv')
dadosTeste = ImportCsv('dadosNC.csv')

main()
