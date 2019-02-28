import csv

arqDados = '/sdcard/git/trabsRedesNeurais/Trab1_Perceptron/dados.csv'
arqTrein = '/sdcard/git/trabsRedesNeurais/Trab1_Perceptron/treinamento.csv'
 
x0 = []
x1 = []
x2 = []

with open(arqTrein) as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    for row in csvReader:
        x0.append(row[0])
        x1.append(row[1])
        x2.append(row[2])

#print(x0)
#print(x1)
#print(x2)

class Neuronio():
    def __init__(self, g, teta, w):
        self.g=g
        self.teta=teta
        self.w=w
    
    def out(self, x):
        res=0
        for i in range(0,2):
            res=res + self.w[i]*x[i]
        res=res - self.teta
        return G(res)
    
    def G(self, u):
        if u<0:
            return -1
        else:
            return 1
            
class rede():
    

def main():
    print "hello"












main()
