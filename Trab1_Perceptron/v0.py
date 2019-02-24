import csv
 
x0 = []
x1 = []
x2 = []

with open('/sdcard/git/trabsRedesNeurais/Trab1_Perceptron/dados.csv') as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    for row in csvReader:
        x0.append(row[0])
        x1.append(row[1])
        x2.append(row[2])

#print(x0)
#print(x1)
#print(x2)

