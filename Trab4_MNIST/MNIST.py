#from sklearn import preprocessing

from sklearn.neural_network import MLPClassifier
#from sklearn.datasets import fetch_mldata

#from sklearn.model_selection import train_test_split
#from sklearn.metrics import accuracy_score
#from sklearn.metrics import confusion_matrix

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from PIL import Image
def gen_image(arr):					#gera imagem da matriz ja em escala 0-255
    two_d = 255-(np.reshape(arr, (28, 28))).astype(np.uint8)
    img = Image.fromarray(two_d, 'L')
    return img
def gen_image_normalized(arr):		#gera umagem de matriz normalizada, escala 0-1
    two_d = 255-(np.reshape(arr, (28, 28)) * 255).astype(np.uint8)
    img = Image.fromarray(two_d, 'L')
    return img
#gen_image(np.array(x_test)[0])
#im = gen_image(np.array(x_test)[0])
#im.save('temp.png')

def teste(mlp, arr, y):
	yp = mlp.predict(np.reshape(arr, (1, 784)))
	print('real: '+str(y)+'    obtido: '+str(yp[0]))
	plt.imshow(gen_image(arr))
	

#x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.25, random_state=27)
def import_file(file_name):		#return x and y in np.array format
	df = pd.read_csv(file_name)
	y = df['label']
	x = df.drop(['label'], axis=1)
	return np.array(x), np.array(y)

x_train, y_train = import_file('mnist_100.csv')
x_test,  y_test  = import_file('mnist_10000.csv')


topologias = []	##criar 6 topologias diferentes
topologias.append(MLPClassifier(hidden_layer_sizes=(128, 64, 10), max_iter=10000, alpha=1e-4,
								solver='sgd', verbose=True, tol=1e-6, random_state=1,
								learning_rate_init=.1, activation='logistic'))

topologias.append(MLPClassifier(hidden_layer_sizes=(50, 10), max_iter=10000, alpha=1e-4,
								solver='sgd', verbose=True, tol=1e-6, random_state=1,
								learning_rate_init=.1, activation='logistic'))

#res = np.array([np.array(y_test), np.array(y_predict)]).T
#mlp.score(x_test, y_test)

scores = []

for i in range(len(topologias)):
	topologias[i].fit(x_train, y_train)
	scores.append( topologias[i].score(x_test, y_test) )

print(scores)
index = np.argmax(scores)	#pega o indice da melhor topologia

print('\n\n\n\nImporting 60000 samples!\n\n\n')
x_train, y_train = import_file('mnist_60000.csv')	#importa os dados do vetor de 60k amostras

mlp = topologias[index]
del topologias

print('\n\n\n\nO treino final!\n\nSenta e espera, pois vai demorar\n\n\n')
mlp.fit(x_train, y_train)
print('score = '+str(100*mlp.score(x_test, y_test))+'% de acerto')

print('\n\n\n\nMatriz de confisao\n\n\n')
print(pd.crosstab(y_test, mlp.predict(x_test)))
