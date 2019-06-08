from sklearn import preprocessing

from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_mldata

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from PIL import Image
def gen_image(arr):					#gera imagem da matriz ja em escala 0-255
    two_d = (np.reshape(arr, (28, 28))).astype(np.uint8)
    img = Image.fromarray(two_d, 'L')
    return img
def gen_image_normalized(arr):		#gera umagem de matriz normalizada, escala 0-1
    two_d = (np.reshape(arr, (28, 28)) * 255).astype(np.uint8)
    img = Image.fromarray(two_d, 'L')
    return img
#gen_image(np.array(x_test)[0])
#im = gen_image(np.array(x_test)[0])
#im.save('temp.png')

def teste(arr, y):
	gen_image(arr)
	yp = mlp.predict(arr)
	print()
	


df = pd.read_csv('mnist_100.csv')

y = df['label']
x = df.drop(['label'], axis=1)

#x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.25, random_state=27)

x = np.array(x)
y = np.array(y)





x_train, x_test, y_train, y_test


mlp = MLPClassifier(hidden_layer_sizes=(128, 64, 10), max_iter=10000, alpha=1e-4,
                    solver='sgd', verbose=True, tol=1e-6, random_state=1,
                    learning_rate_init=.1, activation='logistic')

mlp.fit(x_train, y_train)

y_predict = mlp.predict(x_test)

res = np.array([np.array(y_test), np.array(y_predict)]).T