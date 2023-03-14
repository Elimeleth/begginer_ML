import nltk
nltk.download("punkt")

import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
import tensorflow as tf
import random
import pickle
import json
import sys
from nltk.stem.lancaster import LancasterStemmer

with open("intents.json") as file:
	data = json.load(file)
     


stemmer = LancasterStemmer() # stematizamos las palabras ej. loves => love

# FEATURES
words = [] # lista de palabras
labels = [] # lista de etiquetas o targets
X = [] # lista de entradas en formato np.array
Y = [] # lista de etiquetas o targets en formato np.array
     
# recorremos el fichero de entrenamiento y leemos los intentos junto a sus patrones
for intent in data["intents"]:
    for pattern in intent["patterns"]:
      words_aux = nltk.word_tokenize(pattern) # leemos las palabras y tokenizamos ['el', 'la', 'la', 'la']
      words.extend(words_aux) # añadimos las palabras a la lista de palabras
      X.append(words_aux) # añadimos las palabras a la lista de entradas
      Y.append(intent["tag"]) # añadimos las etiquetas a la lista de etiquetas

    # si la etiqueta del intento no esta en las etiquetas la agregamos
    if intent["tag"] not in labels:
      labels.append(intent["tag"])


words = [stemmer.stem(w.lower()) for w in words if w != '?'] # elimino las palabras que no sean letras y las stemming
words = sorted(list(set(words))) # ordeno las palabras

labels = sorted(labels) # ordeno las etiquetas

training = [] # lista de entradas en formato np.array para la red
output = [] # lista de etiquetas o targets en formato np.array para la red

out_empty = [0 for _ in range(len(labels))] # bag of words de las labels si la longitud es 5 sera [0, 0, 0, 0, 0]


# recoremos la variable de entrenamiento para su indice y su patron y creamos un bag of words
for x, pattern in enumerate(X):
    bag = []

    wrds = [stemmer.stem(w) for w in pattern]

    for w in words:
      if w in wrds:
        bag.append(1) # se activa el bit si la palabra esta en el patron
      else:
        bag.append(0)
    
    output_row = out_empty[:] # creamos una nueva fila de etiquetas o targets
    output_row[labels.index(Y[x])] = 1 # se activa el bit si la etiqueta esta en la lista de etiquetas

    training.append(bag) # añadimos la fila a la lista de entradas en bag of words
    output.append(output_row) # añadimos la fila a la lista de etiquetas o targets en bag of words
     

training = np.array(training) # añadimos la lista de entradas en np.array para el entrenamiento
output = np.array(output) # añadimos la lista de etiquetas o targets en np.array para el entrenamiento
     
# training => su forma seria (longitud de filas, longitud de columnas) ej. (23,4) => training.shape => (23,4)



tf.compat.v1.reset_default_graph() # reseteamos el grafo de entrenamiento

# ------------------------------------------------------------------------------------------- 
#  creamos el modelo deep para mayor obtencion de caracteristicas
# cabe destacar que se puede ir improvisando mejores bloques de red para mejores resultados
inp = input_data(shape=[None, len(training[0])]) # input_data es lo que recibe, una tupla con la forma (?, longitud de columnas o caracteristica)
# emb = tflearn.embedding(inp, input_dim=len(training[0])*100, output_dim=128) # embedding para mapear informacion
# lstm = tflearn.lstm(emb, 128, dropout=0.8) # red neuronal recurrente
dense1 = fully_connected(inp, 16, regularizer='L2', weight_decay=0.001) # red oculta super conectada con 16 neuronas
dense2 = fully_connected(dense1, 16)
softmax = fully_connected(dense2, len(output[0]), activation="softmax") # red de salida super conectada con x neuronas a predecir y una activacion softmax para decisiones mas precisas dado que es binario

# -------------------------------------------------------------------------
sgd = tflearn.SGD(learning_rate=0.1, lr_decay=0.96, decay_step=1000) # stochastis gradient descent
top_k = tflearn.metrics.Top_k(3) # metrica top_k 3
net_reg = tflearn.regression(softmax, optimizer=sgd, metric=top_k,
                         loss='categorical_crossentropy', learning_rate=0.001, restore=False) # se acopla la red a un modelo de regresión lineal

# ---------------------------------------------------------------------------

model = tflearn.DNN(net_reg, tensorboard_verbose=0) # se crea el modelo
     
# try:
# 	model.load("model.tflearn")
# except:

def train():
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True) # entrenamos el modelo por lotes de 16
    model.save("model.tflearn") # guardamos el modelo

# creamos el bag of words del input que recibiremos
# ejemplo: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1] siendo los 1 los valores activados
def create_bag_of_words(inp, words):
    bag = [0 for _ in range(len(words))]

    word_t = nltk.word_tokenize(inp) # tokenizamos la entrada
    word_t = [stemmer.stem(word.lower()) for word in word_t] # stemming de la entrada

    for se in word_t:
        for i, w in enumerate(words):
            if w == se: 
                bag[i] = 1 # se activa el bit si la palabra esta en el patron
    return np.array(bag) # retorna un array de numpy

# implementacion de chatbot basado en intentos
def chatBot(message):
    results = model.predict([create_bag_of_words(message,words)])[0] # se intenta predecir el primer valor de la lista 
    results_index = np.argmax(results) # se obtiene el maximo valor de la lista predicha
    
    tag = labels[results_index] 
    if results[results_index] > 0.5: # umbral del resultado debe ser mayor que .5
        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['tag']
                break
        # print(random.choice(responses))
        print(responses)
    else:
        print("No entiendo lo que dices sorry")

def chat():
    while True:
        try:
            inp = input("User: ")

            if inp.lower() == "exit":
                break

            print(chatBot(inp))
        except KeyboardInterrupt:
            break
        except:
            pass



if __name__ == "__main__":
    if ("--train" in sys.argv): train()
    else : model.load("model.tflearn")
    chat()