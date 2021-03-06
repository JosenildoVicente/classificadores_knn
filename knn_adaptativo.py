from math import dist
from scipy.io.arff.arffread import test_weka
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time
import numpy as np

K = 5

# modelo = KNeighborsClassifier(n_neighbors=K, weights='distance')

vetor_adaptativo = []

def distancia(a,b):
    return np.sqrt(np.sum((a-b)**2))

def rodar(x_treinamento, y_treinamento, x_teste, y_teste):

    #começo do treinamento
    tempo_inicio = time.time()
    
    vetor_adaptativo = np.zeros([len(x_treinamento)])

    for i in range(len(x_treinamento)):
        raio_minimo = 99999
        for j in range(len(x_treinamento)):
            if i==j or y_treinamento[i]==y_treinamento[j]:
                continue
            raio = distancia(x_treinamento[i],x_treinamento[j])
            if raio < raio_minimo:
                raio_minimo = raio
        vetor_adaptativo[i] = raio_minimo * 0.9999

    tempo_treino = time.time() - tempo_inicio

    # #Começo do teste:
    tempo_inicio = time.time()
    y_modelo=[]

    for instancia_teste in x_teste:

        distancias = np.empty(len(y_treinamento))

        for indice, instancia_treino in enumerate(x_treinamento):
            distancias[indice] = distancia(instancia_teste, instancia_treino)
        
        distancias = distancias / (vetor_adaptativo + 0.00001)

        distancias_minimas_indice = distancias.argsort()[:K]
        
        classificacao_treino = np.array(y_treinamento)[distancias_minimas_indice]
        
        presenca, frequencia = np.unique(classificacao_treino, return_counts=True)
        # print(presenca[frequencia.argmax()])
        result = presenca[frequencia.argmax()]
        y_modelo.append(result)
    
    tempo_teste = time.time() - tempo_inicio

    acuracia = accuracy_score(y_modelo, y_teste)
    
    resultado = {}
    resultado['tempo_treino'] = tempo_treino * 1000
    resultado['tempo_teste'] = tempo_teste * 1000
    resultado['acuracia'] = acuracia
    
    return resultado

# if __name__ == "__main__":
#     rodar([])