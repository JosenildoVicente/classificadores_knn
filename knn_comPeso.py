from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time

K = 5

# Método para rodar o algoritmo Knn com peso
def rodar(x_treinamento, y_treinamento, x_teste, y_teste):

    # Parte de treinamento
    tempo_inicio = time.time()
    modelo = KNeighborsClassifier(n_neighbors=K, weights='distance')
    modelo.fit(x_treinamento, y_treinamento)
    tempo_treino = time.time() - tempo_inicio

    # Parte de teste
    tempo_inicio = time.time()
    y_modelo = modelo.predict(x_teste)
    tempo_teste = time.time() - tempo_inicio

    # Calculo de acerto do modelo em relação ao resultado real
    acuracia = accuracy_score(y_modelo, y_teste)
    
    resultado = {}
    resultado['tempo_treino'] = tempo_treino * 1000
    resultado['tempo_teste'] = tempo_teste * 1000
    resultado['acuracia'] = acuracia
    
    return resultado