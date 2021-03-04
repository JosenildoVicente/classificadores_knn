from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time

K = 5

modelo = KNeighborsClassifier(n_neighbors=K, weights='uniform')

def rodar(x_treinamento, y_treinamento, x_teste, y_teste):

    tempo_inicio = time.time()
    modelo = KNeighborsClassifier(n_neighbors=K, weights='uniform')
    modelo.fit(x_treinamento, y_treinamento)
    tempo_treino = time.time() - tempo_inicio

    tempo_inicio = time.time()
    y_modelo = modelo.predict(x_teste)
    tempo_teste = time.time() - tempo_inicio

    acuracia = accuracy_score(y_modelo, y_teste)
    
    resultado = {}
    resultado['tempo_treino'] = tempo_treino * 1000
    resultado['tempo_teste'] = tempo_teste * 1000
    resultado['acuracia'] = acuracia
    
    # print(y_teste == y_modelo)

    # print(y_modelo)
    # report = classification_report(y_teste, y_modelo, output_dict=True)
    return resultado
    # return report
    # return teste(x_test,y_test)

# def treinamento(x_treinamento, y_treinamento):
#     modelo = KNeighborsClassifier(n_neighbors=K, weights='uniform')
#     modelo.fit(x_treinamento, y_treinamento)
#     y_modelo = modelo.predict(x_treinamento)

# def teste(x_teste, y_teste):
#     y_modelo = modelo.predict(x_teste)
#     report = classification_report(y_teste,y_modelo, output_dict=True)
#     return report

