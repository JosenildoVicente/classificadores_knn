from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

K = 5

modelo = modelo = KNeighborsClassifier(n_neighbors=K, weights='uniform')

def treinamento(x_treinamento, y_treinamento):
    modelo = KNeighborsClassifier(n_neighbors=K, weights='uniform')
    modelo.fit(x_treinamento, y_treinamento)
    # return modelo

def teste(x_teste, y_teste):
    y_modelo = modelo.predict(x_teste)
    report = classification_report(y_teste,y_modelo, output_dict=True)
    return report

