from sklearn.metrics import accuracy_score
import time
import numpy as np

K = 5

# Variável para armazenar o maior raio das instancias de x_treinamento 
# que não contenha nenhuma instancia que seja de uma classe diferente da sua.
vetor_adaptativo = []

# Método para calcular a distância de forma euclidiana
def distancia(a,b):
    return np.sqrt(np.sum((a-b)**2))

# Método para rodar o algoritmo Knn adaptativo
def rodar(x_treinamento, y_treinamento, x_teste, y_teste):

    ## Começo da parte de treinamento
    tempo_inicio = time.time()

    # Inicialização do vetor com valores zeros
    vetor_adaptativo = np.zeros([len(x_treinamento)])

    # Parte de calculo dos raios, em que cada instancia de 
    # x_treinamento calcula a distancia entre ela e as outras
    # instancias que pertenca a outra classe (valor de) e guarda 
    # a que tiver a menor distancia. E para ser o maior raio que 
    # não tenha outra classe, é diminuido um valor mínimo do reio 
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
    ## Fim da parte de treinamento

    ## Começo da parte de teste
    tempo_inicio = time.time()

    # Vetor que armazenará o resultado da previsão do teste
    y_modelo=[]

    # Será calculado a distancia de cada instancia de x_teste com todas as instancias 
    # de X_treinamento e logo depois será feito o calculo da distancia local adaptativa 
    # de cada distancia e as K menores distancias locais serão guarados seus indices para 
    # ser verificado a classe (y_treinamento) de maior frequencia entre essas K instancias
    # e assim a classe de maior frequencia será a classe da instancia atual do teste e 
    # a classe é colocado em y_modelo.
    for instancia_teste in x_teste:
        distancias = np.empty(len(y_treinamento))
        for indice, instancia_treino in enumerate(x_treinamento):
            distancias[indice] = distancia(instancia_teste, instancia_treino)

        distancias = distancias / (vetor_adaptativo + 0.00001)
        distancias_minimas_indice = distancias.argsort()[:K] 

        classificacao_treino = np.array(y_treinamento)[distancias_minimas_indice]
        presenca, frequencia = np.unique(classificacao_treino, return_counts=True)
        result = presenca[frequencia.argmax()]
        y_modelo.append(result)
    
    tempo_teste = time.time() - tempo_inicio
    ## Fim da parte de teste

    # Calculo de acerto do modelo em relação ao resultado real
    acuracia = accuracy_score(y_modelo, y_teste)
    
    resultado = {}
    resultado['tempo_treino'] = tempo_treino * 1000
    resultado['tempo_teste'] = tempo_teste * 1000
    resultado['acuracia'] = acuracia
    
    return resultado
