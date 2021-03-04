import time
from dados import get_dados
from sklearn.model_selection import train_test_split

Ks = [1, 2, 3, 5, 7, 11, 13, 15]

relatorios = {} #--

def treinar(indice_dados, treinador, nome_algoritmo):
    x,y = get_dados(indice_dados)

    for k in Ks:
        treinador.K = k

        x_treino, x_test, y_treino, y_teste = train_test_split(x, y,random_state=1)

        comeco = time.time()
        treinador.treinamento(x_treino, y_treino)
        tempo_treino = time.time() - comeco

        comeco = time.time()
        relatorio = treinador.teste(x_test, y_teste)
        tempo_teste = time.time() - comeco

        relatorio['tempo_treinamento'] = int(tempo_treino * 1000)
        relatorio['tempo_teste'] = int(tempo_teste * 1000)

        relatorios[nome_algoritmo][k].append(relatorio)

    return None

