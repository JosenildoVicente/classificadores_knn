import knn_semPeso
import knn_comPeso
from dados import get_dados, nomes as nome_dados
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.style.use('seaborn-dark-palette')
import os
import pandas as pd

Ks = [1, 2, 3, 5, 7, 11, 13, 15]

relatorios = {} #--

nome_algoritmos = ["knn_sem_peso", "knn_com_peso", "knn_adaptativo"]
treinadores = [knn_semPeso, knn_comPeso, knn_semPeso]

def treinar(indice_dados, treinador, nome_algoritmo):
    x,y = get_dados(indice_dados)

    relatorios[nome_algoritmo] = {}
    for k in Ks:
        treinador.K = k
        relatorios[nome_algoritmo][k] = {}

        x_treino, x_teste, y_treino, y_teste = train_test_split(x, y)
        
        relatorio = treinador.rodar(x_treino, y_treino, x_teste, y_teste)
       
        relatorios[nome_algoritmo][k] = relatorio

def plotar_graficos(indice_dados):
    acuracias = []
    tempos_treino = []
    tempos_teste = []
    
    for i in range(len(treinadores)):

        treinar(indice_dados,treinadores[i], nome_algoritmos[i])
        
        acuracias.append([])
        tempos_teste.append([])
        tempos_treino.append([])
        
        for k in Ks:
            acuracias[i].append(relatorios[nome_algoritmos[i]][k]['acuracia'])
            tempos_treino[i].append(relatorios[nome_algoritmos[i]][k]['tempo_treino'])
            tempos_teste[i].append(relatorios[nome_algoritmos[i]][k]['tempo_teste'])
        
    # folder_name = "graficos/" + nome_dados[indice_dados] + "/"
    # if os.path.exists(folder_name) == False:
    #         os.makedirs(folder_name, mode = 0o666)
    
    df_acuracias = pd.DataFrame(acuracias)
    df_acuracias.index = nome_algoritmos[:len(treinadores)]
    df_acuracias.columns = Ks
    df_acuracias = df_acuracias.T

    df_tempos_treinamento = pd.DataFrame(tempos_treino)
    df_tempos_treinamento.index = nome_algoritmos[:len(treinadores)]
    df_tempos_treinamento.columns = Ks
    df_tempos_treinamento = df_tempos_treinamento.T
    
    df_tempos_teste = pd.DataFrame(tempos_teste)
    df_tempos_teste.index = nome_algoritmos[:len(treinadores)]
    df_tempos_teste.columns = Ks
    df_tempos_teste = df_tempos_teste.T
    
    df_acuracias.plot()
    plt.title("Acurácias dos algoritmos em relação aos Ks")
    plt.xlabel("Valor do K")
    plt.ylabel("Acurácia")
    # plt.savefig(folder_name + 'acuracias.png', format='png')
    plt.show()

    df_tempos_treinamento.plot()
    plt.title("Tempo de treinamento dos algoritmos em relação aos Ks")
    plt.xlabel("Valor do K")
    plt.ylabel("Tempo de treinamento (em ms)")
    # plt.savefig(folder_name + 'tempos_treino.png', format='png')
    plt.show()

    df_tempos_teste.plot()
    plt.title("Tempo de teste dos algoritmos em relação aos Ks")
    plt.xlabel("Valor do K")
    plt.ylabel("Tempo de teste (em ms)")
    # plt.savefig(folder_name + 'tempos_teste.png', format='png')
    plt.show()



if __name__ == '__main__':
    plotar_graficos(0)
    plotar_graficos(1)