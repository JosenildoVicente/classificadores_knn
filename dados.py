from scipy.io import arff
import math
import pandas as pd

nomes = ['cm1', 'pc1']
atributos = [21, 21]

def get_dados(indice):
    data = arff.loadarff('dados/' + nomes[indice]+'.arff')
    df = pd.DataFrame(data[0])
    df['defects'] = df['defects'].astype(str)
    return df


if __name__ == '__main__':
    # data_cm1 = arff.loadarff('dados/cm1.arff')
    # df_cm1 = pd.DataFrame(data_cm1[0])

    # data_pc1 = arff.loadarff('dados/pc1.arff')
    # df_pc1 = pd.DataFrame(data_pc1[0])
    conjuntoDados = get_dados(0)
    x = conjuntoDados.iloc[:, :-1].values
    y = conjuntoDados.iloc[:, atributos[0]].values
    print(x)
    print(y)