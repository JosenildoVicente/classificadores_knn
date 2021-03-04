from scipy.io import arff
import math
import pandas as pd

nomes = ['cm1', 'pc1']
atributos = [21, 21]

def get_dados(indice):
    data = arff.loadarff('dados/' + nomes[indice]+'.arff')
    df = pd.DataFrame(data[0])
    df['defects'] = df['defects'].astype(str)
    x = df.iloc[:, :-1].values
    y = df.iloc[:, atributos[indice]].values
    return [x,y]


if __name__ == '__main__':
    [a,b] = get_dados(0)
    print("X: {}".format(a))
    print("Y: {}".format(b))
