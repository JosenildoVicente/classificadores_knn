from scipy.io import arff
import pandas as pd

# Nome dos arquivos de dados e a quantidade de atributos que eles tem
nomes = ['cm1', 'pc1']
atributos = [21, 21]

# Método que carrega os dados do arquivo e retorna os Xs e Ys
def get_dados(indice):
    data = arff.loadarff('dados/' + nomes[indice]+'.arff')
    df = pd.DataFrame(data[0])
    df['defects'] = df['defects'].astype(str)
    x = df.iloc[:, :-1].values
    y = df.iloc[:, atributos[indice]].values
    return [x,y]

