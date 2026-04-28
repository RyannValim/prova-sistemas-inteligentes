import pandas as pd

from normalizacao import Normalizador
from treinamento import Treinador
from predicao import Preditor
from descricao import Descritor

if __name__ == '__main__':
    df = pd.read_csv('./datasets/ObesityDataSet_raw_and_data_sinthetic.csv', sep=',')
    
    # preparação do df, checagens de valores ausentes
    normalizador = Normalizador()
    df_normalizado = normalizador.normalizar(df)
    normalizador.salvar(df_normalizado)
    
    # treinamento
    df_norm_drop_nobeyesdad = df_normalizado.drop(columns=['NObeyesdad'])
    treinador = Treinador(df_norm_drop_nobeyesdad)
    modelo_treinado = treinador.treinar()
    treinador.salvar(modelo_treinado)
    
    # inferência
    novo_dado = ['Female', 34.29, 1.60, 72.18, 'yes', 'yes', 2.40, 2.60, 'Sometimes', 'no', 1.67, 'no', 0.74, 0.25, 'Sometimes', 'Automobile'] # cluster 7
    
    # predição
    preditor = Preditor()
    predicao = preditor.prever(novo_dado)
    cluster = predicao[0]
    
    # descricao
    descritor = Descritor()
    descritor.descrever_todos(df_normalizado)
    
    print(f'\n======== INFERÊNCIA ========')
    print(f'O dado inserido pertence ao cluster {cluster}.')
    
    print(f'\nInformações deste cluster:')
    descritor.descrever_novo(cluster, df_normalizado)