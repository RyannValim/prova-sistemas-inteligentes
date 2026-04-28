import pickle
import numpy as np
import pandas as pd

class Descritor():
    def __init__(self):
        self.minmax_norm = pickle.load(open('./modelos/minmax_scaler.pkl', 'rb'))
        self.ordinal_encoder = pickle.load(open('./modelos/ordinal_encoder.pkl', 'rb'))
        self.ohe_encoder = pickle.load(open('./modelos/ohe_encoder.pkl', 'rb'))
        self.treinador = pickle.load(open('./modelos/treinador_kmeans.pkl', 'rb'))

    def descrever_todos(self, df_normalizado):
        for i in range(self.treinador.n_clusters):
            self.descrever_novo(i, df_normalizado)

    def descrever_novo(self, cluster, df_normalizado):
        df_normalizado = df_normalizado.copy()
        df_normalizado['cluster'] = self.treinador.predict(df_normalizado.drop(columns=['NObeyesdad']))

        df_cluster = df_normalizado[df_normalizado['cluster'] == cluster]
        predominante = df_cluster['NObeyesdad'].value_counts().idxmax()

        # colunas para facilitar (não fiz algoritmo que detecta automaticamente)
        colunas_numericas = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
        colunas_binarias = ['family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC']
        colunas_ordinais = ['CAEC', 'CALC']
        ordem_ordinal = ['no', 'Sometimes', 'Frequently', 'Always']

        # acha o centróide
        centroide = self.treinador.cluster_centers_[cluster]
        colunas_sem_nobeyesdad = df_normalizado.drop(columns=['NObeyesdad', 'cluster']).columns.tolist()
        centroide_df = pd.DataFrame(data=[centroide], columns=colunas_sem_nobeyesdad)

        centroide_df[colunas_numericas] = self.minmax_norm.inverse_transform(centroide_df[colunas_numericas])

        print(f'\n===== CLUSTER {cluster} =====')
        print(f'Este cluster apresenta os seguintes padrões:')

        # humanizando para print
        for coluna in colunas_sem_nobeyesdad:
            valor = centroide_df[coluna].values[0]

            if coluna in colunas_numericas:
                print(f'  {coluna}: {valor:.2f}')

            elif coluna in colunas_binarias:
                print(f'  {coluna}: {"yes" if round(valor) == 1 else "no"}')

            elif coluna in colunas_ordinais:
                print(f'  {coluna}: {ordem_ordinal[round(valor)]}')

            else:
                pass

        
        colunas_ohe = self.ohe_encoder.get_feature_names_out(['Gender', 'MTRANS'])
        valores_ohe = centroide_df[colunas_ohe].values[0]
        valores_ohe = np.maximum(valores_ohe, 0)

        gender_cols = [c for c in colunas_ohe if c.startswith('Gender')]
        mtrans_cols = [c for c in colunas_ohe if c.startswith('MTRANS')]

        gender = gender_cols[np.argmax([centroide_df[c].values[0] for c in gender_cols])].split('_', 1)[1]
        mtrans = mtrans_cols[np.argmax([centroide_df[c].values[0] for c in mtrans_cols])].split('_', 1)[1]

        print(f'  Gender: {gender}')
        print(f'  MTRANS: {mtrans}')
        print(f'NObeyesdad predominante: {predominante}')