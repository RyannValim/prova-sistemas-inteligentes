import pickle
import pandas as pd

class Preditor():
    def __init__(self):
        self.minmax_norm = pickle.load(open('./modelos/minmax_scaler.pkl', 'rb'))
        self.ordinal_encoder = pickle.load(open('./modelos/ordinal_encoder.pkl', 'rb'))
        self.ohe_encoder = pickle.load(open('./modelos/ohe_encoder.pkl', 'rb'))
        self.treinador = pickle.load(open('./modelos/treinador_kmeans.pkl', 'rb'))
        
    def preparar(self, df):
        colunas_numericas = df.select_dtypes(include='float64').columns
        df[colunas_numericas] = self.minmax_norm.transform(df[colunas_numericas])

        colunas_binarias = ['family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC']
        for coluna in colunas_binarias:
            df[coluna] = df[coluna].map({'yes': 1, 'no': 0})

        colunas_ordinais = ['CAEC', 'CALC']
        df[colunas_ordinais] = self.ordinal_encoder.transform(df[colunas_ordinais])

        colunas_ohe = ['Gender', 'MTRANS']
        df_ohe = pd.DataFrame(
            data=self.ohe_encoder.transform(df[colunas_ohe]).toarray(),
            columns=self.ohe_encoder.get_feature_names_out(colunas_ohe),
            index=df.index
        )
        
        df = df.drop(columns=colunas_ohe)
        return pd.concat([df, df_ohe], axis=1)
    
    def prever(self, novo_dado):
        colunas = ['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight',
                'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE',
                'CALC', 'MTRANS']
        df = pd.DataFrame([novo_dado], columns=colunas)
        df = self.preparar(df)
        return self.treinador.predict(df)