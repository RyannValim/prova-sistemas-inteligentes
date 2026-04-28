import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OrdinalEncoder, OneHotEncoder

class Normalizador():
    def __init__(self):
        self.minmax_scaler = MinMaxScaler()
        self.ordinal_encoder = OrdinalEncoder()
        self.ohe_encoder = OneHotEncoder()
        
    def normalizar(self, df):
        # min max scaler
        colunas_numericas = df.select_dtypes(include='float64').columns
        self.minmax_scaler.fit(df[colunas_numericas])
        df[colunas_numericas] = self.minmax_scaler.transform(df[colunas_numericas])

        # tratamento dos dados 'yes' e 'no' (1 e 0), que seria papel do label encoder,
        # porém para utilizar depois ele apresenta limitações, então ficará assim:
        colunas_binarias = ['family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC']
        for coluna in colunas_binarias:
            df[coluna] = df[coluna].map({'yes': 1, 'no': 0})

        # ordinal encoder
        colunas_ordinal_encoder = ['CAEC', 'CALC']
        self.ordinal_encoder = OrdinalEncoder(categories=[
            ['no', 'Sometimes', 'Frequently', 'Always'],
            ['no', 'Sometimes', 'Frequently', 'Always']
        ])
        self.ordinal_encoder.fit(df[colunas_ordinal_encoder])
        df[colunas_ordinal_encoder] = self.ordinal_encoder.transform(df[colunas_ordinal_encoder])

        # one hot encoder
        colunas_ohe_encoder = ['Gender', 'MTRANS']
        self.ohe_encoder.fit(df[colunas_ohe_encoder])
        
        df_ohe = pd.DataFrame(
            data=self.ohe_encoder.transform(df[colunas_ohe_encoder]).toarray(),
            columns=self.ohe_encoder.get_feature_names_out(colunas_ohe_encoder),
            index=df.index
        )
        
        df = df.drop(columns=colunas_ohe_encoder)
        
        return pd.concat([df, df_ohe], axis=1)

    def salvar(self, df):
        df.to_csv('./datasets/obesity_dataset_normalizado.csv', index=False)
        pickle.dump(self.minmax_scaler, open('./modelos/minmax_scaler.pkl', 'wb'))
        pickle.dump(self.ordinal_encoder, open('./modelos/ordinal_encoder.pkl', 'wb'))
        pickle.dump(self.ohe_encoder, open('./modelos/ohe_encoder.pkl', 'wb'))