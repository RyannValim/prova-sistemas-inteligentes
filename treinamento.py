import pickle
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

class Treinador():
    def __init__(self, df):
        self.df = df
    
    def treinar(self):
        k_otimo = self.calc_elbow()
        return KMeans(n_clusters=k_otimo, random_state=42).fit(self.df)
    
    def calc_elbow(self):
        distorcoes = []
        clusters = range(1, 101)
        for c in clusters:
            treinador = KMeans(n_clusters=c, random_state=42).fit(self.df)
            distorcoes.append(
                sum(np.min(cdist(
                    self.df, treinador.cluster_centers_, 'euclidean'),
                           axis=1) / self.df.shape[0])
            )
        
        distancias = []
        for i in range(len(distorcoes)):
            x = clusters[i]
            y = distorcoes[i]
            
            x0 = clusters[0]
            y0 = distorcoes[0]
            xn = clusters[-1]
            yn = distorcoes[-1]

            distancias.append(abs(((
                (yn - y0)*x) - (xn - x0)*y) + (xn * y0) - (yn * x0)
            ) / sqrt((yn - y0)**2 + (xn - x0)**2))
        
        plt.plot(clusters, distorcoes)
        plt.savefig('./plotagens/elbow_curve.png')
        plt.close()
        
        return clusters[distancias.index(np.max(distancias))]
        
    def salvar(self, treinador):
        pickle.dump(treinador, open('./modelos/treinador_kmeans.pkl', 'wb'))