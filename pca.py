# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.decomposition import PCA

# importando os dados do dataset
df = pd.read_csv('data.csv', sep=',')

pca = PCA(n_components=3)
pca_components = pca.fit_transform(df.drop('club', axis=1))
pca_df = pd.DataFrame(
    data=pca_components, 
    columns=[
        'pca_1',
        'pca_2',
        'pca_3',
    ]
)
pca_df.to_csv('data-pca.csv', index=False)

