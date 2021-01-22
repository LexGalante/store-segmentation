# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.decomposition import PCA

df = pd.read_csv('data.csv')

pca = PCA(n_components=2)
pca_components = pca.fit_transform(df[])
pca_df = pd.DataFrame(
    data=pca_components, 
    columns=['principal component 1', 'principal component 2']
)

