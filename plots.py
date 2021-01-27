# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import seaborn as sns
# importando os dados do dataset
df = pd.read_csv('stores.csv', sep=',')

sns.heatmap(df.isnull(), cbar=False)

sns.pairplot(df, hue='club')

sns.countplot(x='club', data=df, hue='marca_1')
sns.countplot(x='club', data=df, hue='marca_2')
sns.countplot(x='club', data=df, hue='marca_3')
sns.countplot(x='club', data=df, hue='regiao')
sns.countplot(x='club', data=df, hue='canal')
sns.countplot(x='club', data=df, hue='localizacao')
sns.countplot(x='club', data=df, hue='numero_empregados')
sns.countplot(x='club', data=df, hue='modelo_arquitetonico')
sns.countplot(x='club', data=df, hue='tem_estoque_proprio')
sns.countplot(x='club', data=df, hue='legado_familia')
sns.countplot(x='club', data=df, hue='sistema_1')
sns.countplot(x='club', data=df, hue='sistema_2')
sns.countplot(x='club', data=df, hue='sistema_3')
sns.countplot(x='club', data=df, hue='sistemas_4')
sns.countplot(x='club', data=df, hue='sistema_5')
sns.countplot(x='club', data=df, hue='sistema_6')
sns.countplot(x='club', data=df, hue='maturidade_processo')
