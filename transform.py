# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from utils.helpers import handle_boolean

# importando os dados do dataset
df = pd.read_csv('stores.csv', sep=',')
# aplicando o tratamento de para os valores N e S vamos converter para valores numericos
# necessários para os algorimos que iremos utilizar
columns_to_handle_boolean = [
    'marca_1',
    'marca_2',
    'marca_3',
    'tem_estoque_proprio',
    'usa_despachante',
    'legado_familia',
    'sistema_1',
    'sistema_2',
    'sistema_3',
    'sistemas_4',
    'sistema_5',
    'sistema_6',
]
for column in columns_to_handle_boolean:
    df[column] = df[column].apply(handle_boolean)
# aplicando o tratamento para variaveis categoricas
# vamos trazer elas para eixo horizontal ou seja para cada categoria teremos
# uma coluna com 0 1 para informar seu preenchimento
df_canal = pd.get_dummies(df['canal'])
df_canal = df_canal.rename(columns={
    'BO': 'canal_bo',
    'LJ': 'canal_loja',
    'VD': 'canal_vd'
})
df_localizacao = pd.get_dummies(df['localizacao'])
df_localizacao = df_localizacao.rename(columns={
    'BAIRRO': 'local_bairro',
    'CENTRO': 'local_centro',
    'RUA_FAMOSA': 'local_rua_famosa',
    'RURAL': 'local_rural',
    'SHOPPING': 'local_shopping'
})
df_modelo_arquitetonico = pd.get_dummies(df['modelo_arquitetonico'])
df_modelo_arquitetonico = df_modelo_arquitetonico.rename(columns={
    '1': 'arquitetura_1',
    '2': 'arquitetura_2',
    '3': 'arquitetura_3',
    '4': 'arquitetura_4',
})
df_faturamento_ultimo_ano = pd.get_dummies(df['faturamento_ultimo_ano'])
df_faturamento_ultimo_ano = df_faturamento_ultimo_ano.rename(columns={
    'A': 'faturamento_a',
    'B': 'faturamento_b',
    'C': 'faturamento_c',
    'D': 'faturamento_d',
})
df_maturidade_processo = pd.get_dummies(df['maturidade_processo'])
df_maturidade_processo = df_maturidade_processo.rename(columns={
    'ALTA': 'maturidade_alta',
    'MÉDIA': 'maturidade_media',
    'BAIXA': 'maturidade_baixa',
    'INEXISTENTE': 'maturidade_inexistente',
})
# agora vamos colocar estas novas colunas no dataframe final
df = pd.concat([
    df,
    df_canal,
    df_localizacao,
    df_modelo_arquitetonico,
    df_faturamento_ultimo_ano,
    df_maturidade_processo
], axis=1)
# vamos remover as colunas que passaram pelo processo de dummie
df = df.drop([
    'loja',
    'faturamento_ultimo_ano',
    'canal', 
    'localizacao', 
    'modelo_arquitetonico', 
    'faturamento_ultimo_ano', 
    'maturidade_processo'
], axis=1)