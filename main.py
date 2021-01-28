# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from helpers import clean_directory
from models import get_models
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report,
                             confusion_matrix,
                             accuracy_score)

# este é o dataset com as transformacoes realizadas, veja transform.py
df = pd.read_csv('data.csv')
# este é o dataset com aplicao de principal component analisys, veja pca.py
df_pca = pd.read_csv('data-pca.csv')
# separação dos dados de treino e teste
X = df.drop('club', axis=1)
# este X exclusivo de pca
X_PCA = df_pca
# aqui está nossa classe
Y = df['club']
# aqui estamos criando vários models com várias
# parametrizacões diferentes, veja models.py
models = get_models()
# a acuracia, matriz de confusao serão armazenadas aqui
results = {}


def build_models(X, Y, sufix_model_name: str = ""):
    # separando os dados de teste e treino
    X_train, x_test, Y_train, y_test = train_test_split(X, Y, test_size=0.2)
    for index, (name, model) in enumerate(models.items()):
        try:
            name = name + sufix_model_name
            print(f"{index} - Training model {name} ...")
            model.fit(X_train, Y_train)
            # predicões
            predicts = model.predict(x_test)
            # valor da acuracia total
            accuracy = round(accuracy_score(y_test, predicts) * 100, 2)
            print(f"{index} - Model {name} trained, accuracy {accuracy}")
            # armazenamos os resultados para análise posterior
            results[name] = {
                'accuracy_score': accuracy,
                'confusion_matrix': confusion_matrix(y_test, predicts),
                'classification_report': classification_report(y_test, predicts),
            }
            print(name)
            print(results[name]['confusion_matrix'])
            print(f"{index} - Dumping model {name} ...")
            # deploy do modelo
            dump(model, f"./dumps/{name}.joblib")
            print(f"{index} - Saving model {name} ...")
            # salvando os resultados
            with open(f"./results/{name}_{results[name]['accuracy_score']}.txt", 'w') as file:
                file.write(results[name]['classification_report'])
        except Exception as e:
            print(f"Error on trying to training model {name}: {str(e)}")
    

if __name__ == "__main__":
    clean_directory('./results')
    clean_directory('./dumps')

    build_models(X, Y)
    build_models(X, Y, "_pca")

