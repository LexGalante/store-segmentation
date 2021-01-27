# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from shutil import rmtree
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report,
                             confusion_matrix,
                             accuracy_score)

df = pd.read_csv('data.csv')
# separação dos dados de treino e teste
X = df.drop('club', axis=1)
Y = df['club']
X_train, x_test, Y_train, y_test = train_test_split(X, Y, test_size=0.2)
# aqui estamos experimentando várias parametrizacões
models = {
    # regressão logistica
    'logistic_regression_1': LogisticRegression(),
    'logistic_regression_2': LogisticRegression(
        penalty='l2',
        max_iter=1000
    ),
    'logistic_regression_3': LogisticRegression(
        penalty='l2',
        dual=False,
        max_iter=1000
    ),
    'logistic_regression_4': LogisticRegression(
        class_weight='balanced'
    ),
    'logistic_regression_cv_5': LogisticRegressionCV(
        cv=5,
        random_state=0
    ),
    # arvore de decisão
    'decision_tree_1': DecisionTreeClassifier(),
    'decision_tree_2': DecisionTreeClassifier(
        criterion='entropy',
        class_weight='balanced'
    ),
    'decision_tree_3': DecisionTreeClassifier(
        criterion='gini',
        splitter='random',
        class_weight='balanced'
    ),
    'decision_tree_4': DecisionTreeClassifier(
        criterion='entropy',
        splitter='random',
        class_weight='balanced'
    ),
    # florestas aleatórias
    'random_forest_1': RandomForestClassifier(),
    'random_forest_2': RandomForestClassifier(
        n_estimators=1000,
        criterion='entropy',
        class_weight='balanced'
    ),
    'random_forest_3': RandomForestClassifier(
        n_estimators=1000,
        criterion='gini',
        class_weight='balanced'
    ),
    # xgboost
    'x_boost_1': GradientBoostingClassifier(),
    'x_boost_2': GradientBoostingClassifier(
        n_estimators=300,
    ),
    # redes baysianas
    'naive_bayes_1': GaussianNB(),
    'naive_bayes_2': MultinomialNB(),
    # maquinas de suporte de vetores
    'svm_1': SVC(),
    'svm_2': SVC(
        kernel='poly',
    ),
    'svm_3': SVC(
        kernel='sigmoid',
    ),
    # rede neural
    'neural_network_1': MLPClassifier(),
    'neural_network_2': MLPClassifier(
        activation='relu',
        solver='adam',
        learning_rate='constant',
        max_iter=1000,
        random_state=100
    )
}
results = {}
if os.path.exists('./results'):
    rmtree('./results')
    
os.mkdir('./results')
# predições
for index, (name, model) in enumerate(models.items()):
    try:
        print(f"{index} - Training model {name} ...")
        model.fit(X_train, Y_train)
        predicts = model.predict(x_test)
        accuracy = round(accuracy_score(y_test, predicts) * 100, 2)
        print(f"{index} - Model {name} trained, accuracy {accuracy}")
        results[name] = {
            'accuracy_score': accuracy,
            'confusion_matrix': confusion_matrix(y_test, predicts),
            'classification_report': classification_report(y_test, predicts),
        }
        print(name)
        print(results[name]['confusion_matrix'])
        print(f"{index} - Dumping model {name} ...")
        dump(model, f"./dumps/{name}.joblib")
        print(f"{index} - Saving model {name} ...")
        with open(f"./results/{name}_{results[name]['accuracy_score']}.txt", 'w') as file:
            file.write(results[name]['classification_report'])
    except Exception as e:
        print(f"Error on trying to training model {name}: {str(e)}")




