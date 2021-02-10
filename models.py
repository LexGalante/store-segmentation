# -*- coding: utf-8 -*-
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.svm import SVC
from tensorflow import keras, nn


def get_models():
    return {
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
        'svm_1': SVC(C=100),
        'svm_2': SVC(
            C=100,
            kernel='poly',
            degree=7
        ),
        'svm_3': SVC(
            C=100,
            kernel='sigmoid',
        ),
        # rede neural
        'neural_network_1': keras.Sequential([
            keras.layers.Dense(80, activation=nn.sigmoid),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(160, activation=nn.sigmoid),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(160, activation=nn.sigmoid),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(80, activation=nn.sigmoid),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(40, activation=nn.sigmoid),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(20, activation=nn.sigmoid),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(10, activation=nn.sigmoid),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(4, activation=nn.softmax)
        ])
    }