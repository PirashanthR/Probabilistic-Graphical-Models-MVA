#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Othmane SAYEM et Pirashanth RATNAMOGAN
Fichier contenant le test comparant les différents modèles finales
"""


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import math
import seaborn as sns
from Utils import *
from Generative_model import Generative_model
from Regression_lineaire import LinearRegression
from Regression_logistique import LogisticRegression
from QDA_model import QDA_model


##############Créations des données de test et d'apprentissage##############
A_x_train = np.array([classificationA_train['x_1'],classificationA_train['x_2']]).transpose()
A_y_train = np.array(classificationA_train['y'])

A_x_test = np.array([classificationA_test['x_1'],classificationA_test['x_2']]).transpose()
A_y_test = np.array(classificationA_test['y'])

B_x_train = np.array([classificationB_train['x_1'],classificationB_train['x_2']]).transpose()
B_y_train = np.array(classificationB_train['y'])

B_x_test = np.array([classificationB_test['x_1'],classificationB_test['x_2']]).transpose()
B_y_test = np.array(classificationB_test['y'])

C_x_train = np.array([classificationC_train['x_1'],classificationC_train['x_2']]).transpose()
C_y_train = np.array(classificationC_train['y'])

C_x_test = np.array([classificationC_test['x_1'],classificationC_test['x_2']]).transpose()
C_y_test = np.array(classificationC_test['y'])
##############Fin Créations des données de test et d'apprentissage##############

# Generative model pour A, puis pour B puis pour C
A_generative_model = Generative_model()
A_generative_model.fit(A_x_train,A_y_train)
A_y_predict_g_t = A_generative_model.predict_class(A_x_train)
A_y_predict_g = A_generative_model.predict_class(A_x_test)
A_rate_g_t = Utils_Misclassification_rate(A_y_predict_g_t,A_y_train)
A_rate_g = Utils_Misclassification_rate(A_y_predict_g,A_y_test)

B_generative_model = Generative_model()
B_generative_model.fit(B_x_train,B_y_train)
B_y_predict_g_t = B_generative_model.predict_class(B_x_train)
B_y_predict_g = B_generative_model.predict_class(B_x_test)
B_rate_g_t = Utils_Misclassification_rate(B_y_predict_g_t,B_y_train)
B_rate_g = Utils_Misclassification_rate(B_y_predict_g,B_y_test)

C_generative_model = Generative_model()
C_generative_model.fit(C_x_train,C_y_train)
C_y_predict_g = C_generative_model.predict_class(C_x_test)
C_y_predict_g_t = C_generative_model.predict_class(C_x_train)
C_rate_g = Utils_Misclassification_rate(C_y_predict_g,C_y_test)
C_rate_g_t = Utils_Misclassification_rate(C_y_predict_g_t,C_y_train)


# Linear regression model pour A, puis pour B puis pour C
A_linear_regression_model = LinearRegression()
A_linear_regression_model.fit(A_x_train,A_y_train)
A_y_predict_l = A_linear_regression_model.predict_class(A_x_test)
A_y_predict_l_t = A_linear_regression_model.predict_class(A_x_train)
A_rate_l = Utils_Misclassification_rate(A_y_predict_l,A_y_test)
A_rate_l_t = Utils_Misclassification_rate(A_y_predict_l_t,A_y_train)

B_linear_regression_model = LinearRegression()
B_linear_regression_model.fit(B_x_train,B_y_train)
B_y_predict_l = B_linear_regression_model.predict_class(B_x_test)
B_y_predict_l_t = B_linear_regression_model.predict_class(B_x_train)
B_rate_l = Utils_Misclassification_rate(B_y_predict_l,B_y_test)
B_rate_l_t = Utils_Misclassification_rate(B_y_predict_l_t,B_y_train)

C_linear_regression_model = LinearRegression()
C_linear_regression_model.fit(C_x_train,C_y_train)
C_y_predict_l = C_linear_regression_model.predict_class(C_x_test)
C_y_predict_l_t = C_linear_regression_model.predict_class(C_x_train)
C_rate_l = Utils_Misclassification_rate(C_y_predict_l,C_y_test)
C_rate_l_t = Utils_Misclassification_rate(C_y_predict_l_t,C_y_train)

# QDA model pour A, puis pour B puis pour C
A_QDA_model = QDA_model()
A_QDA_model.fit(A_x_train,A_y_train)
A_y_predict_q = A_QDA_model.predict_class(A_x_test)
A_y_predict_q_t = A_QDA_model.predict_class(A_x_train)
A_rate_q = Utils_Misclassification_rate(A_y_predict_q,A_y_test)
A_rate_q_t = Utils_Misclassification_rate(A_y_predict_q_t,A_y_train)

B_QDA_model = QDA_model()
B_QDA_model.fit(B_x_train,B_y_train)
B_y_predict_q = B_QDA_model.predict_class(B_x_test)
B_y_predict_q_t = B_QDA_model.predict_class(B_x_train)
B_rate_q = Utils_Misclassification_rate(B_y_predict_q,B_y_test)
B_rate_q_t = Utils_Misclassification_rate(B_y_predict_q_t,B_y_train)

C_QDA_model = QDA_model()
C_QDA_model.fit(C_x_train,C_y_train)
C_y_predict_q = C_QDA_model.predict_class(C_x_test)
C_y_predict_q_t = C_QDA_model.predict_class(C_x_train)
C_rate_q = Utils_Misclassification_rate(C_y_predict_q,C_y_test)
C_rate_q_t = Utils_Misclassification_rate(C_y_predict_q_t,C_y_train)

# Logistic Regression model pour A, puis pour B puis pour C
A_Logistic_Regression = LogisticRegression()
A_Logistic_Regression.fit(A_x_train,A_y_train,lambda_regularisation=0.1)
A_y_predict_lr = A_Logistic_Regression.predict_class(A_x_test)
A_y_predict_lr_t = A_Logistic_Regression.predict_class(A_x_train)
A_rate_lr = Utils_Misclassification_rate(A_y_predict_lr,A_y_test)
A_rate_lr_t = Utils_Misclassification_rate(A_y_predict_lr_t,A_y_train)

B_Logistic_Regression = LogisticRegression()
B_Logistic_Regression.fit(B_x_train,B_y_train)
B_y_predict_lr = B_Logistic_Regression.predict_class(B_x_test)
B_y_predict_lr_t = B_Logistic_Regression.predict_class(B_x_train)
B_rate_lr = Utils_Misclassification_rate(B_y_predict_lr,B_y_test)
B_rate_lr_t = Utils_Misclassification_rate(B_y_predict_lr_t,B_y_train)

C_Logistic_Regression = LogisticRegression()
C_Logistic_Regression.fit(C_x_train,C_y_train)
C_y_predict_lr = C_Logistic_Regression.predict_class(C_x_test)
C_y_predict_lr_t = C_Logistic_Regression.predict_class(C_x_train)
C_rate_lr = Utils_Misclassification_rate(C_y_predict_lr,C_y_test)
C_rate_lr_t = Utils_Misclassification_rate(C_y_predict_lr_t,C_y_train)

# Trace de l'histogramme des résultats
x = ['Generative Model','Logistic Regression','Linear Regression','QDA Model']
y_A = [A_rate_g,A_rate_lr,A_rate_l,A_rate_q]
y_B = [B_rate_g,B_rate_lr,B_rate_l,B_rate_q]
y_C = [C_rate_g,C_rate_lr,C_rate_l,C_rate_q]

y_A_t = [A_rate_g_t,A_rate_lr_t,A_rate_l_t,A_rate_q_t]
y_B_t = [B_rate_g_t,B_rate_lr_t,B_rate_l_t,B_rate_q_t]
y_C_t = [C_rate_g_t,C_rate_lr_t,C_rate_l_t,C_rate_q_t]

f,(ax1,ax2,ax3) = plt.subplots(3, 1, figsize=(8, 6), sharex=True)

sns.barplot(x,y_A,ax=ax1,palette="Blues_d")
ax1.set_ylabel('Erreur  sur Base A')

sns.barplot(x,y_B,ax=ax2,palette="Blues_d")
ax2.set_ylabel('Erreur sur Base B')

sns.barplot(x,y_C,ax=ax3,palette="Blues_d")
ax3.set_ylabel('Erreur sur Base C')
# Finalize the plot
sns.despine(bottom=True)

f_t,(ax1_t,ax2_t,ax3_t) = plt.subplots(3, 1, figsize=(8, 6), sharex=True)

sns.barplot(x,y_A_t,ax=ax1_t,palette="Blues_d")
ax1_t.set_ylabel('Erreur  sur Base A')

sns.barplot(x,y_B_t,ax=ax2_t,palette="Blues_d")
ax2_t.set_ylabel('Erreur sur Base B')

sns.barplot(x,y_C_t,ax=ax3_t,palette="Blues_d")
ax3_t.set_ylabel('Erreur sur Base C')
# Finalize the plot
sns.despine(bottom=True)
