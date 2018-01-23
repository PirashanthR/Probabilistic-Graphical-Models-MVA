#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 20:12:22 2017

@author: ratnamogan
"""
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from QDA_model import QDA_model
import math

##################Lecture de tous les fichiers de données#######################"
chemin_dossiers_donnees = '/home/ratnamogan/Documents/Probabilistic Graphical Models/TP_1/classification_data_HWK1/classification_data_HWK1/'
classificationA_train = pd.read_csv(chemin_dossiers_donnees + 'classificationA.train',delimiter='\t',names = ['x_1','x_2','y'])
classificationB_train = pd.read_csv(chemin_dossiers_donnees + 'classificationB.train',delimiter='\t',names = ['x_1','x_2','y'])
classificationC_train = pd.read_csv(chemin_dossiers_donnees + 'classificationC.train',delimiter='\t',names = ['x_1','x_2','y'])
classificationA_test = pd.read_csv(chemin_dossiers_donnees + 'classificationA.test',delimiter='\t',names = ['x_1','x_2','y'])
classificationB_test = pd.read_csv(chemin_dossiers_donnees + 'classificationB.test',delimiter='\t',names = ['x_1','x_2','y'])
classificationC_test = pd.read_csv(chemin_dossiers_donnees + 'classificationC.test',delimiter='\t',names = ['x_1','x_2','y'])
##################Fin Lecture de tous les fichiers de données#######################"


inverse_sigmoid_kind = lambda x,a: math.log(1/x-1) - math.log(a) #Fonction inverse de sigmoid avec une constante supplémentaire


def Utils_Plot_const_proba_line(Classifier,x,y,proba_const): ##Fonction qui doit etre appele dans une autre (ou les regresseurs sont connus)
    '''
    Fonction Utils_Plot_const_proba_line: Trace les échantillons (distingué selon leurs labels) ainsi
    que la frontiere séparant les éléments des différents labels 
    Paramètres: - Classifier : Classifier sur lequel est basé la frontiere tracé
                - x: (np.array(nb_samples,nb_composante)) Les échantillons sur lesquels sera basé l'apprentissaage
                - y: (np.array(nb_samples,)) label associé aux échantillons
                - proba_const: définit p(y=1|x)=proba_const, la frontiere que l'on trace
    Return: Les probabilités données par le modèle
    '''
    
    if type(Classifier)==QDA_model: ###On trace une conique
        borne_inf = min(min(x[:,0]),min(x[:,1]))
        born_sup = max(max(x[:,0]),max(x[:,1]))
        x1_for_plot_05 = np.linspace(borne_inf,born_sup,100)
        x2_for_plot_05 = np.linspace(borne_inf,born_sup,100)
        
        x1_for_plot_05, x2_for_plot_05 = np.meshgrid(x1_for_plot_05,x2_for_plot_05)
        Z, const = create_z_and_const_from_model_to_curve_plot(Classifier,x1_for_plot_05,x2_for_plot_05,0.5)
        
        
        plt.figure()
        
        color = ['r' if y[i]==0 else 'g' for i in range(y.shape[0])]
        plt.scatter(x= x[:,0],y=x[:,1],color=color)
        plt.contour(x1_for_plot_05,x2_for_plot_05,Z,[const],color='purple')
        
        red_patch = mpatches.Patch(color='red', label='y=0')
        green_patch = mpatches.Patch(color='green', label='y=1')
        blue_patch = mpatches.Patch(color='purple', label='p(y=1|x)=0.5')
        plt.legend(handles=[red_patch,green_patch,blue_patch])
        plt.show()
    else: ###ALors on trace une ligne
        x1_for_plot_05 = np.linspace(min(x[:,0]),max(x[:,0]),100)
        x2_for_plot_05 = Classifier.give_x2_for_given_proba(proba_const,x1_for_plot_05)
        plt.figure()
    
        color = ['r' if y[i]==0 else 'g' for i in range(y.shape[0])]
        plt.scatter(x= x[:,0],y=x[:,1],color=color)
        plt.plot(x1_for_plot_05,x2_for_plot_05,color='b')
        
        red_patch = mpatches.Patch(color='red', label='y=0')
        green_patch = mpatches.Patch(color='green', label='y=1')
        blue_patch = mpatches.Patch(color='blue', label='p(y=1|x)=0.5')
        plt.legend(handles=[red_patch,green_patch,blue_patch])
        
        
        plt.show()
    
def Utils_Misclassification_rate(y_predict,y_true):
    '''
    Fonction Utils_Misclassification_rate qui calcule le taux d'erreur d'une série de prédiction sachant la vraie réponse
    Paramètres: - y_predict: np.array((nb_samples)) predictions
                - y_true: np.array((nb_samples)) vraie réponse
    Return: Taux d'erreur
    '''
    diff = np.abs(y_true-y_predict)
    if diff.ndim ==2:
        diff.resize(diff.shape[1])
    return sum(diff)/y_true.shape[0]

def create_z_and_const_from_model_to_curve_plot(Model,tab_data_1,tab_data_2,prob):
    '''
    Fonction create_z_and_const_from_model_to_curve_plot calcule pour un mesh donnée les coordonnées de la conique défini par le Model
    Paramètres: - Model: Modèle de classification QDA
                - tab_data_1,tab_data_2 : mesh des coordonnées x,y sur lesquels
                - prob : proba p(y=1|x)=prob, la frontiere que l'on trace
    Return: Les coordonnées de la conique ainsi que la constante de normalisation
    '''
    Z_to_return = np.zeros(tab_data_1.shape)
    for i in range(tab_data_1.shape[0]):
        for j in range(tab_data_1.shape[1]):
            data = np.array([tab_data_1[i,j],tab_data_2[i,j]])
            data.resize([1,2])
            Z_to_return[i,j] = Model.calculate_exponential_term(data)
    det_0 = np.linalg.det(Model.Sigma_0)
    det_1 = np.linalg.det(Model.Sigma_1)
    const_term = inverse_sigmoid_kind(prob,(1-Model.pi)*math.sqrt(det_1)/(Model.pi*math.sqrt(det_0)))
    return Z_to_return,const_term

