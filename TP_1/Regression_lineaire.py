#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Othmane SAYEM et Pirashanth RATNAMOGAN
Definition de la classe:
Class LinearRegression
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import math
from Utils import *

class LinearRegression:
    '''
    Class LinearRegression: permet de créer un regresseur basé sur la regression linéaire
    Attributs: - coef : Les coefficient w de la regression linéaire, la derniere composante
    correspond a la composante affine
    '''
    def __init__(self): #constructeur
        '''
        Fonction Constructeur: Initialise les attributs de la classe
        '''
        self.coef = 0
        
    def fit(self,data,label):
        '''
        Fonction fit: Permet de calculer les estimateurs de maximum de vraisemblance des paramètres
        du modèle: va calibrer les attributs de la classe
        Paramètres: data: (np.array(nb_samples,nb_composante)) Les échantillons sur lesquels sera basé l'apprentissaage
                    label: (np.array(nb_samples,)) label associé aux échantillons 
        Return: Les paramètres du modèle qui ont été appris
        '''   
        one_vector = np.ones([data.shape[0],1])
        data = np.concatenate((data,one_vector),axis=1)
        inverse_square_x = np.linalg.inv(np.dot(data.transpose(),data))
        self.coef= np.dot(np.dot(inverse_square_x,data.transpose()),label)
        self.coef.resize([self.coef.shape[0],1])
        return list(self.coef)
        
        
    def predict(self,data):
        '''
        Fonction predict: Donne la probabilité d'obtenir le label y=1 pour d'échantillons données
        Paramètres: - data : (np.array(nb_samples,nb_composante)) échantillon à évaluer
        Return: Les probabilités données par le modèle
        '''
        if data.ndim == 1:
            data.resize([data.shape[0],1])
        one_vector = np.ones([data.shape[0],1])
        return np.dot(self.coef.transpose(),np.concatenate((data,one_vector),axis=1).transpose())
    
    def give_x2_for_given_proba(self,proba,data_x1): #que pour dimension 2
        '''
        Fonction give_x2_for_given_proba: La donnée est sous la forme (x_1,x_2).
        Etant donnée x_1 cette fonction renvoie le x_2 qui permet d'avoir 
        pour notre modèle p(y|(x_1,x_2))= proba
        Paramètres: - data_x1 : (np.array(nb_samples,)) Les x_1 initiaux
                    - proba: (0<const<1) probabilité que l'on veut pour le couple (x_1,x_2)
        Return: L'ensemble des x_2 associés aux x_1 évalués
        '''
        affine_coef = (proba - self.coef[2])/self.coef[1]
        linear_coef = -self.coef[0]/self.coef[1]
        return data_x1*linear_coef+affine_coef
    
    def get_coef(self):
        '''
        Fonction get_coef: Récupère les attributs de la classe
        Paramètres: -
        Return: Les paramètres du modèle
        '''
        return list(self.coef)
    
    def predict_class(self,data):
        '''
        Fonction predict_class: Donne le label évalué pour un ensemble d'échantillions données
        Paramètres: - data : (np.array(nb_samples,nb_composante)) échantillon à évaluer
        Return: Les labels évalues
        '''
        return np.array(self.predict(data)>=0.5,dtype=int)


