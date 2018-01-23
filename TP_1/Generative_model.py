#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Othmane SAYEM et Pirashanth RATNAMOGAN
Definition de la classe:
Class Generative_model
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import math
import seaborn as sns
from Utils import *


inverse_sigmoid_kind = lambda x,a: math.log(1/x-1) - math.log(a) 
class Generative_model:
    '''
    Class Generative_model: permet de créer un regresseur LDA
    Attributs: - pi : Paramètres de Bernouilli de la loi des labels
               - mu_0 : Moyenne lorsque le label est 0
               - mu_1 : Moyenne lorsque le label est 1
               - Sigma : Matrice de covariance
    '''
    def __init__(self):
        '''
        Fonction Constructeur: Initialise les attributs de la classe
        '''
        self.pi = 0
        self.mu_0 = 0
        self.mu_1 = 0
        self.Sigma= 0
    
    def fit(self,p_x,p_y):
        '''
        Fonction fit: Permet de calculer les estimateurs de maximum de vraisemblance des paramètres
        du modèle: va calibrer les attributs de la classe
        Paramètres: p_x: (np.array(nb_samples,nb_composante)) Les échantillons sur lesquels sera basé l'apprentissaage
                    p_y: (np.array(nb_samples,)) label associé aux échantillons
        Return: Les paramètres du modèle qui ont été appris
        '''
        p_x= p_x.transpose()
        size_changed= False
        if p_y.ndim ==1:
            p_y.resize([1,p_y.shape[0]])
            size_changed=True
        self.mu_0 = np.dot(p_x,(1-p_y).transpose())/np.sum(1-p_y) 
        self.mu_1 = np.dot(p_x,p_y.transpose())/np.sum(p_y) 
        mu_associated_data = p_y*self.mu_1 + (1-p_y)*self.mu_0
        
        nb_data = p_x.shape[1]
        Sigma = np.dot(p_x-mu_associated_data,(p_x-mu_associated_data).transpose())
        self.Sigma = Sigma/nb_data
        
        self.pi = np.sum(p_y) / nb_data
        
        if size_changed:
            p_y.resize([p_y.shape[1],])
            
        return self.mu_0, self.mu_1, self.Sigma, self.pi
    
    def get_coef(self):
        '''
        Fonction get_coef: Récupère les attributs de la classe
        Paramètres: -
        Return: Les paramètres du modèle
        '''
        return self.mu_0, self.mu_1, self.Sigma, self.pi
    
    def predict(self,data):
        '''
        Fonction predict: Donne la probabilité d'obtenir le label y=1 pour d'échantillons données
        Paramètres: - data : (np.array(nb_samples,nb_composante)) échantillon à évaluer
        Return: Les probabilités données par le modèle
        '''
        Sigma_inv = np.linalg.inv(self.Sigma)
        y_0_term = np.array([np.dot(np.dot((data[i,:]-self.mu_0.transpose()),Sigma_inv),(data[i,:]-self.mu_0.transpose()).transpose()) for i in range(data.shape[0])])
        y_1_term = np.array([np.dot(np.dot((data[i,:]-self.mu_1.transpose()),Sigma_inv),(data[i,:]-self.mu_1.transpose()).transpose()) for i in range(data.shape[0])])
        y_0_term = y_0_term[:,0,0].transpose()
        y_1_term = y_1_term[:,0,0].transpose()
        return 1./(1+(1-self.pi)/(self.pi)*np.exp(-1/2*(y_0_term-y_1_term)))
    
    def give_x2_for_given_proba(self,proba,data_x1): #que pour dimension 2
        '''
        Fonction give_x2_for_given_proba: La donnée est sous la forme (x_1,x_2).
        Etant donnée x_1 cette fonction renvoie le x_2 qui permet d'avoir 
        pour notre modèle p(y|(x_1,x_2))= proba
        Paramètres: - data_x1 : (np.array(nb_samples,)) Les x_1 initiaux
                    - proba: (0<const<1) probabilité que l'on veut pour le couple (x_1,x_2)
        Return: L'ensemble des x_2 associés aux x_1 évalués
        '''
        Sigma_inv = np.linalg.inv(self.Sigma)
        affine_composante_devel = 1/2*np.dot(np.dot(self.mu_1.transpose(),Sigma_inv),self.mu_1) - np.dot(np.dot(self.mu_0.transpose(),Sigma_inv),self.mu_0)
        composantes_x = (np.dot(self.mu_0.transpose(),Sigma_inv))-np.dot(self.mu_1.transpose(),Sigma_inv)
        composante_x_1 = composantes_x[0,0]
        composante_x_2 = composantes_x[0,1]
        affine_coef = (inverse_sigmoid_kind(proba,(1-self.pi)/self.pi) -affine_composante_devel)/composante_x_2
        linear_coef = -composante_x_1/composante_x_2
        predicted_data_x2 = data_x1*linear_coef+affine_coef
        predicted_data_x2.resize(data_x1.shape)
        return predicted_data_x2
    
    def predict_class(self,data):
        '''
        Fonction predict_class: Donne le label évalué pour un ensemble d'échantillions données
        Paramètres: - data : (np.array(nb_samples,nb_composante)) échantillon à évaluer
        Return: Les labels évalues
        '''
        return np.array(self.predict(data)>=0.5,dtype=int)
        