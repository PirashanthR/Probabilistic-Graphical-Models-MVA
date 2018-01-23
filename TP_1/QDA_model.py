#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Othmane SAYEM et Pirashanth RATNAMOGAN
Definition de la classe:
Class QDA_model
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import math
import seaborn as sns
from Utils import *

inverse_sigmoid_kind = lambda x,a: math.log(1/x-1) - math.log(a) 

class QDA_model:
    '''
    Class QDA_model: permet de créer un regresseur QDA
    Attributs: - pi : Paramètres de Bernouilli de la loi des labels
               - mu_0 : Moyenne lorsque le label est 0
               - mu_1 : Moyenne lorsque le label est 1
               - Sigma_0 : Matrice de covariance pour le label 0
               - Sigma_1 : Matrice de covariance pour le label 1
    '''
    def __init__(self):
        '''
        Fonction Constructeur: Initialise les attributs de la classe
        '''
        self.pi = 0
        self.mu_0 = 0
        self.mu_1 = 0
        self.Sigma_0= 0
        self.Sigma_1 = 0
        
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
        x_0 = np.array([p_x[:,i] for i in range(p_x.shape[1]) if p_y[0,i]==0])
        x_1 = np.array([p_x[:,i] for i in range(p_x.shape[1]) if p_y[0,i]==1])
        nb_data = p_x.shape[1]
        self.Sigma_0 = np.dot((x_0-self.mu_0.transpose()).transpose(),(x_0-self.mu_0.transpose()))/np.sum(1-p_y) 
        self.Sigma_1 = np.dot((x_1-self.mu_1.transpose()).transpose(),(x_1-self.mu_1.transpose()))/np.sum(p_y)
        self.pi = np.sum(p_y) / nb_data
        if size_changed:
            p_y.resize([p_y.shape[1],])
            
        return self.mu_0, self.mu_1, self.Sigma_0,self.Sigma_1, self.pi
    
    def get_coef(self):
        '''
        Fonction get_coef: Récupère les attributs de la classe
        Paramètres: -
        Return: Les paramètres du modèle
        '''
        return self.mu_0, self.mu_1, self.Sigma_0,self.Sigma_1, self.pi
    
    def predict(self,data):
        '''
        Fonction predict: Donne la probabilité d'obtenir le label y=1 pour d'échantillons données
        Paramètres: - data : (np.array(nb_samples,nb_composante)) échantillon à évaluer
        Return: Les probabilités données par le modèle
        '''
        Sigma_inv_0 = np.linalg.inv(self.Sigma_0)
        Sigma_inv_1 = np.linalg.inv(self.Sigma_1)
        det_0 = np.linalg.det(self.Sigma_0)
        det_1 = np.linalg.det(self.Sigma_1)
        y_0_term = np.array([np.dot(np.dot((data[i,:]-self.mu_0.transpose()),Sigma_inv_0),(data[i,:]-self.mu_0.transpose()).transpose()) for i in range(data.shape[0])])
        y_1_term = np.array([np.dot(np.dot((data[i,:]-self.mu_1.transpose()),Sigma_inv_1),(data[i,:]-self.mu_1.transpose()).transpose()) for i in range(data.shape[0])])
        y_0_term = y_0_term[:,0,0].transpose()
        y_1_term = y_1_term[:,0,0].transpose()
        return 1./(1+(1-self.pi)*math.sqrt(det_1)/(self.pi*math.sqrt(det_0))*np.exp(-1/2*(y_0_term-y_1_term)))
    
    
    def predict_class(self,data):
        '''
        Fonction predict_class: Donne le label évalué pour un ensemble d'échantillions données
        Paramètres: - data : (np.array(nb_samples,nb_composante)) échantillon à évaluer
        Return: Les labels évalues
        '''
        return np.array(self.predict(data)>=0.5,dtype=int)
    
    def calculate_exponential_term(self,data):
        '''
        Fonction qui calcule le terme dans l'exponentille de la loi de p(y=1|x) = 1/1+exp(X)
        Paramètres: - data : (np.array(nb_samples,nb_composante)) échantillon à évaluer
        
        Return: Les termes évalués
        '''
        Sigma_inv_0 = np.linalg.inv(self.Sigma_0)
        Sigma_inv_1 = np.linalg.inv(self.Sigma_1)
        y_0_term = np.array([np.dot(np.dot((data[i,:]-self.mu_0.transpose()),Sigma_inv_0),(data[i,:]-self.mu_0.transpose()).transpose()) for i in range(data.shape[0])])
        y_1_term = np.array([np.dot(np.dot((data[i,:]-self.mu_1.transpose()),Sigma_inv_1),(data[i,:]-self.mu_1.transpose()).transpose()) for i in range(data.shape[0])])
        y_0_term = y_0_term[:,0,0].transpose()
        y_1_term = y_1_term[:,0,0].transpose()
        return -1/2*(y_0_term-y_1_term)
    
    



 
