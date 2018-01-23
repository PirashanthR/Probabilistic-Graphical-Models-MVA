#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Othmane SAYEM et Pirashanth RATNAMOGAN
Definition de la classe:
Class LogisticRegression
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import math
import seaborn as sns
from Utils import *

###########"Définition de fonctions simples mais utilisés fréquemment
sigmoid = lambda x: 1./(1+np.exp(-x)) # sigmoid
eta = lambda x,w: sigmoid(np.dot(w.transpose(),x)) #fonction eta du rapport
inverse_sigmoid = lambda x: math.log(1/x-1)  #inverse de la sigmoid
#####################################################################

def compute_D_eta(X_data,w):
    '''
    Fonction qui permet de calculer la matrice D_eta nécessaire au calcul de la hessienne (voir rapport)
    Paramètres: X_data :(np.array(nb_samples,nb_composante)) Les échantillons
                w : np.array(nb_composante,1)coefficients de la regression logistique
    Retrun: La matrice D_eta evalue
    '''
    diag_compo = eta(X_data,w)
    diag_compo = diag_compo*(1-diag_compo) 
    return np.diag(diag_compo[0,:])

class LogisticRegression:
    '''
    Class LogisticRegression: permet de créer un regresseur basé sur la regression logistique
    Attributs: - coef : Les coefficient w de la regression logistique, la derniere composante
    correspond a la composante affine
    '''
    
    def __init__(self):
        '''
        Fonction Constructeur: Initialise les attributs de la classe
        '''
        self.coef = 0
    
    def fit(self,data_raw,label,coef_old= np.array([[0],[0],[1]]),tolerance=0.01,lambda_regularisation=0):
        '''
        Fonction fit: Permet de calculer les estimateurs de maximum de vraisemblance des paramètres
        du modèle: va calibrer les attributs de la classe
        Paramètres: data_raw: (np.array(nb_samples,nb_composante)) Les échantillons sur lesquels sera basé l'apprentissaage
                    label: (np.array(nb_samples,)) label associé aux échantillons
                    coef_old:np.array(nb_composante,1)  anciens coefficients (utilise dans l'appel recursif)
                    tolerance: const, critere d'arret de l'algorithme de Newton
                    lambda_regularisation: 0=<const=<1 parametre de régularisation du modèle, 
        Return: Les paramètres du modèle qui ont été appris
        '''   
        ### matrice diagonale
        size_changed=False
        if label.ndim==1:
            label.resize([1,label.shape[0]])
            size_changed = True
        one_vector = np.ones((data_raw.shape[0],1))
        data = np.concatenate((data_raw,one_vector),axis=1)
        
        Diag = compute_D_eta(data.transpose(),coef_old)
        
        ### Hessienne et son inverse : pas de descente
        Hessian = np.dot(data.transpose(), np.dot(Diag,data)) + lambda_regularisation*np.eye(data.shape[1])
        Inv= np.linalg.inv(Hessian)
        Grad = np.dot((label-eta(data.transpose(),coef_old)),data).transpose() - lambda_regularisation*coef_old
        #### Terme de descente 
        D = np.dot(Inv,Grad)
        
        if size_changed:
            label.resize([label.shape[1],])
        #print(sum(label*np.log(eta(data.transpose(),coef_old))[0]) + sum((1-label)*np.log(eta(data.transpose(),-coef_old))[0]))
        if (np.linalg.norm(D)<tolerance):
            self.coef = coef_old
        else:
            self.fit(data_raw,label,coef_old+D,tolerance,lambda_regularisation)
    
    def get_coef(self):
        '''
        Fonction get_coef: Récupère les attributs de la classe
        Paramètres: -
        Return: Les paramètres du modèle
        '''
        return list(self.coef)

    def predict(self,data):
        '''
        Fonction predict: Donne la probabilité d'obtenir le label y=1 pour d'échantillons données
        Paramètres: - data : (np.array(nb_samples,nb_composante)) échantillon à évaluer
        Return: Les probabilités données par le modèle
        '''
        if data.ndim == 1:
            data.resize([data.shape[0],1])
        one_vector = np.ones([1,data.shape[0]])
        return eta(np.concatenate((data.transpose(),one_vector)),self.coef)
    
    def predict_class(self,data):
        '''
        Fonction predict_class: Donne le label évalué pour un ensemble d'échantillions données
        Paramètres: - data : (np.array(nb_samples,nb_composante)) échantillon à évaluer
        Return: Les labels évalues
        '''
        return np.array(self.predict(data)>=0.5,dtype=int)
    
    def give_x2_for_given_proba(self,proba,data_x1): #que pour dimension 2
        '''
        Fonction give_x2_for_given_proba: La donnée est sous la forme (x_1,x_2).
        Etant donnée x_1 cette fonction renvoie le x_2 qui permet d'avoir 
        pour notre modèle p(y|(x_1,x_2))= proba
        Paramètres: - data_x1 : (np.array(nb_samples,)) Les x_1 initiaux
                    - proba: (0<const<1) probabilité que l'on veut pour le couple (x_1,x_2)
        Return: L'ensemble des x_2 associés aux x_1 évalués
        '''
        affine_coef = (inverse_sigmoid(proba) - self.coef[2])/self.coef[1]
        linear_coef = -self.coef[0]/self.coef[1]
        return data_x1*linear_coef+affine_coef

