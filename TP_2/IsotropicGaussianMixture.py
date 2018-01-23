#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Othmane SAYEM et Pirashanth RATNAMOGAN
Definition de la classe:
IsotropicGaussianMixture 

Test de l'utilisation de la IsotropicGaussianMixture sur le jeu d'entrainement que l'on a,
affichage du résultat en couleur, et affichage de la log-vraisemblance.

On aurait pu creer cette classe comme une option dans la classe GaussianMixture 
mais puisque c'est une question a part nous fournissons le code complet
Seul la maniere de calculer les covariances dans le M step change.
"""

import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
from Utils import EM_data_train,EM_data_test
import math
from Kmeans import KMeans

EM_train = np.array([EM_data_train['x_1'],EM_data_train['x_2']]).transpose()
EM_test = np.array([EM_data_test['x_1'],EM_data_test['x_2']]).transpose()

############### Fonction pour calculer les lois gaussiennes sans utiliser scipy#############
Gaussian_law_estimation_unidimensional = lambda x,sigma,mu: 1/math.sqrt(2*math.pi*sigma**2)*np.exp((x-mu)**2/(sigma**2))
compute_exponential_const_term = lambda x,Sigma,mu: (2*math.pi)**(Sigma.shape[0]/2)*math.sqrt(np.linalg.det(Sigma))
Gaussian_law_estimation_multidimensional = lambda x,Sigma,mu: 1/compute_exponential_const_term(x,Sigma,mu)*np.exp(-1/2*np.dot(np.dot((x-mu),np.linalg.inv(Sigma)),(x-mu).transpose()))
############### End Fonction pour calculer les lois gaussiennes sans utiliser scipy#############

    
class IsotropicGaussianMixture:
    '''
    Class IsotropicGaussianMixture: permet de créer classifieur basé sur un modèle type IsotropicGaussianMixture Mixture
    Attributs: - k : nombre de cluster final, a fixer
               - Sigma_list : list(np.array) Liste des matrices de covariances indexé dans un ordre déterminé
               - mu_list :  list(np.array) Liste des moyennes dans le même ordre que Sigma_list
               - pi_list :  list(float)Probabilités d'appartions d'éléments dans les différents clusters dans le meme ordre que Sigma_list
               - q_e_step : np.array probabilité que l'élément i appartienne au cluster k (intermed EM)
    '''
    def __init__(self,nb_cluster=2):
        '''
        Fonction Constructeur: Initialise les attributs de la classe
        '''
        self.k = nb_cluster
        self.Sigma_list= 0
        self.mu_list =0 
        self.pi_list = 0
        self.q_e_step = 0
        
    def compute_E_step(self,data):
        '''
        Fonction qui calcule la M step de  notre algorithme EM
        Paramètres: data:(np.array(nb_samples,nb_composante)) Les échantillons sur lesquels sera calculé EM
        '''
        for i in range(data.shape[0]):
            for k in range(self.k):
                self.q_e_step[i,k] = self.pi_list[k]*Gaussian_law_estimation_multidimensional(data[i,:],self.Sigma_list[k],self.mu_list[k])
            
            self.q_e_step[i,:] = self.q_e_step[i,:]/(np.sum(self.q_e_step[i,:]))
    
    def compute_M_step(self,data):
        '''
        Fonction qui calcule la M step de  notre algorithme EM
        Paramètres: data:(np.array(nb_samples,nb_composante)) Les échantillons sur lesquels sera calculé EM
        '''
        for k in range(self.k):
            self.mu_list[k] = np.dot(data.transpose(),self.q_e_step[:,k])/(np.sum(self.q_e_step[:,k]))
            self.mu_list[k].resize((1,self.mu_list[k].shape[0]))
            self.pi_list[k] = np.sum(self.q_e_step[:,k])/np.sum(self.q_e_step)
            sigma_square = 0
            for i in range(data.shape[0]):
                sigma_square +=np.sum(self.q_e_step[i,k]*(data[i,:] -self.mu_list[k])**2)
            sigma_square = sigma_square/(2*np.sum(self.q_e_step[:,k]))
            self.Sigma_list[k] = sigma_square*np.eye(data.shape[1])
            
    
    def init_q_with_kmeans(self,data):
        '''
        Fonction qui initialise notre algorithme EM
        Paramètres: data:(np.array(nb_samples,nb_composante)) Les échantillons sur lesquels sera calculé EM
        '''
        self.q_e_step = np.zeros([data.shape[0],self.k])
        km = KMeans(self.k)
        km.fit(data)
        prediction = km.predict(data)
        for i in range(data.shape[0]):
            self.q_e_step[i,prediction[i]]=1
    
    def compute_log_likelihood_approx(self,data):
        '''
        Fonction qui calcule l'approximation utilisé pour minorer la vraie log likehood des données avec le modèle de gaussian mixture utilisé
        Paramètres: data:(np.array(nb_samples,nb_composante)) Les échantillons sur lesquels sera calculé la log likelihood
        '''
        q = np.zeros([data.shape[0],self.k])
        current_log=0
        for i in range(data.shape[0]):
            for k in range(self.k):
                q[i,k] = self.pi_list[k]*Gaussian_law_estimation_multidimensional(data[i,:],self.Sigma_list[k],self.mu_list[k])
            q[i,:] = q[i,:]/(np.sum(q[i,:]))
            for k in range(self.k):
                current_log += self.q_e_step[i,k]*(math.log(self.pi_list[k]) + math.log(Gaussian_law_estimation_multidimensional(data[i,:],self.Sigma_list[k],self.mu_list[k])))
        return current_log
    
    
    
    def compute_current_log_likelihood(self,data):
        '''
        Fonction qui calcule le vrai log-likelihood des données avec le modèle de gaussian mixture utilisé
        Paramètres: data:(np.array(nb_samples,nb_composante)) Les échantillons sur lesquels sera calculé la log likelihood
        '''
        
        current_log = 0
        for i in range(data.shape[0]):
            current_log_k=0
            for k in range(self.k):
                current_log_k += self.pi_list[k]*(Gaussian_law_estimation_multidimensional(data[i,:],self.Sigma_list[k],self.mu_list[k]))
            current_log += math.log(current_log_k)
        return current_log
    
    def fit(self,data,epsilon = 1e-5,verbose=0):
        '''
        Fonction fit: Permet de calculer les paramètres du modèle en utilisant EM
        Paramètres: data: (np.array(nb_samples,nb_composante)) Les échantillons sur lesquels sera basé l'apprentissaage
                    epsilon: (float) seuil de convergence de l'algorithme
                    verbose: (0 ou 1) afficher les calculs de log vraisemblance a chaque iteration ou non
        Return: Rien
        '''
        self.init_q_with_kmeans(data)
        
        self.mu_list = [None]*self.k
        self.pi_list = [None]*self.k
        self.Sigma_list = [None]*self.k
        
        self.compute_M_step(data)
        old_lg_like = -float('Inf')
        lg_like = 0
        nb_iteration = 0
        
        while (abs(lg_like-old_lg_like)>epsilon):
            
            nb_iteration +=1
            old_lg_like = lg_like
            self.compute_E_step(data)
            self.compute_M_step(data)
            lg_like= self.compute_current_log_likelihood(data)

            if (verbose==1):
                print('Iteration ',nb_iteration,'Log likelihood ',lg_like)
    
    def predict(self,data):
        '''
        Fonction predict: Hard clustering de toutes les données
        Paramètres: - data : (np.array(nb_samples,nb_composante)) échantillon à évaluer
        Return: Les labels associés à chaque cluster
        '''
        q = np.zeros([self.k])
        label = []
        for i in range(data.shape[0]):
            for k in range(self.k):
                q[k] = self.pi_list[k]*Gaussian_law_estimation_multidimensional(data[i,:],self.Sigma_list[k],self.mu_list[k])
            
            label.append(np.argmax(q))
        return np.array(label)
            

def compute_all_tests():
    '''
    Fonction qui permet de dérouler le test sur la base EM_train
    '''
    global EM_train
    global EM_test
    all_colors = list(matplotlib.colors.cnames.keys())
           
    Gm = IsotropicGaussianMixture(nb_cluster=4)
    print('Learn Parameters')
    Gm.fit(EM_train,verbose=1)
    prediction = Gm.predict(EM_train)
    h = .02
    
    x_min, x_max = EM_train[:, 0].min() - 1, EM_train[:, 0].max() + 1
    y_min, y_max = EM_train[:, 1].min() - 1, EM_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    
    colors = [all_colors[i+10] for i in prediction]
    plt.scatter(EM_train[:, 0], EM_train[:, 1],color = colors)
    # Trace le centroid
    centroids = np.array(Gm.mu_list)[:,0,:]
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='black', zorder=10)
    
    
    #Trace la matrice de covariance
    print('Plot Ellipse d\'incertitude, l\' opération peut prendre un certain temps')
    proba = 0.1
    for k in range(Gm.k):
        print('Tracé du cluster numeoro',k)
        Z_covar = np.zeros(xx.shape)
        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                    data = np.array([xx[i,j],yy[i,j]])
                    Z_covar[i,j]=compute_exponential_const_term(data,Gm.Sigma_list[k],Gm.mu_list[k])*Gaussian_law_estimation_multidimensional(data,Gm.Sigma_list[k],Gm.mu_list[k])
            
        plt.contour(xx,yy,Z_covar,[proba],colors=all_colors[k+10])
    
                    
    plt.title('EM clustering')
    
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show() 
    
    print('Log likelihood of the training set',Gm.compute_current_log_likelihood(EM_train))
    print('Log likelihood of the test set',Gm.compute_current_log_likelihood(EM_test))
    return Gm    