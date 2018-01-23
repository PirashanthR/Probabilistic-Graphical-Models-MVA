#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 00:30:18 2017

@author: sayemothmane
"""
#import pandas as pd
import numpy as np
import math
import matplotlib 
import matplotlib.pyplot as plt
from Utils import EM_data_train,EM_data_test
from GaussianMixture import GaussianMixture
#from scipy.special import logsumexp


EM_train = np.array([EM_data_train['x_1'],EM_data_train['x_2']]).transpose()
EM_test = np.array([EM_data_test['x_1'],EM_data_test['x_2']]).transpose()
Gaussian_multi = lambda x,Sigma,mu: math.exp(-0.5*np.dot( np.transpose(x-mu),np.dot(np.linalg.inv(Sigma),x-mu)))/(2*math.pi*math.sqrt(np.linalg.det(Sigma)))   



def LogSumExp(x):
    n= len(x) 
    x_max = np.max(x)
    s=0
    for i in range(n) :
        s += np.exp(x[i]-x_max)
    return x_max + np.log(s) 

class EM_HMM:
    '''
    Class GaussianMixture: permet de créer classifieur basé sur un modèle type Gaussian Mixture
    Attributs: - k : nombre de cluster final, a fixer
               - Sigma_list : list(np.array) Liste des matrices de covariances indexé dans un ordre déterminé
               - mu_list :  list(np.array) Liste des moyennes dans le même ordre que Sigma_list
               - pi_list :  list(float)Probabilités d'appartions d'éléments dans les différents clusters dans le meme ordre que Sigma_list
               - q_e_step : np.array probabilité que l'élément i appartienne au cluster k (intermed EM)
    '''
    def __init__(self,A0,Sigma_list,mu_list,data):
        '''
        Fonction Constructeur: Initialise les attributs de la classe
        '''
        self.k = 4
        self.sigma = Sigma_list 
        self.mu = mu_list
        self.A = A0
        self.pi_0 = 0.25*np.ones((1,4))
        self.q_e_step = np.zeros([data.shape[0],self.k]) 
        
    def compute_log_alpha(self, data):
        alpha_list = []
        alpha_0 = []
        for q in range(4): 
            alpha_0.append( np.log(self.pi_0[0,q]*Gaussian_multi(data[0],self.sigma[q],self.mu[q]) ) ) 
        
        alpha_0 = np.array(alpha_0)    
        alpha_list.append(alpha_0)
        T=len(data)
        for t in range(1,T):
           alpha_prev = alpha_list[-1]
           alpha_t = []
    
           for z in range(4):
               s=0
               log_s=0
               p = Gaussian_multi(data[t],self.sigma[z],self.mu[z])
               s = np.log(self.A[z,:]) + alpha_prev[:]
               log_s = LogSumExp(s)
               alpha_t.append(log_s + np.log(p))
    
           alpha_t = np.array(alpha_t)
           alpha_list.append(alpha_t)
         
        return np.array(alpha_list)   
     
        
    
    
    def compute_log_beta(self,data):
        beta_list = []
        beta_T = np.array([0,0,0,0])
        beta_list.append(beta_T)
        T=len(data)
        for t in range(1,T):
        
            beta_prev = beta_list[-1]
            beta_t = []
        
            for z in range(4):
                s=0
                s = [np.log(self.A[q,z]*Gaussian_multi(data[-t],self.sigma[q],self.mu[q]))+beta_prev[q] for q in range(4)]
                log_s = LogSumExp(s)
                beta_t.append(log_s)
        
        
            beta_t = np.array(beta_t)
            beta_list.append(beta_t)
    
        beta_list.reverse()
        return np.array(beta_list)                     
    
        
    def compute_E_step(self,data):
        '''
        Fonction qui calcule la M step de  notre algorithme EM
        Paramètres: data:(np.array(nb_samples,nb_composante)) Les échantillons sur lesquels sera calculé EM
        '''
        alpha = self.compute_log_alpha(data)
        beta = self.compute_log_beta(data)
        for t in range(data.shape[0]):
            for q in range(self.k):
                self.q_e_step[t,q] = alpha[t,q] + beta[t,q] - LogSumExp(alpha[t,:] + beta[t,:])
            
        
        self.q_e_step = np.exp(self.q_e_step)
    
    
    def compute_xhi(self,data):
        log_alpha = self.compute_log_alpha(data)
        log_gamma = np.log(self.q_e_step)
        T = data.shape[0]
        log_xhi = np.zeros((T-1,4,4))
        for t in range(T-1): 
            for i in range(4):
                for j in range(4) : 
                    p = np.log(Gaussian_multi(data[t+1],self.sigma[j],self.mu[j]))
                    log_xhi[t,i,j] = log_alpha[t,i] + p + log_gamma[t+1,j] + np.log(self.A[j,i]) - log_alpha[t+1,j]
        
        return np.exp(log_xhi)
    
    
    def compute_M_step(self,data):
        '''
        Fonction qui calcule la M step de  notre algorithme EM
        Paramètres: data:(np.array(nb_samples,nb_composante)) Les échantillons sur lesquels sera calculé EM
        '''
        xhi = self.compute_xhi(data)
        for i in range(self.k):
            self.pi_0[0,i] = self.q_e_step[0,i]
            for j in range(self.k):
                self.A[i,j] = np.sum(xhi[:,i,j])/np.sum(xhi[:,i,:])
            
            self.mu[i] = np.dot(data.transpose(),self.q_e_step[:,i])/(np.sum(self.q_e_step[:,i]))
            self.mu[i].resize((1,self.mu[i].shape[0]))
            self.sigma[i] = sum([self.q_e_step[t,i]*np.dot(np.reshape((data[t,:] -self.mu[i]),(2,1)),np.reshape((data[t,:] -self.mu[i]),(1,2))) for t in range(data.shape[0])])
            self.sigma[i]= self.sigma[i]/np.sum(self.q_e_step[:,i])
            
    
    def compute_log_likelihood_approx(self,data):
        '''
        Fonction qui calcule l'approximation utilisé pour minorer la vraie log likehood des données avec le modèle de gaussian mixture utilisé
        Paramètres: data:(np.array(nb_samples,nb_composante)) Les échantillons sur lesquels sera calculé la log likelihood
        '''
        xhi = self.compute_xhi(data)
        current_log=0
        T=data.shape[0]
        for i in range(self.k) : 
            current_log+= self.q_e_step[0,i]*np.log(self.pi_0[i])
            for t in range(T-1):
                for j in range(self.k):
                    current_log += xhi[t,j,i]*np.log(self.A[i,j])
                current_log += self.q_e_step[t,i]*np.log(Gaussian_multi(data[t],self.sigma[i],self.mu[i]))
            
            current_log += self.q_e_step[T-1,i]*np.log(Gaussian_multi(data[T-1],self.sigma[i],self.mu[i]))
        return current_log
        
       
    
    
    def compute_current_log_likelihood(self,data):
        '''
        Fonction qui calcule le vrai log-likelihood des données avec le modèle de gaussian mixture utilisé
        Paramètres: data:(np.array(nb_samples,nb_composante)) Les échantillons sur lesquels sera calculé la log likelihood
        '''
        
        log_alpha = self.compute_log_alpha(data)
        log_beta = self.compute_log_beta(data)
        ll= np.zeros((data.shape[0],1))
        for t in range(data.shape[0]):
            ll[t,0] = LogSumExp(log_alpha[t,:]+log_beta[t,:])
        
        return np.mean(ll)
        
    
    def fit(self,data,epsilon = 1e-5,verbose=1):
        '''
        Fonction fit: Permet de calculer les paramètres du modèle en utilisant EM
        Paramètres: data: (np.array(nb_samples,nb_composante)) Les échantillons sur lesquels sera basé l'apprentissaage
                    epsilon: (float) seuil de convergence de l'algorithme
                    verbose: (0 ou 1) afficher les calculs de log vraisemblance a chaque iteration ou non
        Return: Rien
        '''
        
        self.compute_E_step(data)
        old_lg_like = -float('Inf') #initialisation 
        lg_like = self.compute_current_log_likelihood(data)
        nb_iteration = 0
        print('Iteration 0','Log likelihood ',lg_like)
        while abs(lg_like-old_lg_like)>epsilon: #critere d'arret
            
            nb_iteration +=1
            old_lg_like = lg_like
            self.compute_M_step(data)
            lg_like= self.compute_current_log_likelihood(data)
            if (verbose==1): 
                print('Iteration ',nb_iteration,'Log likelihood ',lg_like)
            self.compute_E_step(data)
            
    def predict(self,data):
        '''
        Fonction predict: Hard clustering de toutes les données
        Paramètres: - data : (np.array(nb_samples,nb_composante)) échantillon à évaluer
        Return: Les labels associés à chaque cluster
        '''
        nb_state = self.pi_0.shape[1]
        T = data.shape[0]
        
        viterbi_matrix = np.zeros([nb_state,T])
        best_path = np.zeros([nb_state,T],dtype=int)
        
        #init
        for s in range(nb_state):
            viterbi_matrix[s,0] = np.log(self.pi_0[0,s]*Gaussian_multi(data[0],self.sigma[s],self.mu[s]))
        
        for t in range(1,T):
            for s in range(nb_state):
               p = Gaussian_multi(data[t],self.sigma[s],self.mu[s])
               z = np.log(self.A[:,s]) + viterbi_matrix[:,t-1]
               viterbi_matrix[s,t]= max(z + np.log(p))
               best_path[s,t] = int(np.argmax(z + np.log(p)))
              
        final_state = np.argmax(viterbi_matrix[:,T-1])
        
        label=[]
        label.append(final_state)
        
        for t in range(0,T-1):
            t_r = T-1-t
            best_next_pos= label[-1]
            label.append(best_path[best_next_pos,t_r])
        label.reverse()
        return viterbi_matrix,best_path,np.array(label)
    
 
    
Gm = GaussianMixture(nb_cluster=4)
print('Learn Parameters')
Gm.fit(EM_train,verbose=0)
sigma = Gm.Sigma_list 
mu = np.array(Gm.mu_list)[:,0,:]    
A0 = (1/6)*np.ones((4,4))
np.fill_diagonal(A0,1/2)     


hmm = EM_HMM(A0,sigma,mu,EM_train)           
hmm.fit(EM_train)
viterbi,best_path,prediction = hmm.predict(EM_train)

all_colors = list(matplotlib.colors.cnames.keys())
colors = [all_colors[i+10] for i in prediction]
plt.scatter(EM_train[:, 0], EM_train[:, 1],color = colors)