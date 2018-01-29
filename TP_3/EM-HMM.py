#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ratnamogan Pirashanth - Sayem Othmane

Ce fichier contient tout le rendu du DM3 du cours de probabilistic graphical
models.
Il se base sur l'implémentation de Gaussian Mixtures qui a été fait dans le 
DM2.

"""
#import pandas as pd
import numpy as np
import math
import matplotlib 
import matplotlib.pyplot as plt
from Utils import EM_data_train,EM_data_test
from GaussianMixture import GaussianMixture

##################Function that reads the data##############
EM_train = np.array([EM_data_train['x_1'],EM_data_train['x_2']]).transpose()
EM_test = np.array([EM_data_test['x_1'],EM_data_test['x_2']]).transpose()

#################Define Gaussian multi################""""""""
Gaussian_multi = lambda x,Sigma,mu: math.exp(-0.5*np.dot( np.transpose(x-mu),np.dot(np.linalg.inv(Sigma),x-mu)))/(2*math.pi*math.sqrt(np.linalg.det(Sigma)))   



def LogSumExp(x):
    '''
    Compute the sum in the log domain (in order to avoid numerical issues)
    Attributs: x : elements to sum
    Return: Final sum
    '''
    n= len(x) 
    x_max = np.max(x)
    s=0
    for i in range(n) :
        s += np.exp(x[i]-x_max)
    return x_max + np.log(s) 

class EM_HMM:
    '''
    Class EM_HMM: HMM with underlying GM law
    Attributs: - k : number of clusters
               - Sigma_list : list(np.array) list of covariance matrices
               - mu_list :  list(np.array) list of means
               - pi_list :  list(float) list of prior probabilities for each cluster
               - q_e_step : np.array: probability that element i is in cluster k 
    '''
    def __init__(self,A0,Sigma_list,mu_list,data):
        '''
        Constructor: Initialize the class
        '''
        self.k = 4
        self.sigma = Sigma_list 
        self.mu = mu_list
        self.A = A0
        self.pi_0 = 0.25*np.ones((1,4))
        self.q_e_step = np.zeros([data.shape[0],self.k]) 
        
    def compute_log_alpha(self, data):
        '''
        Alpha recursion in logarithm domain given the observations and the GM law
        Parameter: np.array data : observations
        return: alphas for all the time steps
        '''
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
        '''
        Beta recursion in logarithm domain given the observations and the GM law
        Parameter: np.array data : observations
        return: betas for all the time steps
        '''
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
        Compute the E step for the HMM
        parameters: np.array data: observations
        '''
        alpha = self.compute_log_alpha(data)
        beta = self.compute_log_beta(data)
        for t in range(data.shape[0]):
            for q in range(self.k):
                self.q_e_step[t,q] = alpha[t,q] + beta[t,q] - LogSumExp(alpha[t,:] + beta[t,:])
            
        
        self.q_e_step = np.exp(self.q_e_step)
    
    
    def compute_xi(self,data):
        '''
        Compute p(q_t,q_{t+1}|y)
        parameters: np.array data: observations
        '''
        log_alpha = self.compute_log_alpha(data)
        log_gamma = np.log(self.q_e_step)
        T = data.shape[0]
        log_xi = np.zeros((T-1,4,4))
        for t in range(T-1): 
            for i in range(4):
                for j in range(4) : 
                    p = np.log(Gaussian_multi(data[t+1],self.sigma[j],self.mu[j]))
                    log_xi[t,i,j] = log_alpha[t,i] + p + log_gamma[t+1,j] + np.log(self.A[j,i]) - log_alpha[t+1,j]
        
        return np.exp(log_xi)
    
    
    def compute_M_step(self,data):
        '''
        Compute the M step for the HMM
        parameters: np.array data: observations
        '''
        xi = self.compute_xi(data)
        for i in range(self.k):
            self.pi_0[0,i] = self.q_e_step[0,i]
            for j in range(self.k):
                self.A[i,j] = np.sum(xi[:,i,j])/np.sum(xi[:,i,:])
            
            self.mu[i] = np.dot(data.transpose(),self.q_e_step[:,i])/(np.sum(self.q_e_step[:,i]))
            self.mu[i].resize((1,self.mu[i].shape[0]))
            self.sigma[i] = sum([self.q_e_step[t,i]*np.dot(np.reshape((data[t,:] -self.mu[i]),(2,1)),np.reshape((data[t,:] -self.mu[i]),(1,2))) for t in range(data.shape[0])])
            self.sigma[i]= self.sigma[i]/np.sum(self.q_e_step[:,i])
            
    
    def compute_log_likelihood_approx(self,data):
        '''
        Fonction qui calcule l'approximation utilisé pour minorer la vraie log likehood des données avec le modèle de gaussian mixture utilisé
        Paramètres: data:(np.array(nb_samples,nb_composante)) Les échantillons sur lesquels sera calculé la log likelihood
        '''
        xi = self.compute_xi(data)
        current_log=0
        T=data.shape[0]
        for i in range(self.k) : 
            current_log+= self.q_e_step[0,i]*np.log(self.pi_0[i])
            for t in range(T-1):
                for j in range(self.k):
                    current_log += xi[t,j,i]*np.log(self.A[i,j])
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
        
    
    def fit(self,data,epsilon = 1e-5,verbose=1,validation_set=np.array(None)):
        '''
        Fonction fit: Compute the EM based learning to compute the parameters of the model
        Paramètres: data: (np.array(nb_samples,nb_composante)) observations
                    epsilon: (float) stopping criterio
                    verbose: (0 ou 1) print verbose
        Return: Rien
        '''
        lg_like = self.compute_current_log_likelihood(data)
        self.compute_E_step(data)
        old_lg_like = -float('Inf') #initialisation 
        likelihood = []
        likelihood.append(lg_like)
        
        if (validation_set!=None).all():
            likelihood_test = []
            likelihood_test.append(self.compute_current_log_likelihood(validation_set))
        nb_iteration = 0
        print('Iteration 0','Log likelihood ',lg_like)
        while abs(lg_like-old_lg_like)>epsilon: #stopping criteria
            
            nb_iteration +=1
            old_lg_like = lg_like
            self.compute_M_step(data)
            lg_like= self.compute_current_log_likelihood(data)
            likelihood.append(lg_like)
            if (validation_set!=None).all(): #compute logloss for a validationset
                likelihood_test.append(self.compute_current_log_likelihood(validation_set))
            if (verbose==1): 
                print('Iteration ',nb_iteration,'Log likelihood ',lg_like)
            self.compute_E_step(data)
        if (validation_set!=None).all():
            return likelihood,likelihood_test 
        else:
            return likelihood
            
    def predict_viterbi(self,data):
        '''
        Fonction predict: Hard clustering of all the data using viterbi decoding
        Paramètres: - data : (np.array(nb_samples,nb_composant): observations
        Return: labels associated to each data
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
    
    def compute_marginal_probabilities(self,data):
        '''
        Fonction marginal_probabilities: Compute the marginal probabilities  of the observations given the model
        Paramètres: - data : (np.array(nb_samples,nb_composant): observations
        Return: np.array() marginal probabilities
        '''
        alpha = self.compute_log_alpha(data)
        beta = self.compute_log_beta(data)
        marginal_probabilities = np.zeros([data.shape[0],self.k]) 
        for t in range(data.shape[0]):
            for q in range(self.k):
                marginal_probabilities[t,q] = alpha[t,q] + beta[t,q] - LogSumExp(alpha[t,:] + beta[t,:])
        
        
        marginal_probabilities = np.exp(marginal_probabilities)
        return marginal_probabilities
    
    def predict_marginal(self,data):
        '''
        Fonction marginal_probabilities: Hard clustering using marginal probabilities  of the observations given the model
        Paramètres: - data : (np.array(nb_samples,nb_composant): observations
        Return: np.array() marginal probabilities
        '''
        marginal = self.compute_marginal_probabilities(data)
        label = []
        for t in range(data.shape[0]):
            label.append(np.argmax(marginal[t,:]))
        return label
    
            
######Initialize HMM for training using previously computed GM#####
Gmtrain = GaussianMixture(nb_cluster=4)
print('Learn Parameters')
gm_likelihood_train = Gmtrain.fit(EM_train,verbose=0)
sigma_train = Gmtrain.Sigma_list 
mu_train = np.array(Gmtrain.mu_list)[:,0,:]    
A0 = (1/6)*np.ones((4,4))
np.fill_diagonal(A0,1/2)     
all_colors = list(matplotlib.colors.cnames.keys())


#####Question 2 Test alpha beta recursion using parameers of GM on test data#########
Gmtest = GaussianMixture(nb_cluster=4)
print('Learn Parameters')
gm_likelihood_test = Gmtest.fit(EM_test,verbose=0)
sigma_test = Gmtest.Sigma_list 
mu_test = np.array(Gmtest.mu_list)[:,0,:] 

hmm2 = EM_HMM(A0,sigma_test,mu_test,EM_test)
hmm2.compute_E_step(EM_test)
gamma = hmm2.q_e_step

#####Question 2 Plot the marginals, see the reports for details###
f, axarr = plt.subplots(2, 2)

f1, axarr1 = plt.subplots(2, 2)

f2, axarr2 = plt.subplots(2, 2)

f3, axarr3 = plt.subplots(2, 2)


col = ['green', 'red', 'blue', 'orange']
axarr[0, 0].scatter(range(25), gamma[0:25,0], color = col[0])
axarr1[0, 0].scatter(range(25), gamma[0:25,1], color = col[1])
axarr2[0, 0].scatter(range(25), gamma[0:25,2], color = col[2])
axarr3[0, 0].scatter(range(25), gamma[0:25,3], color = col[3])

axarr[0, 1].scatter(range(25,50), gamma[25:50,0], color = col[0])
axarr1[0, 1].scatter(range(25,50), gamma[25:50,1], color = col[1])
axarr2[0, 1].scatter(range(25,50), gamma[25:50,2], color = col[2])
axarr3[0, 1].scatter(range(25,50), gamma[25:50,3], color = col[3])
   
axarr[1, 0].scatter(range(50,75), gamma[50:75,0], color = col[0])
axarr1[1, 0].scatter(range(50,75), gamma[50:75,1], color = col[1])
axarr2[1, 0].scatter(range(50,75), gamma[50:75,2], color = col[2])
axarr3[1, 0].scatter(range(50,75), gamma[50:75,3], color = col[3])

axarr[1, 1].scatter(range(75,100), gamma[75:100,0], color = col[0])
axarr1[1, 1].scatter(range(75,100), gamma[75:100,1], color = col[1])
axarr2[1, 1].scatter(range(75,100), gamma[75:100,2], color = col[2])
axarr3[1, 1].scatter(range(75,100), gamma[75:100,3], color = col[3])

plt.figure(1)   
plt.savefig('plotq.png', dpi=800)
plt.figure(2)   
plt.savefig('plotq2.png', dpi=800)
plt.figure(3)   
plt.savefig('plotq3.png', dpi=800)
plt.figure(4)   
plt.savefig('plotq4.png', dpi=800)


print('Compute EM HMM')


### Question 4, learn model from train data
hmm_train = EM_HMM(A0,sigma_train,mu_train,EM_train)           
train_likelihood,test_likelihood = hmm_train.fit(EM_train,validation_set=EM_test)

### Question 5, plot log likelihoods of the model
f, ax = plt.subplots()
ax.plot(range(len(test_likelihood)),test_likelihood, label = 'Test')
ax.plot(range(len(train_likelihood)),train_likelihood, label = 'Train')
ax.legend()
ax.set_ylabel('Likelihood')
ax.set_xlabel('Iteration')
plt.savefig('likelihood.png', dpi=800)

#### Question 8, viterbi decoding
viterbi,best_path,prediction = hmm_train.predict_viterbi(EM_train)

colors = [all_colors[i+10] for i in prediction]
plt.scatter(EM_train[:, 0], EM_train[:, 1],color = colors)
 

#Question 9, Compute the marginals
gamma= hmm_train.compute_marginal_probabilities(EM_test)


#Ploting of the marginals and dump them
f, axarr = plt.subplots(2, 2)

f1, axarr1 = plt.subplots(2, 2)

f2, axarr2 = plt.subplots(2, 2)

f3, axarr3 = plt.subplots(2, 2)


col = ['green', 'red', 'blue', 'orange']
axarr[0, 0].scatter(range(25), gamma[0:25,0], color = col[0])
axarr1[0, 0].scatter(range(25), gamma[0:25,1], color = col[1])
axarr2[0, 0].scatter(range(25), gamma[0:25,2], color = col[2])
axarr3[0, 0].scatter(range(25), gamma[0:25,3], color = col[3])

axarr[0, 1].scatter(range(25,50), gamma[25:50,0], color = col[0])
axarr1[0, 1].scatter(range(25,50), gamma[25:50,1], color = col[1])
axarr2[0, 1].scatter(range(25,50), gamma[25:50,2], color = col[2])
axarr3[0, 1].scatter(range(25,50), gamma[25:50,3], color = col[3])
   
axarr[1, 0].scatter(range(50,75), gamma[50:75,0], color = col[0])
axarr1[1, 0].scatter(range(50,75), gamma[50:75,1], color = col[1])
axarr2[1, 0].scatter(range(50,75), gamma[50:75,2], color = col[2])
axarr3[1, 0].scatter(range(50,75), gamma[50:75,3], color = col[3])

axarr[1, 1].scatter(range(75,100), gamma[75:100,0], color = col[0])
axarr1[1, 1].scatter(range(75,100), gamma[75:100,1], color = col[1])
axarr2[1, 1].scatter(range(75,100), gamma[75:100,2], color = col[2])
axarr3[1, 1].scatter(range(75,100), gamma[75:100,3], color = col[3])

plt.figure(6)   
plt.savefig('plotq5.png', dpi=800)
plt.figure(7)   
plt.savefig('plotq6.png', dpi=800)
plt.figure(8)   
plt.savefig('plotq7.png', dpi=800)
plt.figure(9)   
plt.savefig('plotq8.png', dpi=800)

#Question 10, most likely state using marginals
label = hmm_train.predict_marginal(EM_test)

f, axarr = plt.subplots(2, 2)

axarr[0, 0].scatter(range(25), label[:25], color = col[0])
axarr[0, 1].scatter(range(25,50), label[25:50], color = col[0])
axarr[1, 0].scatter(range(50,75), label[50:75], color = col[0])
axarr[1, 1].scatter(range(75,100), label[75:100], color = col[0])

plt.figure(10)
plt.savefig('mostlikelyseq-marginals.png', dpi=800)

#Question 11, Viterbi on test data
viterbi,best_path,prediction_test = hmm_train.predict_viterbi(EM_test)

f, axarr = plt.subplots(2, 2)

axarr[0, 0].scatter(range(25), prediction_test[:25], color = col[0])
axarr[0, 1].scatter(range(25,50), prediction_test[25:50], color = col[0])
axarr[1, 0].scatter(range(50,75), prediction_test[50:75], color = col[0])
axarr[1, 1].scatter(range(75,100), prediction_test[75:100], color = col[0])


plt.figure(11)

plt.savefig('mostlikelyseq-viterbi.png', dpi=800)