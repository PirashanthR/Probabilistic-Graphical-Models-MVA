#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 22:22:32 2017

@author: sayemothmane
"""
import pandas as pd
import numpy as np
import math
import matplotlib 
import matplotlib.pyplot as plt
from Utils import EM_data_train,EM_data_test
from GaussianMixture import GaussianMixture


Gaussian_multi = lambda x,Sigma,mu: math.exp(-0.5*np.dot( np.transpose(x-mu),np.dot(np.linalg.inv(Sigma),x-mu)))/(2*math.pi*math.sqrt(np.linalg.det(Sigma)))   


def compute_alpha(A,y,sigma,mu):
    alpha_list = []
    alpha_0 = [0.25, 0.25, 0.25, 0.25]
    alpha_list.append(alpha_0)
    for t in range(0,T-1):
        
        alpha_prev = alpha_list[-1]
        alpha_t = []
        
        for z in range(4):
            s=0
            p = Gaussian_multi(y[t],sigma[z],mu[z])
            s = [A[z,q]*alpha_prev[q] for q in range(4)]
            s = sum(s)
            alpha_t.append(s*p)
        
        alpha_t = np.array(alpha_t)
        alpha_t = alpha_t/sum(alpha_t)
        alpha_list.append(alpha_t)
        
    return np.array(alpha_list)


def compute_beta(A,y,sigma,mu):
    beta_list = []
    beta_T = [1,1,1, 1]
    beta_list.append(beta_T)
    for t in range(1,T):
        
        beta_prev = beta_list[-1]
        beta_t = []
        
        for z in range(4):
            s=0
            s = [A[q,z]*beta_prev[q]*Gaussian_multi(y[-t],sigma[q],mu[q]) for q in range(4)]
            s = sum(s)
            beta_t.append(s)
        
        
#        beta_t = np.array(beta_t)
#        beta_t = 4*beta_t/sum(beta_t)
#        beta_list.append(beta_t)
        beta_list.append(beta_t)
    
    beta_list.reverse()
    return np.array(beta_list)     

EM_train = np.array([EM_data_train['x_1'],EM_data_train['x_2']]).transpose()
EM_test = np.array([EM_data_test['x_1'],EM_data_test['x_2']]).transpose()

    
T=500    
A = (1/6)*np.ones((4,4))
np.fill_diagonal(A,1/2) 


Gm = GaussianMixture(nb_cluster=4)
print('Learn Parameters')
Gm.fit(EM_train,verbose=1)

sigma = Gm.Sigma_list 
mu = np.array(Gm.mu_list)[:,0,:]
y=EM_test[0:T]

alpha_test = compute_alpha(A,y,sigma,mu)
beta_test = compute_beta(A,y,sigma,mu)
              

#normalizing  forward factors 
norm = np.transpose(np.tile(np.sum(alpha_test, axis=1),(4,1)))
alpha_norm  = alpha_test/norm
beta_norm = beta_test*norm

## for the state 1 

T_visu = 30
for z in range(4):
    gamma = []
    for t in range(T_visu):
        gamma_t  = alpha_norm[t,z]*beta_norm[t,z]
        gamma_t = gamma_t/np.dot(np.transpose(beta_norm[t,:]), alpha_norm[t,:])
        gamma.append(gamma_t)
    plt.scatter(range(T_visu), gamma)