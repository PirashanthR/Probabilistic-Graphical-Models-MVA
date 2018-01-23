#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 14:08:52 2017

@author: ratnamogan
"""

import pandas as pd 
import numpy as np 

##################Lecture de tous les fichiers de données#######################"
chemin_dossiers_donnees = './classification_data_HWK2/'
EM_data_train = pd.read_csv(chemin_dossiers_donnees + 'EMGaussian.data',delimiter=' ',names = ['x_1','x_2'])
EM_data_test = pd.read_csv(chemin_dossiers_donnees + 'EMGaussian.test',delimiter=' ',names = ['x_1','x_2'])
##################Fin Lecture de tous les fichiers de données#######################"

EM_train = np.array([EM_data_train['x_1'],EM_data_train['x_2']]).transpose()
EM_test = np.array([EM_data_test['x_1'],EM_data_test['x_2']]).transpose()

