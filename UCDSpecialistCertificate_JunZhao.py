# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 20:42:39 2021

@author: zhaoj
"""

#Import all relevant libraries for the project
import pandas as pd
import numpy as np
import re
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#Import dataset of Crabs with various measures and their age

crab = pd.read_csv(r'C:/Users/ZHAOJ/Desktop/UCD Specialist Cert Project/CrabAgePrediction.csv')

print (crab.info())

#This dataset has no null values and duplicate values, but for project milestone purpose, coding are included

print (crab.isnull().sum())
#use dropna to drop any null values in the dataset and show the shape for comparison
dropnacrab = crab.dropna()

print (crab.shape,dropnacrab.shape)

print (crab['Sex'][0])


print (crab['Sex'].value_counts())
print (crab.head())

#Dropped all rows with Inderminate Sex categoried as 'I' to remove noice for training/test dataset

crab_dropped_sex = crab[crab['Sex'] != 'I'].reset_index(drop=True)

print (crab_dropped_sex['Sex'])
#print (crab_dropped_sex.columns[0].value_counts())
print(crab_dropped_sex.info())


def Unit_Converter (df):
    lengthFactor = 30.48
    weightFactor = 28.35
    for col in df.columns:
        for i in range(len(df[col])):
            if col == 'Length' or col=='Diameter' or col=='Height':
                df[col][i] *= lengthFactor
            elif 'Weight' in col:
                df[col][i] *=weightFactor
    return df             
                
template = crab_dropped_sex
template = Unit_Converter(template)
        
print (template.head())





#corr = crab.corr(method='spearman')

#mask = np.zeros_like(corr, dtype=np.bool)
#mask[np.triu_indices_from(mask)] = True

#fig, ax = plt.subplots(figsize=(6, 5))

#cmap = sns.diverging_palette(220, 10, as_cmap=True, sep=100)

#sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0, linewidths=.5)

#fig.suptitle('Correlation matrix of features', fontsize=15)


#fig.tight_layout()