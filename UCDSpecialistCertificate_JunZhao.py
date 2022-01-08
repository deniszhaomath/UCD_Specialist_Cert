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

#Import csv dataset of Crabs with various measures and their age

crab = pd.read_csv(r'C:/Users/ZHAOJ/Desktop/UCD Specialist Cert Project/CrabAgePrediction.csv')

print (crab.info())

#This dataset has no null values and duplicate values, but for project milestone purpose, coding are included

print (crab.isnull().sum())

#use dropna to drop any null values in the dataset and show the shape for comparison
dropnacrab = crab.dropna()

#Compare the shape of two files before and after dropping null values
print (crab.shape,dropnacrab.shape)

#There are three types value in 'sex' column, M for male, F for female and I for indeterminate
#Using value_counts function, there are 1233 rows of data with I for sex which accounts for close to around 30% of the whole dataset
print (crab['Sex'][0])

print (crab['Sex'].value_counts())
print (crab.head())

#Initial thought was to use Clustering Model to predict which rows of Indeterminate sex were for M or F
#However, using 2 models will increase the inaccuracy of the final prediction, so decided to drop the 30% data
#Dropped all rows with Inderminate Sex categoried as 'I' to remove noice for training/test dataset

crab_dropped_sex = crab[crab['Sex'] != 'I'].reset_index(drop=True)

print (crab_dropped_sex['Sex'])
#print (crab_dropped_sex.columns[0].value_counts())
print(crab_dropped_sex.info())

#The variables in the dataset: all Length, Diameter and Height are in Foot (unit) 
#All Weight columns are in Ounce (unit)
#Define a function that takes in the dataframe and loop through each row if column name matches
#either Length or Weight related and convert them into Centimeter or Gram (units)
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
crab_converted = Unit_Converter(template)
        
print (crab_converted.head())

#Subset two datasets male and female crabs to generate correlation heatmaps
crab_converted_male = crab_converted[crab_converted['Sex'] == 'M'].reset_index(drop=True)
print (crab_converted_male.head())

crab_converted_female = crab_converted[crab_converted['Sex'] == 'F'].reset_index(drop=True)
print (crab_converted_female.head())


#Heatmap showing correlation in Male crabs between age and physical attributes
corr = crab_converted_male.corr(method='spearman')

mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

fig, ax = plt.subplots(figsize=(6, 5))

cmap = sns.diverging_palette(220, 10, as_cmap=True, sep=100)

sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0, linewidths=.5)

fig.suptitle('Correlation matrix of features', fontsize=15)


fig.tight_layout()

#Heatmap showing correlation in Female crabs between age and physical attributes
corr = crab_converted_female.corr(method='spearman')

mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

fig, ax = plt.subplots(figsize=(6, 5))

cmap = sns.diverging_palette(220, 10, as_cmap=True, sep=100)

sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0, linewidths=.5)

fig.suptitle('Correlation matrix of Female features', fontsize=15)


fig.tight_layout()

#Create training and test data 
X = crab_converted.drop('Age', axis = 1).values
y = crab_converted['Age'].values

print (X.shape)
print (y.shape)


#Making one single prediction using 'Length'
crab_length = X[:,1]
print (crab_length.shape)
print (type(crab_length),type(y))
y = y.reshape(-1,1)
crab_length = crab_length.reshape(-1,1)

plt.scatter(crab_length,y)
plt.ylabel('Crab Age')
plt.xlabel('Crab Length (CM)')
plt.show()

crab_length_reg = LinearRegression()
crab_length_reg.fit(crab_length,y)
prediction_space = np.linspace(min(crab_length), max(crab_length)).reshape(-1,1)
plt.scatter(crab_length,y,color='blue')
plt.plot(prediction_space,crab_length_reg.predict(prediction_space),color='black',linewidth=3)
plt.show()

#Making single prediction using 'Diameter'
crab_diameter = X[:,2]
#y = y.reshape(-1,1)
crab_diameter = crab_diameter.reshape(-1,1)

plt.scatter(crab_diameter,y)
plt.ylabel('Crab Age')
plt.xlabel('Crab Diameter (CM)')
plt.show()

#Making single prediction using 'Height'
crab_height = X[:,3]
#y = y.reshape(-1,1)
crab_height = crab_height.reshape(-1,1)

plt.scatter(crab_height,y)
plt.ylabel('Crab Age')
plt.xlabel('Crab Height (CM)')
plt.show()

#Making single prediction using 'Weight'
crab_weight = X[:,4]
#y = y.reshape(-1,1)
crab_weight = crab_weight.reshape(-1,1)

plt.scatter(crab_weight,y)
plt.ylabel('Crab Age')
plt.xlabel('Crab Weight (Gram)')
plt.show()