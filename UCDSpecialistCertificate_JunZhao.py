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
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from scipy.stats import randint
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import  cross_val_score
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

###########Tried dropping Sex equals Indeterminate, and model score was maxed at 49.49%#######
#There are three types value in 'sex' column, M for male, F for female and I for indeterminate
#Using value_counts function, there are 1233 rows of data with I for sex which accounts for close to around 30% of the whole dataset
#print (crab['Sex'][0])

#print (crab['Sex'].value_counts())
#print (crab.head())

#Initial thought was to use Clustering Model to predict which rows of Indeterminate sex were for M or F
#However, using 2 models will increase the inaccuracy of the final prediction, so decided to drop the 30% data
#Dropped all rows with Inderminate Sex categoried as 'I' to remove noice for training/test dataset

#crab_dropped_sex = crab[crab['Sex'] != 'I'].reset_index(drop=True)

#print (crab_dropped_sex['Sex'])
#print (crab_dropped_sex.columns[0].value_counts())
#print(crab_dropped_sex.info())
###########This is dashed out and the dataset with Sex equals Indeterminate will be used to test model score########

#Check origianl dataset structure again
print(crab.head())


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
                
template = crab
crab_converted = Unit_Converter(template)
        
print (crab_converted.head())

## To be continued

#Subset two datasets male and female crabs to generate correlation heatmaps
crab_converted_male = crab_converted[crab_converted['Sex'] == 'M'].reset_index(drop=True)
print (crab_converted_male.head())

crab_converted_female = crab_converted[crab_converted['Sex'] == 'F'].reset_index(drop=True)
print (crab_converted_female.head())

#Merge of the two dataframe male and female together by using concatenate method
crab_merge = pd.concat([crab_converted_male, crab_converted_female], ignore_index=True)
print (crab_merge.head())

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

#Data preprocessing for Sex column
le = LabelEncoder()
crab_converted['Gender'] = le.fit_transform(crab_converted['Sex'])

onehot = OneHotEncoder()
crab_sex_df = pd.DataFrame(onehot.fit_transform(crab_converted[['Sex']]).toarray())
crab_converted = crab_converted.join(crab_sex_df)
print (crab_converted.head())
crab_converted.rename(columns = {0: "Female", 1: "Intermediate", 2 : "Male"}, inplace = True)
print (crab_converted.head())

#Drop Sex and Gender columns
crab_converted.drop(columns = ["Sex", "Gender"], inplace = True)
print (crab_converted.head())


#Create training and test data 
X = crab_converted.drop('Age', axis = 1).values
y = crab_converted['Age'].values

print (X.shape)
print (y.shape)


#Making one single prediction using 'Length'
crab_length = X[:,0]
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
plt.ylabel('Crab Age')
plt.xlabel('Crab Length (CM)')
plt.plot(prediction_space,crab_length_reg.predict(prediction_space),color='black',linewidth=3)
plt.show()
print (crab_length_reg.score(crab_length,y))

#Making single prediction using 'Diameter'
crab_diameter = X[:,1]
#y = y.reshape(-1,1)
crab_diameter = crab_diameter.reshape(-1,1)

plt.scatter(crab_diameter,y)
plt.ylabel('Crab Age')
plt.xlabel('Crab Diameter (CM)')
plt.show()

crab_diameter_reg = LinearRegression()
crab_diameter_reg.fit(crab_diameter,y)
prediction_space = np.linspace(min(crab_diameter), max(crab_diameter)).reshape(-1,1)
plt.scatter(crab_diameter,y,color='blue')
plt.ylabel('Crab Age')
plt.xlabel('Crab Diameter (CM)')
plt.plot(prediction_space,crab_diameter_reg.predict(prediction_space),color='black',linewidth=3)
plt.show()
print (crab_diameter_reg.score(crab_diameter,y))

#Making single prediction using 'Height'
crab_height = X[:,2]
#y = y.reshape(-1,1)
crab_height = crab_height.reshape(-1,1)

plt.scatter(crab_height,y)
plt.ylabel('Crab Age')
plt.xlabel('Crab Height (CM)')
plt.show()

crab_height_reg = LinearRegression()
crab_height_reg.fit(crab_height,y)
prediction_space = np.linspace(min(crab_height), max(crab_height)).reshape(-1,1)
plt.scatter(crab_height,y,color='blue')
plt.ylabel('Crab Age')
plt.xlabel('Crab Height (CM)')
plt.plot(prediction_space,crab_height_reg.predict(prediction_space),color='black',linewidth=3)
plt.show()
print (crab_height_reg.score(crab_height,y))

#Making single prediction using 'Weight'
crab_weight = X[:,3]
#y = y.reshape(-1,1)
crab_weight = crab_weight.reshape(-1,1)

plt.scatter(crab_weight,y)
plt.ylabel('Crab Age')
plt.xlabel('Crab Weight (Gram)')
plt.show()

crab_weight_reg = LinearRegression()
crab_weight_reg.fit(crab_weight,y)
prediction_space = np.linspace(min(crab_weight), max(crab_weight)).reshape(-1,1)
plt.scatter(crab_weight,y,color='blue')
plt.ylabel('Crab Age')
plt.xlabel('Crab Weight (Gram)')
plt.plot(prediction_space,crab_weight_reg.predict(prediction_space),color='black',linewidth=3)
plt.show()
print (crab_weight_reg.score(crab_weight,y))

#Split the prepared dataset into Training and Testing in 70% and 30% ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 101)

#Fit the model and predict
crab_final = LinearRegression()
crab_final.fit(X_train, y_train)
y_pred = crab_final.predict(X_test)

#Show score
print("R^2: {}".format(crab_final.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
print("Root Mean Squared Error: {}".format(rmse))

#Cross validation score
cv_results = cross_val_score(crab_final, X, y, cv = 5)
print (cv_results)

#Perform Ridge Regression
ridge = Ridge(alpha = 0.1, normalize=True)
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)
print (ridge.score(X_test, y_test))

#Perform Lasso Regression to show what features are best for predicting target
names = crab_converted.drop('Age', axis = 1).columns
lasso = Lasso(alpha=0.1)
lasso_coef = lasso.fit(X, y).coef_
_ = plt.plot(range(len(names)), lasso_coef)
_ = plt.xticks(range(len(names)), names, rotation=60)
_ = plt.ylabel('Coefficients')
plt.show()

#Hyperparameter tuning GridSearchCV for KNeighbourClassifier
param_dist = {"max_depth": [3, None],"max_features": randint(1, 9),
              "min_samples_leaf": randint(1, 9),
              "criterion": ["gini", "entropy"]}

tree = DecisionTreeClassifier()
tree_cv = RandomizedSearchCV(tree,param_dist , cv=5)
tree_cv.fit(X,y)
print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))
print("Best score is {}".format(tree_cv.best_score_))

#Hyperparameter take 2
l1_space = np.linspace(0, 1, 30)
param_grid = {'l1_ratio': l1_space}
elastic_net = ElasticNet()
gm_cv = GridSearchCV(elastic_net, param_grid, cv=5)
gm_cv.fit(X_train,y_train)
y_pred = gm_cv.predict(X_test)
r2 = gm_cv.score(X_test, y_test)
mse = mean_squared_error(y_test, y_pred)
print("Tuned ElasticNet l1 ratio: {}".format(gm_cv.best_params_))
print("Tuned ElasticNet R squared: {}".format(r2))
print("Tuned ElasticNet MSE: {}".format(mse))


#49.49% is the best score for this dataset

#Regex milestone

r1 = re.findall(r"certificate", "I have completed the certificate for Specialist Certificate for Data Analytics")
print (r1)

r2 = re.findall(r"c\w+", "I have completed the certificate for Specialist Certificate for Data Analytics")
print (r2)



mylist = ['bulldog', 'Beagledog', 'cat', 'fish', 'Shepherddog']
print (re.search(r"dog", mylist))





