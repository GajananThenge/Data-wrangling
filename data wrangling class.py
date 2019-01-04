# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 09:37:51 2019

@author: gajanan.thenge
"""

import pandas as pd
import numpy as np


# Load Training and Test Data Sets
headers = ['age', 'workclass', 'fnlwgt', 
           'education', 'education-num', 
           'marital-status', 'occupation', 
           'relationship', 'race', 'sex', 
           'capital-gain', 'capital-loss', 
           'hours-per-week', 'native-country', 
           'predclass']
train_data_path=r"D:\Acadgild Classroom\Data-wrangling-master\train.csv"
test_data_path = r"D:\Acadgild Classroom\Data-wrangling-master\test.csv"
training_raw = pd.read_csv(train_data_path, 
                       header=None, 
                       names=headers,na_values=[" ?"])
test_raw = pd.read_csv(test_data_path, 
                      header=None, 
                      names=headers,na_values=[" ?"],skiprows=[0])

# Join Train and test dataset Datasets
dataset_raw = training_raw.append(test_raw)

#Reset the index
dataset_raw.reset_index(inplace=True)

dataset_raw.drop('index',inplace=True,axis=1)

# Describing all the Numerical Features
dataset_raw.describe()

# Describing all the Categorical Features
dataset_raw.describe(include=['O'])



# =============================================================================
# Misssing values
# =============================================================================

#Test to see if we can drop columns
my_columns = list(dataset_raw.columns)

# Get all missing column names
missing_info = list(dataset_raw.columns[dataset_raw.isnull().any()])


## Get the sum of missing values in each column 
dataset_raw.isnull().sum()


#Find the percentage of missing values in each column
for col in missing_info:
    percent_missing = float(dataset_raw[dataset_raw[col].isnull() == True].shape[0] / dataset_raw.shape[0])
    print('percent missing for column {}: {}'.format(
        col, percent_missing*100)) #percentage of missing data
    
    
# =============================================================================
# Ways to Cleanse Missing Data in Python
# a. Dropping Missing Values
# =============================================================================
non_na_df = dataset_raw.dropna()

## Verify by cheking sumof missing values 
non_na_df.isnull().sum()

# =============================================================================
# b. Replacing Missing Values in columns'capital-gain','capital-loss','hours-per-week' with 0
# =============================================================================
import numpy as np
replce_df = dataset_raw
replce_df[['capital-gain','capital-loss','hours-per-week']].replace({np.NaN:0})
# =============================================================================
# c. Replacing with a Scalar Value (mean,median etc)
# =============================================================================
mean_fill_df = dataset_raw.fillna(dataset_raw.mean())
# =============================================================================
# d. Filling Forward or Backward it willl take care of bi=oth the variable
# To fill forward, use the methods pad or fill, and to fill backward, use bfill and backfill.
# =============================================================================

ffil_df = dataset_raw.fillna(method="pad")
bfill_df = dataset_raw.fillna(method="bfill")

# =============================================================================
# formatting (strings, numbers, delimiters, etc)
# get the unique values of categorical data
# =============================================================================
str_formated_df = mean_fill_df

# Let's recode the Class Feature
str_formated_df.loc[str_formated_df['predclass'] == ' >50K', 'predclass'] = 1
str_formated_df.loc[str_formated_df['predclass'] == ' >50K.', 'predclass'] = 1
str_formated_df.loc[str_formated_df['predclass'] == ' <=50K', 'predclass'] = 0
str_formated_df.loc[str_formated_df['predclass'] == ' <=50K.', 'predclass'] = 0



    
str_formated_df['occupation'] = str_formated_df['occupation'].apply(lambda x : str(x).strip())
str_formated_df['native-country'] = str_formated_df['native-country'].apply(lambda x : str(x).strip())
str_formated_df['education'] = str_formated_df['education'].apply(lambda x : str(x).strip())
str_formated_df['marital-status'] = str_formated_df['marital-status'].apply(lambda x : str(x).strip())


#Get list 
cat_col = str_formated_df.select_dtypes(include=['O']).columns.tolist()
for col in cat_col:
    print('-'*50)
    print("{} unique values are: {}".format(col,str_formated_df[col].unique()))
    #print('-'*50)
    
##Select All objects df
#df_obj = df.select_dtypes(['object'])

# =============================================================================
# Fill categorical data with mode
# =============================================================================
fill_cat_df = str_formated_df
fill_cat_df = fill_cat_df.fillna(fill_cat_df.mode().iloc[0])
#for col in cat_col:
#    fill_cat_df[col]= fill_cat_df[col].fillna(fill_cat_df[col].mode(), inplace=True)  

fill_cat_df.isnull().sum()
# =============================================================================
# Create dummy variable
# =============================================================================
df_with_dummy_var = pd.get_dummies(fill_cat_df,drop_first=True)
