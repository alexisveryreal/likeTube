# Project set up 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import linear_model

data = pd.read_csv('USvideos.csv')
data.head()

print('There are', str(len(data)), 'rows in this dataset')
print('\n')

# prints out the data information
data.info()


# seperation of numerical and categorical data
# categorical is basically the unique stuff per video
# numerical are the items that our model is based on + others

categorical = ['description','video_error_or_removed', 'ratings_disabled', 'comments_disabled', 'thumbnail_link', 'tags', 'publish_time', 'channel_title', 'title', 'trending_date', 'video_id']
numerical = ['comment_count', 'dislikes', 'views', 'likes', 'category_id']


# describe is smart and get information based on the data
# for categorical it gets things like frequencey, count, etc.
# for numerical it shows us the count, mean, standard deviation, min, etc.

# printing both for the categorical and the numerical

print(data[categorical].describe())

print(data[numerical].describe())
print()



# --------------------------------------------------------------------------------------
# Data separtion

# make sure we can read the numbers from the list

numerics = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']

print(data.select_dtypes(include=numerics).columns)
# print(data.select_dtypes(include=numerics).shape)

# make the numeric data anything that falls under the array above 
# should be the same 5 that we set apart above with int64 value
# but putting all data types incase these values change in the future or something
numericData = data.select_dtypes(include=numerics)

# show first 3 numeric Data just so we can see the values correctly
# print(numericData.head(3))

# same thing as above but for categorical data
# we just have to make one with object and bool
categorics = ['object', 'bool']

# --> prints to make sure we got the right ones
# print(data.select_dtypes(include=categorics).columns)
# print(data.select_dtypes(include=categorics).shape)


categoricalData = data.select_dtypes(include=categorics)
# print(categoricalData.head(3))

# ------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------
# Data preprocessing -----------------------------------------------------------------

# First we look at which categorical features have the most unique values
# that way we know what to drop

uniqueCategoryData = categoricalData.nunique().reset_index()

uniqueCategoryData.columns = ['Features', 'Unique Values']

uniqueCategoryData = uniqueCategoryData.sort_values('Unique Values', ascending=False)

print(uniqueCategoryData)

print()
# Dropping values
# we are dropping description, title, tn_l, v_i, p_t, tags, c_t, t_d
# since they have more than a couple of unique values
# we are just going to assume they don't belong round these parts , YEEEHAW

dropData = data.drop(['description', 'title', 'thumbnail_link', 'video_id', 'publish_time', 'tags', 'channel_title', 'trending_date'], axis=1)

print(dropData)


#  Dupe values
# make sure there isnt any duplicate values
print(dropData.duplicated().sum()) # results in 48 duplicates smh

# drop them duplicates
dropData = dropData.drop_duplicates()
print()
# print again to be sure they are 0
print(dropData.duplicated().sum())

# final features -----

finalFeatures = ['category_id', 'views', 'likes', 'dislikes', 'comment_count']

# copy the dropData 
preNorm = dropData.copy()

# Normalization!
for var in finalFeatures:
    preNorm['std_'+var] = preprocessing.MinMaxScaler().fit_transform(preNorm[var].values.reshape(len(preNorm), 1))
    
# print both out to see comparisons between not and normalized
print(dropData.describe())
print()
print(preNorm.head())
# --------------------------------------------------

# Split train and split here

    # make x = all the std_ values that show up in preNorm.head
    # so x = preNorm[['std_likes', 'etc']]
    # i forgot if we wanted to do just views or we can do the rest 

    # make y be the std_likes
    # y = preNorm['std_likes']
    
    # use sklearns model_selection.train_test_split (cookie for short) to split them up nicely
    # not sure what we could use for testsize and random state
    # xtrain, xtest, yrain, ytest = cookie(x,y,test_size=?, random_state=?)

# Linear regression
    # we can just use sklearns linear_model.LinearRegression
    # regREST = LinearRegression()
    # regREST.fit(xtrain, ytrain)

# Prediction
    # pred = regREST.predict(xtest) # predicting likes
    

# evaluate
    # evaluate here
    # we can use kfold or whatever we want 
    # the easy way is to just use eval_regression
    # -> eval_regression(regRest, pred, xtrain, ytain, xtest, ytest)
