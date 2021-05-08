# Project set up 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import linear_model
from sklearn import metrics
from sklearn import tree
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from mpl_toolkits.mplot3d import Axes3D

filename = 'USvideos.csv'
try:
    print("Reading input file %s ..." % filename)
    data = pd.read_csv(filename)
except:
    print("Error reading %s" % filename)
    exit(1)
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

# make y be the std_likes   #can this be 2 vars, likes/dislikes? Or would views be a better metric? (engagement is more important than approval these days)
y = preNorm['std_likes']

# make x = the remaining std_values in preNorm
x = preNorm[['std_category_id', 'std_views','std_dislikes', 'std_comment_count']]
#x = preNorm['std_views']

# use sklearns model_selection.train_test_split (cookie for short) to split them up nicely  
# currently using explicit defaults for test_size and random_state(seed)
xtrain, xtest, ytrain, ytest = model_selection.train_test_split(x,y,test_size=0.25, random_state=None)


# Decision tree model

deTree = tree.DecisionTreeRegressor()
deTree.fit(xtrain, ytrain)
dePred = deTree.predict(xtest)
evalTree = metrics.r2_score(ytest, dePred)
print("Decision Tree R2 score: %.2f" % evalTree)
print("Decistion Tree model score: ", str(round(evalTree, 2) * 100), "%")

print("dTree train accuracy: ", str(round(deTree.score(xtrain, ytrain),2)*100), '%')
print("dTree test accuracy: ", str(round(deTree.score(xtest, ytest),2)*100), '%')

# do percent error acc here
# ((Actual Likes-Predicted Likes)/Actual Likes) * 100 





# Linear regression
print()
print("Performing Linear Regression")
regREST = linear_model.LinearRegression()
regREST.fit(xtrain, ytrain)

# Prediction
print("Performing Prediction")
pred = regREST.predict(xtest) # predicting likes

# evaluate
    # evaluate here
    # we can use kfold or whatever we want 
    # the easy way is to just use eval_regression
    # -> eval_regression(regRest, pred, xtrain, ytain, xtest, ytest)
    


# upadted CV score
# prints out 10 scores, then gets the mean
score = model_selection.cross_val_score(regREST, xtrain, ytrain, cv=10)
print("scores", score)
print("mean: ", score.mean())











# this don't do anything tho lol, it just just prints numbers ;--;
kf = KFold(n_splits= 10)
print("Performing KFold validation")
i = 0
for train, test in kf.split(x):
    print(i, "...")
    i = i + 1
    #print("%s %s" % (xtrain, ytest))

score = metrics.r2_score(ytest, pred)
# this isn't accuracy lol
print("Accuracy is %.2f" % score)

###################################################
# dummy this part out in case we don't want a second test, it's finnicky and has a few videos that did not trend
file2 = 'scraped_data_manual.csv'
try:
    print("Reading input file %s ..." % file2)
    data2 = pd.read_csv(file2)
except:
    print("Error reading %s" % file2)
    exit(1)

test2 = data2
for var in finalFeatures:
    test2['std_'+var] = preprocessing.MinMaxScaler().fit_transform(test2[var].values.reshape(len(test2), 1))

y2 = test2['std_likes']
x2 = test2[['std_category_id', 'std_views','std_dislikes', 'std_comment_count']]
xtrain2, xtest2, ytrain2, ytest2 = model_selection.train_test_split(x2,y2,test_size=0.25, random_state=None)

print("Testing random videos...")
testReg = linear_model.LinearRegression()
testReg = testReg.fit(xtrain2, ytrain2)
pred2 = regREST.predict(xtest2)
score2 = metrics.r2_score(ytest2, pred2)
# again not accuracy ;-;
print("score is %.2f" % score2)

# testing 2 ends here
#####################################

xdum = xtest['std_views']
xdum2 = xtrain['std_views']

plt.figure(figsize=(4, 3))
ax = plt.axes()
ax.scatter(xdum2, ytrain, color='black')
ax.plot(xdum, pred)
#ax.plot(pred, xdum)

ax.set_xlabel('x')
ax.set_ylabel('y')

ax.axis('tight')
plt.show()