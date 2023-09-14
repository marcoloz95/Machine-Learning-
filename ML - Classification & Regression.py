#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 13:23:07 2023

@author: marco
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 17:18:58 2023

@author: marco
"""
#Group 1 Final 

#%% Classification: Price Range FINAL #~97% accurate 

from sklearn.ensemble import RandomForestClassifier #the classification method we used
from sklearn import preprocessing #necessary to re-label the descriptive columns
import pandas as pd #necessary to load the dataset
from sklearn.model_selection import train_test_split #used to split the dataset into X and Y 
import sklearn.metrics as sm #confusion matrix
from plotnine import *  #for graphing 
import seaborn as sns #for heatmap 
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix #confusion matrix for checking


# Create a sample dataframe
df = pd.read_csv('supermarket_sales.csv')#.dropna() #1292 rows with NAN (in choice_description) remove
df['item_price'] = df['item_price'].replace('[\$,]', '', regex=True).astype(float) #remove $ sign from item_price column, and convert numbers into float 
df_filtered = df[df['quantity'] == 1] #remove those with quantity > 1
df_filtered.describe()
df_filtered['price_range'] = pd.cut(df.item_price, [0, 4, 8,12], labels=["low",'medium', "high"], include_lowest=True) #set price ranges and label them 
df_filtered.describe()

#relabel descriptive columns
le1 = preprocessing.LabelEncoder()
le1.fit(df_filtered.price_range)
df_filtered.price_range = le1.transform(df_filtered.price_range) #from low, medium, high, it was assigned 0, 1


le2 = preprocessing.LabelEncoder()
le2.fit(df_filtered.item_name)
df_filtered.item_name = le2.transform(df_filtered.item_name) #numbers assigned to item names 

le3 = preprocessing.LabelEncoder() 
le3.fit(df_filtered.choice_description)
df_filtered.choice_description = le3.transform(df_filtered.choice_description) #numbers assigned to choice description 


Y = df_filtered.loc[:,["price_range"]] #this is the column you want to predict 
X = df_filtered.loc[:,['choice_description', 'item_name']] #variables used to predict Y 

# split the data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, shuffle=True)

# train the model
dt_model = RandomForestClassifier() #method we used 
dt_model.fit(X_train, Y_train) #fitting the training sets into the model 
dt_model.feature_importances_ #check what X variables were most important in determining Y 

imp = pd.DataFrame() #empty dataframe called 'imp'
imp["variable"] = X_train.columns #creates a new column called 'variable' which contains the column labels of X_train (choice_description and item_name) in the rows
imp["imp"] = dt_model.feature_importances_ #creates a new column called 'imp' which contains the values of the feature_importances_ above 
ggplot(aes(x="variable",y="imp"),imp) + geom_bar(stat="identity")

# make predictions and evaluate the model
predictions = dt_model.predict(X_test) #using X_test, we make our predictions for the price_range
accuracy = dt_model.score(X_test, Y_test) #the model is ~95% accurate
cm = sm.confusion_matrix(Y_test, predictions) 

#looking at the matrix diagonally from left to right ~ 97% accurate
print("Accuracy:", accuracy)
print(sm.classification_report(Y_test, predictions))

#heatmap
ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges")
ax.set_xlabel("Predictions")
ax.set_ylabel("Reality")



#%% 
#Double check of above through KNN ~98% Accurate (GOOD!)

#1 Build the 4 datasets: 
#Train: 80% of 4355 = 3483
#Test: 20% of 4355 = 871

from sklearn import preprocessing
import pandas as pd
import sklearn.metrics as sm
from plotnine import * 

# load the dataset
df = pd.read_csv('supermarket_sales.csv')#.dropna() #1292 rows with NAN (in choice_description) remove
df['item_price'] = df['item_price'].replace('[\$,]', '', regex=True).astype(float) #remove $ sign from item_price column, and convert numbers into float 
df_filtered = df[df['quantity'] == 1] #remove those with quantity > 1
df_filtered.describe()
df_filtered['price_range'] = pd.cut(df.item_price, [0, 4, 8,12], labels=["low",'medium', "high"], include_lowest=True) 

#Rename some columns 

le1 = preprocessing.LabelEncoder()
le1.fit(df_filtered.price_range)
df_filtered.price_range = le1.transform(df_filtered.price_range)


le2 = preprocessing.LabelEncoder()
le2.fit(df_filtered.item_name)
df_filtered.item_name = le2.transform(df_filtered.item_name)

le3 = preprocessing.LabelEncoder() 
le3.fit(df_filtered.choice_description)
df_filtered.choice_description = le3.transform(df_filtered.choice_description)

#split X and Y into train and test data sets
train = df_filtered.head(2700) #80% of the set
test = df_filtered.tail(676) #20% of the set

#assigning X variables used to predict Y (same as above)    
X_train = train.loc[:,["choice_description", "item_name"]]
X_test = test.loc[:,["choice_description", "item_name"]]
Y_train = train.price_range 
Y_test = test.price_range

#2. Build the model: 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import confusion_matrix 

mp1 = KNeighborsClassifier()
mp1.fit(X_train,Y_train)

predictions = mp1.predict(X_test) 
confusion_matrix(predictions, Y_test) #(1+150+498)/676 same as above 
mp1.score(X_test, Y_test)  #accuracy score of ~98%!




#%% Regression XGBoost 
#method that can handle non-linear relationship between X & Y variables
#XGBoost is a machine learning algorithm that uses decision trees to make predictions. 
#It repeatedly builds small trees that focus on the errors of the previous tree, 
#and then combines the predictions of all the trees into a single final prediction.

import xgboost as xgb
import numpy as np #for root mean squared error 
from sklearn.metrics import mean_squared_error #to check accuracy of model
from sklearn.metrics import mean_absolute_error #to check accuracy of model 
from sklearn.model_selection import train_test_split
import pandas as pd 
from sklearn import preprocessing
from plotnine import * 
from sklearn.metrics import confusion_matrix #confusion matrix for checking
import sklearn.metrics as sm


df = pd.read_csv('supermarket_sales.csv')
df['item_price'] = df['item_price'].replace('[\$,]', '', regex=True).astype(float) #remove $ sign from item_price column, and convert numbers into float 
df_filtered = df[df['quantity'] == 1] #remove quantity > 1 since we want item prices and not items

#relabel 
le1 = preprocessing.LabelEncoder()
le1.fit(df_filtered.item_name)
df_filtered.item_name = le1.transform(df_filtered.item_name)

le2 = preprocessing.LabelEncoder() 
le2.fit(df.choice_description)
df_filtered.choice_description = le2.transform(df_filtered.choice_description)

Y = df_filtered.loc[:,["item_price"]] #this is the column you want to predict 
X = df_filtered.loc[:,['choice_description', 'item_name']] #variables used to predict Y 

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, shuffle=True)


# Instantiate model
#xg_reg = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=5000, seed=50)
xg_reg = xgb.XGBRegressor() #went with default settings 

#n_estimators = the number of trees to be made for this analysis 
#reg:squarederror = model is trained to minimize the MSE between the predicted and actual values 

# Fit the model
xgb_model = xg_reg.fit(X_train, Y_train)

# Make predictions
y_pred = xg_reg.predict(X_test)

# Evaluate model performance (similar to accuracy in our classification model )
rmse = np.sqrt(mean_squared_error(Y_test, y_pred)) 
#Root Mean Squared Error (RMSE), measures the avg. distance between the predicted values and average values 

mse = mean_squared_error(Y_test, y_pred)
#Mean squared error (MSE) measures the average of the squared differences between predicted and actual values. 
#A lower MSE indicates a better fit of the model to the data.

mae = mean_absolute_error(Y_test, y_pred) 
#Mean absolute error (MAE) measures the average of the absolute differences between predicted and actual values. 
#A lower MAE also indicates a better fit of the model to the data.

xgb_model.score(X_test, Y_test) 

# get the feature importances
xgb_model.feature_importances_
    
imp = pd.DataFrame() #empty dataframe called 'imp'
imp["variable"] = X_train.columns #creates a new column called 'variable' which contains the column labels of X_train (choice_description and item_name) in the rows
imp["imp"] = xgb_model.feature_importances_ #creates a new column called 'imp' which contains the values of the feature_importances_ above 
ggplot(aes(x="variable",y="imp"),imp) + geom_bar(stat="identity")
    

# Convert pandas DataFrame to ggplot DataFrame
predtest = pd.DataFrame({'rmse': rmse, 'y_pred': y_pred, 'y_test': Y_test.values.ravel()}) #.values.ravel() to ensure 1 dimensional array
#creates a dataframe called 'predtest' that has the columns rmse, y_pred and y_test with the corresponding rows

# Plotting the predicted vs real values:
ggplot(predtest, aes(x='Y_test', y='y_pred')) + geom_point() + geom_abline(color="red", slope=1, intercept=0)

#Model accuracy
# Create a copy of Y_test and convert to a dataframe
allinfo = Y_test.copy()
allinfo = pd.DataFrame(allinfo)
# Rename the column to "Reality"
allinfo.columns = ["Reality"]
# Add the predictions as a new column named "Prediction"
allinfo["Prediction"] = y_pred
# Calculate the error as the difference between Reality and Prediction
allinfo["Error"] = allinfo.Reality - allinfo.Prediction
# Calculate the absolute error
allinfo["ErrorAbs"] = abs(allinfo.Error)
# Calculate the percentage error
allinfo['Percentage'] = 100* (allinfo.ErrorAbs / allinfo.Reality )
allinfo.describe()

#categorize prices below or over 8.5 as cheap or expensive
cheap = allinfo.loc[allinfo.Reality <= 8.5,:]
expensive = allinfo.loc[allinfo.Reality >= 8.5,:]

#error percentages labeling as good or bad depending on whether if they were cheap or expensive items 
cheap['Prediction_Accuracy'] = pd.cut(cheap.Percentage, [-0.1,10,1000], labels=["Good","Bad"])
cheap["Type"] = "Cheap"
expensive['Prediction_Accuracy'] = pd.cut(expensive.Percentage, [-0.01,10,1000], labels=["Good","Bad"])
expensive["Type"] = "Expensive"
allinfo = cheap.append(expensive)

allinfo.Prediction_Accuracy.value_counts() #749/873 = ~86%

ggplot(aes(x="Prediction",color = "Prediction_Accuracy", y="Reality"),allinfo) + geom_point() + geom_abline(color="red",slope=1, intercept=0)

ggplot(aes(x="ErrorAbs"), allinfo) + geom_histogram() + facet_wrap('Type')
ggplot(aes(x="Error"), allinfo) + geom_histogram() + facet_wrap('Type')



#%%

#just in case we wanted to see what items were in the predictions
allinfo_no_header = allinfo.iloc[1:]
num_rows = len(allinfo_no_header.index)
allinfo_no_header['item_name'] = X_test.item_name
complete = allinfo_no_header.reindex(columns=['item_name','Reality','Error','ErrorAbs','Percentage','Prediction', 'Prediction_Accuracy'])
complete['item name'] = df.item_name
complete = complete.reindex(columns=['item name','Reality','Prediction','Error','Percentage', 'Prediction_Accuracy'])

#%% Data Cleaning for Tableau 

import pandas as pd 
df = pd.read_csv('supermarket_sales.csv')#.dropna() #1292 rows with NAN (in choice_description) removed 
df['item_price'] = df['item_price'].replace('[\$,]', '', regex=True).astype(float) #remove $ sign from item_price column, and convert numbers into float 
df_filtered = df[df['quantity'] == 1]


item_quantity_price = df_filtered.loc[:,['item_name','quantity',"item_price"]].groupby(by=["item_name"], as_index = False).mean().sort_values(by='item_price', ascending=0)

item_quantity_price.to_excel(r'/Users/marco/Desktop/avgitemprice.xlsx', index=True)

x = df_filtered.loc[:,['quantity', 'item_name','choice_description',"item_price"]].groupby(by=["item_name"]).sum().sort_values(by='item_price', ascending=0)

x.to_excel(r'/Users/marco/Desktop/revenue.xlsx', index=True)

orders = df.loc[:,['order_id','quantity']].drop_duplicates(subset='order_id')

orders.to_excel(r'/Users/marco/Desktop/orders.xlsx', index=True)

#%%

