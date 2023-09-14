######################
### REG.1
######################
import pandas as pd
from plotnine import *
from sklearn.datasets import load_boston
from sklearn import linear_model
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# Data:
boston = load_boston()
X = pd.DataFrame(boston.data) 
#boston.data contains the predictor variables (also known as features), 
#and is converted to a df and stored in the variable X
X.columns = boston.feature_names #labels the columns of the boston data set 
X.head(3)

Y = pd.DataFrame(boston.target) #Y value, what we want to predict, converted to a df and stored in Y 
Y.head(3)

# Train/Test sets, obtained from the X & Y variables 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, shuffle = True)

# Model
regr = linear_model.LinearRegression() #initiates the linear regression model 
regr.fit(X_train, Y_train) #incorporates the X and Y train sets into the model 

regr.coef_ 
regr.intercept_
coefs = pd.DataFrame()
coefs["Variables"] = X_train.columns
coefs["Coeff"] = regr.coef_[0]
print(coefs)


# Predictions:
predictions = regr.predict(X_test)

# Accuracy of the model:
# Mode 1: with the numerical MAE and MSE
mean_squared_error(Y_test, predictions)
mean_absolute_error(Y_test, predictions)


# Mode 2: Plotting the quality of the model:
allinfo = Y_test.copy() #saves a copy of the Y_test data toallinfo 
allinfo = pd.DataFrame(allinfo) #converts to a df 
allinfo.columns = ["Reality"] #changes column name to 'Reality' 
allinfo.head(3)
allinfo["Prediction"] = predictions #makes a new column called 'Prediction' which uses the predictions obtained earlier as values 
allinfo.head(7)
allinfo["Error"] = allinfo.Reality - allinfo.Prediction #makes a new column called 'Error' which calculates the diff between reality and prediction 
allinfo.head(7)


# Plotting the predicted vs real values:
ggplot(aes(x="Prediction",y="Reality"),allinfo) + geom_point() + geom_abline(color="red",slope=1, intercept=0)


# Plotting the residuals and testing non-structure:
ggplot(aes(x="Error"), allinfo) + geom_histogram() 


