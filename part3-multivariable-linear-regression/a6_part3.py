import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#imports and formats the data
data = pd.read_csv("part3-multivariable-linear-regression/car_data.csv")
x = data[["miles","age"]].values
y = data["Price"].values


#split the data into training and testing data
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)


#create linear regression model
model = LinearRegression()


#Find and print the coefficients, intercept, and r squared values. 
#Each should be rounded to two decimal places. 
coefficients = np.round(model.coef_, 2)  
intercept = round(model.intercept_, 2)  
r_squared = round(model.score(xtrain, ytrain), 2)  

#Loop through the data and print out the predicted prices and the 
#actual prices
print("***************")
print("Testing Results")