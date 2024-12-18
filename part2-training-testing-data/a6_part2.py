import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

'''
**********CREATE THE MODEL**********
'''

data = pd.read_csv("part2-training-testing-data/blood_pressure_data.csv")
x = data["Age"].values
y = data["Blood Pressure"].values

# Create your training and testing datasets:
xtrain, xtest, ytrain, ytest = train_test_split(x.reshape(-1, 1), y, test_size=0.2, random_state=42)


# Use reshape to turn the x values into 2D arrays:
# xtrain = xtrain.reshape(-1,1)

# Create the model
model = LinearRegression()
model.fit(x, y)

# Find the coefficient, bias, and r squared values. 
# Each should be a float and rounded to two decimal places. 
coefficient = round(float(model.coef_[0]), 2)  # Coefficient (slope)
bias = round(float(model.intercept_), 2)  # Intercept (bias)
r_squared = round(model.score(xtrain, ytrain), 2)  # R-squared value


# Print out the linear equation and r squared value:
print(f"Linear equation: y = {coefficient}x + {bias}")
print(f"R-squared value on training data: {r_squared}")

'''
**********TEST THE MODEL**********
'''
# reshape the xtest data into a 2D array
xtest = xtest.reshape(-1, 1)

# get the predicted y values for the xtest values - returns an array of the results
prediction = np.round(model.predict, 2)
# round the value in the np array to 2 decimal places


# Test the model by looping through all of the values in the xtest dataset
print("\nTesting Linear Model with Testing Data:")
for actual, predicted in zip(ytest, prediction):
    print(f"Actual: {actual}, Predicted: {predicted}")

'''
**********CREATE A VISUAL OF THE RESULTS**********
'''
plt.scatter(xtrain, ytrain, color='blue', label='Training Data')  # Plot training data
plt.scatter(xtest, ytest, color='green', label='Test Data')  # Plot test data
plt.plot(xtrain, model.predict(xtrain), color='red', label='Line of Best Fit')  # Plot the line of best fit
plt.xlabel("Age")
plt.ylabel("Blood Pressure")
plt.title("Blood Pressure vs Age (Training and Test Data)")
plt.legend()
plt.show()