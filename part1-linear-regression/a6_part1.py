import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv("part1-linear-regression/blood_pressure_data.csv")
x = data["Age"].values
y = data["Blood Pressure"].values

# Use reshape to turn the x values into 2D arrays:
x = x.reshape(-1,1)

# Create the model
model = LinearRegression()
model.fit(x, y)
# Find the coefficient, bias, and r squared values. 
coefficient = round(model.coef_[0], 2)
bias = round(model.intercept_, 2)
r_squared = round(model.score(x, y), 2)
# Each should be a float and rounded to two decimal places. 
# Print out the linear equation and r squared value
print(f"Linear equation: y = {coefficient}x + {bias}")
print(f"R-squared value: {r_squared}")


# Predict the the blood pressure of someone who is 43 years old.
# Print out the prediction
prediction = model.predict(np.array([[43]]))
print(f"Predicted blood pressure for a 43-year-old: {prediction[0]:.2f}")


# Create the model in matplotlib and include the line of best fit
plt.scatter(x, y, color='blue', label='Data Points')  # scatter plot of the data
plt.plot(x, model.predict(x), color='red', label='Line of Best Fit')  # line of best fit
plt.xlabel("Age")
plt.ylabel("Blood Pressure")
plt.title("Blood Pressure vs Age")
plt.legend()
plt.show()
