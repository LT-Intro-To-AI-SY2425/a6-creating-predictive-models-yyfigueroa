import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("part4-classification/suv_data.csv")
data['Gender'].replace(['Male','Female'],[0,1],inplace=True)

x = data[["Age", "EstimatedSalary", "Gender"]].values
y = data["Purchased"].values

# Step 1: Print the values for x and y
print(x)
print(y)

# Step 2: Standardize the data using StandardScaler, 
scaler = StandardScaler()

# Step 3: Transform the data
x_scaled = scaler.fit_transform(x)

# Step 4: Split the data into training and testing data
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)


# Step 5: Fit the data




# Step 6: Create a LogsiticRegression object and fit the data
model = linear_model.LogisticRegression().fit(x_train, y_train
# Step 7: Print the score to see the accuracy of the model
accuracy = log_reg.score(x_test, y_test)
print("Accuracy:", accuracy)
# Step 8: Print out the actual ytest values and predicted y values
# based on the xtest data
y_pred = log_reg.predict(x_test)

print("Actual y_test values:")
print(y_test)
print("Predicted y values:")
print(y_pred)