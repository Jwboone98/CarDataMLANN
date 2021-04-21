import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# Reads data from a csv file
car_data = pd.read_csv("carData_numbered.csv")

outputs = car_data["Selling_Price"]
car_data.drop(labels="Selling_Price", axis=1, inplace=True)

test_size = 0.2
random_state = 0

X_train, X_test, y_train, y_test = train_test_split(car_data, outputs, test_size=test_size, random_state=random_state)

scalar = StandardScaler()
scalar.fit(X_train)

StandardScaler(copy=True, with_mean=True, with_std=True)

X_train = scalar.transform(X_train)
X_test = scalar.transform(X_test)

mlp_regressor = MLPRegressor(hidden_layer_sizes=(13, 13, 13), max_iter=5000)

mlp_regressor.fit(X_train, y_train)

score = mlp_regressor.score(X_test, y_test)
print('Score:', score)
