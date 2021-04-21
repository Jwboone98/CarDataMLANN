import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


# Reads data from a csv file
car_data = pd.read_csv("carData.csv")

print("Car_Name unique: ", len(pd.unique(car_data['Car_Name'])))
print("Fuel_Type unique: ", len(pd.unique(car_data['Fuel_Type'])))
print("Seller_Type unique: ", len(pd.unique(car_data['Seller_Type'])))
print("Transmission unique: ", len(pd.unique(car_data['Transmission'])))

inputs = pd.get_dummies(car_data, columns=["Car_Name", "Fuel_Type", "Seller_Type", "Transmission"])
inputs.fillna(value=0, inplace=True)

outputs = inputs["Selling_Price"]
inputs.drop(labels="Selling_Price", axis=1, inplace=True)

print("Input columns length: ", len(inputs.columns))

test_size = 0.2
random_state = 0


X_train, X_test, y_train, y_test = train_test_split(inputs, outputs,
                                                    test_size=test_size, random_state=random_state)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

score = regressor.score(X_test, y_test)
print('Score: ', score)


def plot_comp():
    car = car_data.plot(x=None, y='Selling_Price', style='.', color='b', figsize=(30, 5))
    pred = plt.plot(regressor.predict(inputs), color='r', ls='', marker='.', label='Predicted Selling Price')
    car.legend()
    # plt.xticks(np.arange(0, 301, 5))
    plt.title(label="Linear Regression")
    plt.xlabel('Car')
    plt.ylabel('Price in thousands')
    plt.show()


plot_comp()




def plot_split_data(xtest, ytest, title, txtstr):
    fig = plt.figure()
    ytest = ytest.reset_index(drop=True)
    splot = ytest.plot(x=None, y='Selling_Price', style='.', color='b', figsize=(10, 5))
    plt.plot(regressor.predict(xtest), color='r', ls='', marker='.', label='Predicted Selling Price')
    plt.title(label=title)
    splot.legend()
    plt.xlabel('Car')
    plt.ylabel('Price in thousands')
    fig.text(0.01, 0.95, txtstr, fontsize=8)
    plt.xlabel('Car')
    plt.ylabel('Price in thousands')
    plt.show()


# txtstr = f"Score:  {score}"
# plot_split_data(X_test, y_test, 'Linear Regression Testing Data Comparison', txtstr)
# plot_split_data(X_train, y_train, 'Linear Regression Training Data Comparison', txtstr)


def format_df(df):
    y_new_inputs = pd.DataFrame({'Car_Name': [], 'Year': [], 'Present_Price': [],
                                 'Kms_Driven': [], 'Fuel_Type': [], 'Seller_Type': [],
                                 'Transmission': [], 'Owner': []})

    y_new_inputs = pd.concat([y_new_inputs, df], axis=0)

    y_new_inputs = pd.get_dummies(y_new_inputs, columns=["Car_Name", "Fuel_Type", "Seller_Type", "Transmission"])
    y_new_inputs.fillna(value=0.0, inplace=True)

    missing_cols = set(inputs.columns) - set(y_new_inputs.columns)

    for i in missing_cols:
        y_new_inputs[i] = 0

    y_new_inputs = y_new_inputs[inputs.columns]
    return y_new_inputs


def plot_sp_cn(data, car_name):
    rows = data.loc[data['Car_Name'] == car_name]
    rows = rows.reset_index(drop=True)
    rows_input = rows.copy()
    rows_input.drop(labels="Selling_Price", axis=1, inplace=True)
    rows_inputs = format_df(rows_input)
    print(rows)

    rows.plot(x=None, y='Selling_Price', style='.')
    plt.plot(regressor.predict(rows_inputs), color='k', ls='', marker='.')
    plt.xlabel('Car')
    plt.ylabel('Price in thousands')
    plt.show()


#plot_sp_cn(car_data, 'sx4')
#plot_sp_cn(car_data, 'Royal Enfield Classic 350')


def error_outputs():
    y_pred = regressor.predict(X_test)
    print('R-squared Error: ', metrics.r2_score(y_test, y_pred))
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


def inline_input(y_user_inputs):
    y_new_inputs = pd.DataFrame({'Car_Name': [], 'Year': [], 'Present_Price': [],
                                 'Kms_Driven': [], 'Fuel_Type': [], 'Seller_Type': [],
                                 'Transmission': [], 'Owner': []})

    y_new_inputs = pd.concat([y_new_inputs, y_user_inputs], axis=0)

    y_new_inputs = pd.get_dummies(y_new_inputs, columns=["Car_Name", "Fuel_Type", "Seller_Type", "Transmission"])
    y_new_inputs.fillna(value=0.0, inplace=True)

    missing_cols = set(inputs.columns) - set(y_new_inputs.columns)

    for i in missing_cols:
        y_new_inputs[i] = 0

    y_new_inputs = y_new_inputs[inputs.columns]

    y_new_inputs_pred = regressor.predict(y_new_inputs)
    return y_new_inputs_pred


def user_input(in_data):
    y_new_inputs = pd.DataFrame({'Car_Name': [], 'Year': [], 'Present_Price': [],
                                 'Kms_Driven': [], 'Fuel_Type': [], 'Seller_Type': [],
                                 'Transmission': [], 'Owner': []})

    y_new_inputs = pd.concat([y_new_inputs, in_data], axis=0)

    y_new_inputs = pd.get_dummies(y_new_inputs, columns=["Car_Name", "Fuel_Type", "Seller_Type", "Transmission"])
    y_new_inputs.fillna(value=0.0, inplace=True)

    missing_cols = set(inputs.columns) - set(y_new_inputs.columns)

    for i in missing_cols:
        y_new_inputs[i] = 0

    y_new_inputs = y_new_inputs[inputs.columns]

    new_output = regressor.predict(y_new_inputs)

    return new_output


y_newInputs_0 = pd.DataFrame({'Car_Name': ['sx4'], 'Year': [2014], 'Present_Price': [12],
                             'Kms_Driven': [45000], 'Fuel_Type': ['Petrol'], 'Seller_Type': ['Dealer'],
                              'Transmission': ['Manual'], 'Owner': [0]})


car_name = input("Car Name:")
year = input("Year:")
present_price = input("Present Price:")
kms = input("Kms Driven:")
fuel = input("Fuel Type:")
seller = input("Seller Type:")
transmission = input("Transmission:")
owner = input("Owner:")

user_data = pd.DataFrame({'Car_Name': [car_name], 'Year': [year], 'Present_Price': [present_price],
                              'Kms_Driven': [kms], 'Fuel_Type': [fuel], 'Seller_Type': [seller],
                              'Transmission': [transmission], 'Owner': [owner]})

print("User Prediction: ", user_input(user_data))
print("Inline prediction:", inline_input(y_newInputs_0))
