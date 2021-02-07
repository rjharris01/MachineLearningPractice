# Importing the libraries
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# Encode the Independent catagorical data
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Split the data set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

np.set_printoptions(precision=2)
results = np.concatenate(
    (y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1)

accuracy = 0
for row in results:
    accuracy = accuracy + row[1]/row[0]*100
accuracy = accuracy / len(results)
print(accuracy - 100)
print(results)

coef = regressor.coef_
interept = regressor.intercept_

print(coef)


print("Profit= {0:.2f} × Dummy State 1 + {1:.2f} × Dummy State 2 + {2:.2f} × Dummy State 3 + {3:.2f} × R&D Spend + {4:.2f} × Administration + {5:.2f} × Marketing Spend + {6:.2f}".format(
    coef[0], coef[1], coef[2], coef[3], coef[4], coef[5], interept))
