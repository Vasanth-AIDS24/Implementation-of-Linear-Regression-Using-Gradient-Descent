# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm 
1. Load and preprocess dataset by selecting features and target values, converting to float arrays.
2. Standardize features and target using StandardScaler..
3. Train linear regression model using gradient descent to optimize weights (theta).
4. Standardize new input data and predict output using learned weights.
5. Inverse-transform the prediction to original scale and display result.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: VASANTH P
RegisterNumber:  212224230295
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate = 0.1, num_iters = 1000):
    X = np.c_[np.ones(len(X1)),X1]
    theta = np.zeros(X.shape[1]).reshape(-1,1)
    
    for _ in range(num_iters):
        predictions = (X).dot(theta).reshape(-1,1)
        errors=(predictions - y ).reshape(-1,1)
        theta -= learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
data=pd.read_csv("50_Startups.csv")
data.head()
X=(data.iloc[1:,:-2].values)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X)
print(X1_Scaled)
theta= linear_regression(X1_Scaled,Y1_Scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")
*/
```

## Output:
DATA INFORMATION
<img width="763" height="264" alt="image" src="https://github.com/user-attachments/assets/94672ee8-a1fa-49c5-a5e0-b77805a972c4" />

VALUE OF X:
<img width="243" height="715" alt="image" src="https://github.com/user-attachments/assets/fd87d6d7-e7a1-4128-bfac-081e1ada6083" />

VALUE OF X1_SCALED:
<img width="352" height="715" alt="image" src="https://github.com/user-attachments/assets/b9dcf9ea-2880-4de2-b441-def7f15b1864" />

PREDICTED VALUE:

<img width="345" height="67" alt="image" src="https://github.com/user-attachments/assets/123e3ae7-9e8c-4dea-9079-87e05bb6b497" />

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
