# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Initialize parameters (slope m and intercept c) with small values (usually 0).

2.Choose learning rate (α) and number of iterations.

3.Calculate predicted values using the equation Y=mX+c.

4.Compute the error between actual values and predicted values.

5.Update parameters (m and c) using gradient descent formulas to reduce error.

6.Repeat the process until the error is minimized and output the final model. 

## Program:
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("Startup.csv")

X = data['R&D Spend'].values
y = data['Profit'].values

X = (X - X.mean()) / X.std()

m = 0
b = 0

learning_rate = 0.01
epochs = 1000
n = len(X)

for i in range(epochs):
    y_pred = m * X + b
    
    dm = (-2/n) * np.sum(X * (y - y_pred))
    db = (-2/n) * np.sum(y - y_pred)
    
    m = m - learning_rate * dm
    b = b - learning_rate * db

print("Slope (m):", m)
print("Intercept (b):", b)

y_pred = m * X + b

plt.scatter(X, y)
plt.plot(X, y_pred)

plt.xlabel("R&D Spend (Normalized)")
plt.ylabel("Profit")
plt.title("Gradient Descent on 50_Startups Dataset")

plt.show()


Developed by: IJAS J
RegisterNumber: 212225230102

```

## Output:
<img width="808" height="625" alt="Screenshot 2026-04-22 093424" src="https://github.com/user-attachments/assets/4e1b088d-c80c-4300-b751-7b934638f6b5" />



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
