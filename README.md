# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required library and read the dataframe.
2.Write a function computeCost to generate the cost function.
3.Perform iterations og gradient steps with learning rate.
4.Plot the Cost function using Gradient Descent and generate the required graph.
## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: SOWBARNIKA M P
RegisterNumber:  212225230271

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
data = pd.read_csv("exp_3_50_Startups.csv")
print("First 5 rows of the dataset:")
display(data.head())
data_encoded = pd.get_dummies(data, columns=['State'], drop_first=True)
print("Encoded dataset:")
display(data_encoded.head())
X_raw = data_encoded.drop('Profit', axis=1).values
y_raw = data_encoded['Profit'].values.reshape(-1, 1)
print("Shape of features:", X_raw.shape)
print("Shape of target:", y_raw.shape)
X_scaler = StandardScaler()
y_scaler = StandardScaler()
X = X_scaler.fit_transform(X_raw)
y = y_scaler.fit_transform(y_raw)
m = X.shape[0]
X = np.hstack((np.ones((m, 1)), X))
print("Shape of X after adding bias term:", X.shape)
def compute_cost(X, y, theta):
    """
    Compute Mean Squared Error cost.
    """
    m = len(y)
    preds = X.dot(theta)
    cost = (1 / (2 * m)) * np.sum((preds - y) ** 2)
    return cost
def gradient_descent(X, y, learning_rate=0.01, num_iters=2000, tol=1e-8, verbose=False):
    """
    Batch Gradient Descent for Linear Regression.
    """
    m, n = X.shape
    theta = np.zeros((n, 1))
    J_history = []

    prev_cost = compute_cost(X, y, theta)

    for i in range(num_iters):
        preds = X.dot(theta)
        errors = preds - y
        grad = (1 / m) * (X.T.dot(errors))
        theta -= learning_rate * grad

        cost = compute_cost(X, y, theta)
        J_history.append(cost)

        if abs(prev_cost - cost) < tol:
            if verbose:
                print(f"Converged at iteration {i}")
            break
        prev_cost = cost

        if verbose and (i % 500 == 0 or i < 5):
            print(f"Iteration {i:4d}, Cost: {cost:.6f}")

    return theta, J_history
alpha = 0.01
theta, J_hist = gradient_descent(X, y, learning_rate=alpha, num_iters=5000, tol=1e-9, verbose=True)

print("\nLearned Parameters (Theta):")
print(theta.flatten())
plt.figure(figsize=(7,4))
plt.plot(J_hist)
plt.xlabel('Iterations')
plt.ylabel('Cost (MSE/2)')
plt.title('Cost Function Convergence')
plt.grid(True)
plt.show()
# Create new sample (same feature names as original)
new_sample = pd.DataFrame([{
    'R&D Spend': 165349.2,
    'Administration': 136897.8,
    'Marketing Spend': 471784.1,
    'State': 'New York'
}])

# Apply one-hot encoding
new_encoded = pd.get_dummies(new_sample, columns=['State'], drop_first=True)

# Align columns with training data (add missing ones as 0)
new_encoded = new_encoded.reindex(columns=data_encoded.drop('Profit', axis=1).columns, fill_value=0)

# Scale new data
new_scaled = X_scaler.transform(new_encoded)

# Add bias
new_design = np.hstack((np.ones((new_scaled.shape[0], 1)), new_scaled))

# Predict (scaled)
scaled_pred = new_design.dot(theta)

# Inverse transform to original units
pred_original = y_scaler.inverse_transform(scaled_pred)

print(f"\nPredicted Profit for the new startup: ₹{pred_original[0][0]:,.2f}")
*/
```

## Output:

<img width="1012" height="390" alt="image" src="https://github.com/user-attachments/assets/000eaab8-fcfb-40fc-97ff-50844f73ffd1" />

<img width="1070" height="418" alt="image" src="https://github.com/user-attachments/assets/78ae3d1e-823f-43dd-9a91-5549c07728d4" />

<img width="1034" height="394" alt="image" src="https://github.com/user-attachments/assets/830447f5-e2ed-4e57-8443-d66b641eb738" />

<img width="1098" height="525" alt="image" src="https://github.com/user-attachments/assets/3050efde-418a-4002-b141-c50626359ce7" />

<img width="1128" height="233" alt="image" src="https://github.com/user-attachments/assets/e5ed769d-9634-4595-bd1e-8c0635b2323c" />



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
