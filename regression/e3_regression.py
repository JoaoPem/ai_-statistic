import numpy as np

data = np.genfromtxt('atividade_enzimatica.csv', delimiter=',', skip_header=1)
X = data[:, :-1]
y = data[:, -1].reshape(-1, 1)

X = np.column_stack((np.ones(X.shape[0]), X))

split_idx = int(0.8 * X.shape[0])
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# 1. Ordinary Least Squares
theta_ols = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train

# 2. Regularized OLS
lambda_reg = 0.1
I = np.identity(X_train.shape[1])
I[0, 0] = 0
theta_ridge = np.linalg.inv(X_train.T @ X_train + lambda_reg * I) @ X_train.T @ y_train

# 3. Mean Model
mean_model = np.mean(y_train) * np.ones_like(y_test)

# Make predictions
y_pred_ols = X_test @ theta_ols
y_pred_ridge = X_test @ theta_ridge

# Calculate MSEs
mse_ols = np.mean((y_test - y_pred_ols)**2)
mse_ridge = np.mean((y_test - y_pred_ridge)**2)
mse_mean = np.mean((y_test - mean_model)**2)

print(f"OLS MSE: {mse_ols:.4f}")
print(f"Ridge MSE: {mse_ridge:.4f}")
print(f"Mean Model MSE: {mse_mean:.4f}")