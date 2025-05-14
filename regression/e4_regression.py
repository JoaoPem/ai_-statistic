import numpy as np

data = np.genfromtxt('atividade_enzimatica.csv', delimiter=',', skip_header=1)
X = data[:, :-1]
y = data[:, -1].reshape(-1, 1)

X = np.column_stack((np.ones(X.shape[0]), X))

split_idx = int(0.8 * X.shape[0])
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

lambdas = [0, 0.25, 0.5, 0.75, 1]

results = []

for lambda_val in lambdas:
    I = np.identity(X_train.shape[1])
    I[0, 0] = 0
    
    theta = np.linalg.inv(X_train.T @ X_train + lambda_val * I) @ X_train.T @ y_train
    
    y_pred = X_test @ theta
    mse = np.mean((y_test - y_pred)**2)
    
    results.append({
        'lambda': lambda_val,
        'theta': theta.flatten(),
        'mse': mse
    })

print("Regularized Model Results:")
print("Lambda\t\tMSE\t\tTheta (intercept, temp, pH)")
for res in results:
    print(f"{res['lambda']:.2f}\t\t{res['mse']:.4f}\t\t{np.round(res['theta'], 4)}")

best = min(results, key=lambda x: x['mse'])
print(f"\nBest lambda: {best['lambda']} with MSE: {best['mse']:.4f}")