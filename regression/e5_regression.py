import numpy as np

data = np.genfromtxt('atividade_enzimatica.csv', delimiter=',', skip_header=1)
X = data[:, :-1]
y = data[:, -1]

R = 500
test_size = 0.2
lambdas = [0.25, 0.5, 0.75, 1]

rss_results = {
    'OLS': [],
    'lambda_0.25': [],
    'lambda_0.5': [],
    'lambda_0.75': [],
    'lambda_1': [],
    'Mean': []
}

for _ in range(R):
    shuffle_idx = np.random.permutation(len(X))
    X_shuffled = X[shuffle_idx]
    y_shuffled = y[shuffle_idx]
    
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X_shuffled[:split_idx], X_shuffled[split_idx:]
    y_train, y_test = y_shuffled[:split_idx], y_shuffled[split_idx:]
    
    X_train_int = np.column_stack((np.ones(len(X_train)), X_train))
    X_test_int = np.column_stack((np.ones(len(X_test)), X_test))
    
    # 1. OLS Model
    theta_ols = np.linalg.inv(X_train_int.T @ X_train_int) @ X_train_int.T @ y_train
    y_pred_ols = X_test_int @ theta_ols
    rss_ols = np.sum((y_test - y_pred_ols)**2)
    rss_results['OLS'].append(rss_ols)
    
    # 2. Regularized Models
    I = np.identity(X_train_int.shape[1])
    I[0, 0] = 0
    
    for lam in lambdas:
        theta_ridge = np.linalg.inv(X_train_int.T @ X_train_int + lam * I) @ X_train_int.T @ y_train
        y_pred_ridge = X_test_int @ theta_ridge
        rss_ridge = np.sum((y_test - y_pred_ridge)**2)
        rss_results[f'lambda_{lam}'].append(rss_ridge)

    mean_pred = np.mean(y_train) * np.ones_like(y_test)
    rss_mean = np.sum((y_test - mean_pred)**2)
    rss_results['Mean'].append(rss_mean)

print("Average RSS across 500 Monte Carlo simulations:")
for model in rss_results:
    avg_rss = np.mean(rss_results[model])
    print(f"{model:10}: {avg_rss:.2f}")

best_model = min(rss_results, key=lambda x: np.mean(rss_results[x]))
print(f"\nBest performing model: {best_model}")