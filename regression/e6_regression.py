import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('atividade_enzimatica.csv', delimiter=',', skip_header=1)
X = data[:, :-1]
y = data[:, -1]

R = 500
test_size = 0.2
lambdas = [0.25, 0.5, 0.75, 1]
model_names = ['Mean', 'OLS', 'λ=0.25', 'λ=0.5', 'λ=0.75', 'λ=1']

rss_results = np.zeros((R, len(model_names)))

np.random.seed(42)

for i in range(R):
    shuffle_idx = np.random.permutation(len(X))
    X_shuffled = X[shuffle_idx]
    y_shuffled = y[shuffle_idx]
    
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X_shuffled[:split_idx], X_shuffled[split_idx:]
    y_train, y_test = y_shuffled[:split_idx], y_shuffled[split_idx:]
    
    X_train_int = np.column_stack((np.ones(len(X_train)), X_train))
    X_test_int = np.column_stack((np.ones(len(X_test)), X_test))
    
    # 1. Mean Model
    mean_pred = np.mean(y_train)
    rss_results[i, 0] = np.sum((y_test - mean_pred)**2)
    
    # 2. OLS Model
    theta_ols = np.linalg.pinv(X_train_int.T @ X_train_int) @ X_train_int.T @ y_train
    y_pred_ols = X_test_int @ theta_ols
    rss_results[i, 1] = np.sum((y_test - y_pred_ols)**2)
    
    # 3. Regularized Models
    I = np.eye(X_train_int.shape[1])
    I[0, 0] = 0
    
    for j, lam in enumerate(lambdas, start=2):
        reg_matrix = X_train_int.T @ X_train_int + lam * I
        theta_ridge = np.linalg.pinv(reg_matrix) @ X_train_int.T @ y_train
        y_pred_ridge = X_test_int @ theta_ridge
        rss_results[i, j] = np.sum((y_test - y_pred_ridge)**2)

# Calculate statistics
stats = np.zeros((len(model_names), 4))
stats[:, 0] = np.mean(rss_results, axis=0)  # Mean
stats[:, 1] = np.std(rss_results, axis=0)   # Std
stats[:, 2] = np.max(rss_results, axis=0)   # Max
stats[:, 3] = np.min(rss_results, axis=0)   # Min

print("Modelo\t\tMédia\t\tDesvio-Padrão\tMaior\t\tMenor")
for name, (mean, std, max_, min_) in zip(model_names, stats):
    print(f"{name:8}\t{mean:.2f}\t\t{std:.2f}\t\t{max_:.2f}\t{min_:.2f}")

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.boxplot(rss_results, tick_labels=model_names)
plt.title('Distribuição do RSS por Modelo')
plt.ylabel('RSS')
plt.xticks(rotation=45)
plt.grid(True)

# Mean comparison
plt.subplot(1, 2, 2)
x_pos = np.arange(len(model_names))
plt.bar(x_pos, stats[:, 0], yerr=stats[:, 1], capsize=5)
plt.xticks(x_pos, model_names, rotation=45)
plt.title('Comparação das Médias de RSS')
plt.ylabel('Média RSS')
plt.grid(True)

plt.tight_layout()
plt.show()

print("\nAverage coefficients for each model:")
X_full = np.column_stack((np.ones(len(X)), X))
for name, idx in zip(model_names[1:], range(1, len(model_names))):
    if idx == 1:
        theta = np.linalg.pinv(X_full.T @ X_full) @ X_full.T @ y
    else:
        I = np.eye(X_full.shape[1])
        I[0, 0] = 0
        lam = lambdas[idx-2]
        theta = np.linalg.pinv(X_full.T @ X_full + lam * I) @ X_full.T @ y
    print(f"{name}: {theta}")