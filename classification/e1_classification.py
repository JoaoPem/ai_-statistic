import numpy as np

data = np.genfromtxt('EMGsDataset.csv', delimiter=',')

# For MQO (Ordinary Least Squares) models:
X_mqo = data[:2, :].T
y_mqo = data[2, :]

C = 5
Y_mqo = np.zeros((X_mqo.shape[0], C))
Y_mqo[np.arange(X_mqo.shape[0]), y_mqo.astype(int)-1] = 1

# For Gaussian Bayesian models:
X_bayes = data[:2, :]
y_bayes = data[2, :]

Y_bayes = np.zeros((C, X_bayes.shape[1]))
Y_bayes[y_bayes.astype(int)-1, np.arange(X_bayes.shape[1])] = 1

print("MQO format:")
print(f"X_mqo shape: {X_mqo.shape} (should be N×p: 50000×2)")
print(f"Y_mqo shape: {Y_mqo.shape} (should be N×C: 50000×5)\n")

print("Bayesian format:")
print(f"X_bayes shape: {X_bayes.shape} (should be p×N: 2×50000)")
print(f"Y_bayes shape: {Y_bayes.shape} (should be C×N: 5×50000)")

print("\nFirst 5 samples (MQO format):")
print("X_mqo:\n", X_mqo[:5])
print("Y_mqo:\n", Y_mqo[:5])

print("\nFirst 5 samples (Bayesian format - transposed for display):")
print("X_bayes.T:\n", X_bayes[:, :5].T)
print("Y_bayes.T:\n", Y_bayes[:, :5].T)