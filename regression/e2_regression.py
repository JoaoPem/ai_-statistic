import numpy as np

data = np.genfromtxt('atividade_enzimatica.csv', delimiter=',', skip_header=1)
X = data[:, :-1]
y = data[:, -1].reshape(-1, 1)

print("Shape of X:", X.shape)
print("Shape of y:", y.shape)
print("\nFirst 5 rows of X:")
print(X[:5])
print("\nFirst 5 rows of y:")
print(y[:5])