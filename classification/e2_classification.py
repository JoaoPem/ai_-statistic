import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('EMGsDataset.csv', delimiter=',')
X = data[:2, :].T
y = data[2, :].astype(int)

plt.figure(figsize=(10, 8))
colors = ['gray', 'blue', 'green', 'orange', 'red']
labels = ['Neutro', 'Sorriso', 'Sobrancelhas', 'Surpreso', 'Rabugento']

for class_id in range(1, 6):
    mask = y == class_id
    plt.scatter(X[mask, 0], X[mask, 1], c=colors[class_id-1], 
                label=labels[class_id-1], alpha=0.7, s=10)

plt.xlabel('Sensor 1 - Corrugador do Supercílio')
plt.ylabel('Sensor 2 - Zigomático Maior')
plt.title('Distribuição das Classes EMG')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
for class_id in range(1, 6):
    mask = y == class_id
    plt.hist(X[mask, 0], bins=50, alpha=0.5, color=colors[class_id-1], label=labels[class_id-1])
plt.xlabel('Valor do Sensor 1')
plt.ylabel('Frequência')
plt.legend()
plt.title('Distribuição do Sensor 1 por Classe')

plt.subplot(1, 2, 2)
for class_id in range(1, 6):
    mask = y == class_id
    plt.hist(X[mask, 1], bins=50, alpha=0.5, color=colors[class_id-1], label=labels[class_id-1])
plt.xlabel('Valor do Sensor 2')
plt.ylabel('Frequência')
plt.legend()
plt.title('Distribuição do Sensor 2 por Classe')

plt.tight_layout()
plt.show()