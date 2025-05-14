import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('atividade_enzimatica.csv', delimiter=',', skip_header=1)
temperature = data[:, 0]
ph = data[:, 1]
activity = data[:, 2]

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(temperature, activity, c=activity, cmap='viridis')
plt.colorbar(label='Enzymatic Activity')
plt.xlabel('Temperature')
plt.ylabel('Activity')
plt.title('Temperature vs Activity')

plt.subplot(1, 2, 2)
plt.scatter(ph, activity, c=activity, cmap='viridis')
plt.colorbar(label='Enzymatic Activity')
plt.xlabel('pH')
plt.ylabel('Activity')
plt.title('pH vs Activity')

plt.tight_layout()
plt.show()