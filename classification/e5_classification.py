import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

data = np.loadtxt('EMGsDataset.csv', delimiter=',')
X = data[:2].T
y = data[2].astype(int)

R = 500
test_size = 0.2
random_state = 42

accs_mqo = []
accs_qda = []
accs_lda = []
accs_nb = []
accs_qda_reg = []

for i in range(R):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    mqo = LinearRegression()
    mqo.fit(X_train, y_train)
    y_pred_mqo = np.round(mqo.predict(X_test)).astype(int)
    y_pred_mqo = np.clip(y_pred_mqo, 1, 5)
    accs_mqo.append(accuracy_score(y_test, y_pred_mqo))

    try:
        qda = QDA(reg_param=0.0)
        qda.fit(X_train, y_train)
        y_pred_qda = qda.predict(X_test)
        accs_qda.append(accuracy_score(y_test, y_pred_qda))
    except:
        accs_qda.append(0)

    lda = LDA()
    lda.fit(X_train, y_train)
    y_pred_lda = lda.predict(X_test)
    accs_lda.append(accuracy_score(y_test, y_pred_lda))

    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_pred_nb = nb.predict(X_test)
    accs_nb.append(accuracy_score(y_test, y_pred_nb))

    try:
        qda_reg = QDA(reg_param=0.25)
        qda_reg.fit(X_train, y_train)
        y_pred_qda_reg = qda_reg.predict(X_test)
        accs_qda_reg.append(accuracy_score(y_test, y_pred_qda_reg))
    except:
        accs_qda_reg.append(0)

models = ['MQO', 'QDA', 'LDA', 'Naive Bayes', 'QDA Reg λ=0.25']
mean_accs = [
    np.mean(accs_mqo),
    np.mean(accs_qda),
    np.mean(accs_lda),
    np.mean(accs_nb),
    np.mean(accs_qda_reg)
]

plt.figure(figsize=(10, 5))
plt.bar(models, mean_accs, color='steelblue')
plt.title('Monte Carlo (R=500) - Average Accuracy of Models')
plt.ylabel('Average Accuracy')
plt.ylim(0, 1)
plt.grid(axis='y')
plt.show()
