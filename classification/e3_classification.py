import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = np.loadtxt('EMGsDataset.csv', delimiter=',')
X = data[:2].T
y = data[2].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_mqo = LinearRegression()
model_mqo.fit(X_train, y_train)
y_pred_mqo = np.round(model_mqo.predict(X_test)).astype(int)
y_pred_mqo = np.clip(y_pred_mqo, 1, 5)
acc_mqo = accuracy_score(y_test, y_pred_mqo)

model_qda = QDA(reg_param=0.01)
model_qda.fit(X_train, y_train)
y_pred_qda = model_qda.predict(X_test)
acc_qda = accuracy_score(y_test, y_pred_qda)

model_lda = LDA()
model_lda.fit(X_train, y_train)
y_pred_lda = model_lda.predict(X_test)
acc_lda = accuracy_score(y_test, y_pred_lda)

classes = np.unique(y_train)
means = []
covs = []

for cls in classes:
    X_cls = X_train[y_train == cls]
    means.append(np.mean(X_cls, axis=0))
    cov = np.cov(X_cls.T)
    cov += np.eye(cov.shape[0]) * 1e-6
    covs.append(cov)

avg_cov = np.mean(covs, axis=0)
inv_avg_cov = np.linalg.inv(avg_cov)

scores = []
for x in X_test:
    s = [x @ inv_avg_cov @ m - 0.5 * m @ inv_avg_cov @ m for m in means]
    scores.append(np.argmax(s) + 1)
y_pred_aggr = np.array(scores)
acc_aggr = accuracy_score(y_test, y_pred_aggr)

model_qda_reg = QDA(reg_param=0.5)
model_qda_reg.fit(X_train, y_train)
y_pred_qda_reg = model_qda_reg.predict(X_test)
acc_qda_reg = accuracy_score(y_test, y_pred_qda_reg)

model_gnb = GaussianNB()
model_gnb.fit(X_train, y_train)
y_pred_gnb = model_gnb.predict(X_test)
acc_gnb = accuracy_score(y_test, y_pred_gnb)

labels = ['MQO', 'QDA', 'LDA', 'Aggregated', 'QDA Reg', 'Naive Bayes']
accuracies = [acc_mqo, acc_qda, acc_lda, acc_aggr, acc_qda_reg, acc_gnb]

plt.figure(figsize=(10, 5))
plt.bar(labels, accuracies, color='steelblue')
plt.ylim(0, 1)
plt.title('Model Accuracies')
plt.ylabel('Accuracy')
plt.show()
