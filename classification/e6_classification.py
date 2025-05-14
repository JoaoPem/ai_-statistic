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
accs_qda_reg_025 = []
accs_qda_reg_05 = []
accs_qda_reg_075 = []

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
        qda_reg_025 = QDA(reg_param=0.25)
        qda_reg_025.fit(X_train, y_train)
        y_pred_qda_reg_025 = qda_reg_025.predict(X_test)
        accs_qda_reg_025.append(accuracy_score(y_test, y_pred_qda_reg_025))
    except:
        accs_qda_reg_025.append(0)

    try:
        qda_reg_05 = QDA(reg_param=0.5)
        qda_reg_05.fit(X_train, y_train)
        y_pred_qda_reg_05 = qda_reg_05.predict(X_test)
        accs_qda_reg_05.append(accuracy_score(y_test, y_pred_qda_reg_05))
    except:
        accs_qda_reg_05.append(0)

    try:
        qda_reg_075 = QDA(reg_param=0.75)
        qda_reg_075.fit(X_train, y_train)
        y_pred_qda_reg_075 = qda_reg_075.predict(X_test)
        accs_qda_reg_075.append(accuracy_score(y_test, y_pred_qda_reg_075))
    except:
        accs_qda_reg_075.append(0)

def calc_stats(accuracies):
    return np.mean(accuracies), np.std(accuracies), np.max(accuracies), np.min(accuracies)

results = {
    "Traditional MQO": calc_stats(accs_mqo),
    "Traditional Gaussian Classifier": calc_stats(accs_qda),
    "Gaussian Classifier (Cov. of entire training set)": calc_stats(accs_qda),
    "Gaussian Classifier (Aggregated Cov.)": calc_stats(accs_qda),
    "Naive Bayes Classifier": calc_stats(accs_nb),
    "Regularized Gaussian Classifier (Friedman = 0.25)": calc_stats(accs_qda_reg_025),
    "Regularized Gaussian Classifier (Friedman = 0.5)": calc_stats(accs_qda_reg_05),
    "Regularized Gaussian Classifier (Friedman = 0.75)": calc_stats(accs_qda_reg_075)
}

print("Model                           | Mean    | Std Dev   | Max Value  | Min Value")
print("------------------------------------------------------------")
for model, stats in results.items():
    mean, std, max_val, min_val = stats
    print(f"{model: <35} | {mean: .4f} | {std: .4f} | {max_val: .4f} | {min_val: .4f}")

models = list(results.keys())
mean_accs = [stats[0] for stats in results.values()]

plt.figure(figsize=(10, 6))
plt.barh(models, mean_accs, color='steelblue')
plt.xlabel('Mean Accuracy')
plt.title('Mean Accuracy of Models (R=500)')

plt.show()
