import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix

dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
Y = dataset.iloc[:, 4].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
classifier = LDA(n_components=1)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# Analyse the model
cm = confusion_matrix(y_test, y_pred)

# Visualization
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min()-1, stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min()-1, stop=X_set[:, 1].max() + 1, step=0.01))
z = classifier.predict(np.c_[X1.ravel(),X2.ravel()]).reshape(X1.shape)
plt.contour(X1, X2, z, 3, colors='black', linewidths=1, linestyles='solid')
plt.xlim(-2.25, 2.25)
plt.ylim(-3.33, 5.25)
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                color=ListedColormap(('red', 'green'))(i), label=j)

plt.title(' SVM(Training set)')
plt.xlabel(' Age')
plt.ylabel(' Estimated Salary')
plt.legend()
plt.show()

X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min()-1, stop=X_set[:, 0].max()+1, step=0.01),
                     np.arange(start=X_set[:, 1].min()-1, stop=X_set[:, 1].max()+1, step=0.01))
z = classifier.predict(np.c_[X1.ravel(),X2.ravel()]).reshape(X1.shape)
plt.contour(X1, X2, z, 3, colors='black', linewidths=1, linestyles='solid')
plt.xlim(-2.25, 2.25)
plt.ylim(-3.33, 5.25)
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                color=ListedColormap(('red', 'green'))(i), label=j)

plt.title(' SVM(Test set)')
plt.xlabel(' Age')
plt.ylabel(' Estimated Salary')
plt.legend()
plt.show()

print(cm)