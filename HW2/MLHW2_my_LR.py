import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd

deliveryData = pd.read_csv('Social_Network_Ads.csv')

X = deliveryData.iloc[:, [2, 3]].values
y = deliveryData.iloc[:, 4].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# X_test = sc.transform(X_test)

# label = np.array(y_train)
# index_0 = np.where(label == 0)
# plt.scatter(X_train[index_0, 0], X_train[index_0, 1], marker='x', color='b', label='0', s=15)
# index_1 = np.where(label == 1)
# plt.scatter(X_train[index_1, 0], X_train[index_1, 1], marker='o', color='r', label='1', s=15)
#
# plt.xlabel('X1')
# plt.ylabel('X2')
# plt.legend(loc='upper left')
# plt.show()


class logistic(object):
    def __init__(self):
        self.W = None

    def train(self, X, y, learn_rate=0.01, num_iters=10000):
        num_train, num_feature = X.shape
        # init the weight
        self.W = 0.01 * np.random.randn(num_feature, 1).reshape((-1, 1))
        loss = []

        for i in range(num_iters):
            error, dW = self.compute_loss(X, y)
            self.W += -learn_rate * dW

            loss.append(error)
            if i % 200 == 0:
                print('i=%d,error=%f' % (i, error))
        return loss

    def compute_loss(self, X, y):
        num_train = X.shape[0]
        h = self.output(X)
        loss = -np.sum((y * np.log(h) + (1 - y) * np.log((1 - h))))
        loss = loss / num_train

        dW = X.T.dot((h - y)) / num_train

        return loss, dW

    def output(self, X):
        g = np.dot(X, self.W)
        return self.sigmod(g)

    def sigmod(self, X):
        return 1 / (1 + np.exp(-X))

    def predict(self, X_test):
        h = self.output(X_test)
        y_pred = np.where(h >= 0.5, 1, 0)
        return y_pred


import matplotlib.pyplot as plt

y_train = y_train.reshape((-1, 1))
y_test = y_test.reshape((-1, 1))
# add the x0=1
one = np.ones((X_train.shape[0], 1))
X_train1 = np.hstack((one, X_train))
classify = logistic()
loss = classify.train(X_train1, y_train)
print(classify.W)

plt.plot(loss)
plt.xlabel('Iteration number')
plt.ylabel('Loss value')
plt.show()
#
# y_pred = classify.predict(X_test)
# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y_test, y_pred)
# print(cm)

# label = np.array(y_test)
# index_0 = np.where(label == 0)
# plt.scatter(X_test[index_0, 0], X_test[index_0, 1], marker='o', color='r', label='0', s=25)
# index_1 = np.where(label == 1)
# plt.scatter(X_test[index_1, 0], X_test[index_1, 1], marker='o', color='g', label='1', s=25)

# Train img
label = np.array(y_train)
index_0 = np.where(label == 0)
plt.scatter(X_train[index_0, 0], X_train[index_0, 1], marker='o', color='r', label='0', s=25)
index_1 = np.where(label == 1)
plt.scatter(X_train[index_1, 0], X_train[index_1, 1], marker='o', color='g', label='1', s=25)

# show the decision boundary
x1 = np.arange(-1.2, 2, 0.01)
x2 = (- classify.W[0] - classify.W[1] * x1) / classify.W[2]
plt.plot(x1, x2, color='black')

plt.xlim(-2.25, 2.25)
plt.ylim(-3.33, 5.25)
plt.title(' LOGISTIC(Training set)')
plt.xlabel(' Age')
plt.ylabel(' Estimated Salary')
# plt.legend(loc='upper left')
plt.legend()
plt.show()