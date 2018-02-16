import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from gnb import GNB

class LR(GNB):

    def sigmoid_fun(self, z):
        return 1 / (1 + np.exp(-z))

    def train(self, X_train, y_train, learning_rate=0.8):
        X_train = X_train.T
        y_train = y_train.T
        m = X_train.shape[1]
        n = X_train.shape[0]
        w = np.zeros((n,1))
        b = 0

        for i in range(500):
            z = np.dot(w.T,X_train)+b
            a = self.sigmoid_fun(z)
            dz = a - y_train
            db = np.sum(dz) / m
            dw = np.matmul(X_train, dz.T) / m

            w = w - learning_rate * dw
            b = b - learning_rate * db

        return [w, b]

    def predict(self, X_test, y_test, learnt_params):
        X_test = X_test.T
        y_test = y_test.T
        m = X_test.shape[1]
        n = X_test.shape[0]
        w = learnt_params[0]
        b = learnt_params[1]

        z = np.dot(w.T, X_test) + b
        a = self.sigmoid_fun(z)
        y_pred = (a > 0.5) * 1.0

        accuracy = np.sum(y_test == y_pred, axis=1) / m
        return accuracy

if __name__ == "__main__":
    obj1 = GNB()
    plot_curve_GNB_dict = obj1.run()
    obj2 = LR()
    plot_curve_LR_dict = obj2.run()
    x, y_gnb = zip(*sorted(plot_curve_GNB_dict.items()))
    x, y_lr = zip(*sorted(plot_curve_LR_dict.items()))
    plt.plot(x, y_gnb, 'r')
    plt.plot(x, y_lr, 'b')
    plt.ylabel('accuracy')
    plt.xlabel('size_of_dataset')
    plt.show()