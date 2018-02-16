import pandas as pd
import numpy as np

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
    obj = LR()
    X, y, df = obj.load_file()
    sets = obj.k_folds_cross_validation(X, y)
    for set in sets:
        learnt_params = obj.train(set[0], set[2])
        accuracy = obj.predict(set[1], set[3], learnt_params)
        print("Accuracy: ", accuracy)
        print(learnt_params)

    print("All set to code!")