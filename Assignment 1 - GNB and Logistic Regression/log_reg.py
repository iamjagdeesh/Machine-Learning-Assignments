import numpy as np
import matplotlib.pyplot as plt

from gnb import GNB

class LR(GNB):
    def __init__(self, algo="Logistic Regression"):
        self.algo = algo

    def sigmoid_fun(self, z):

        return 1 / (1 + np.exp(-z))

    def train(self, X_train, y_train, learning_rate=0.8):
        X_train = X_train.T
        y_train = y_train.T
        m = X_train.shape[1]
        n = X_train.shape[0]
        w = np.zeros((n,1))
        w_0 = 0
        for i in range(500):
            z = np.dot(w.T,X_train)+w_0
            p_y_x_cap = self.sigmoid_fun(z)
            diff = y_train - p_y_x_cap
            dw_0 = np.sum(diff) / m
            dw = np.matmul(X_train, diff.T) / m
            w = w + learning_rate * dw
            w_0 = w_0 + learning_rate * dw_0

        return [w, w_0]

    def predict(self, X_test, y_test, learnt_params):
        X_test = X_test.T
        y_test = y_test.T
        m = X_test.shape[1]
        n = X_test.shape[0]
        w = learnt_params[0]
        w_0 = learnt_params[1]
        z = np.dot(w.T, X_test) + w_0
        p_y_x = self.sigmoid_fun(z)
        y_pred = (p_y_x > 0.5) * 1.0
        accuracy = np.sum(y_test == y_pred, axis=1) / float(m)

        return accuracy

if __name__ == "__main__":
    obj1 = GNB()
    plot_curve_GNB_dict = obj1.run()
    obj2 = LR()
    plot_curve_LR_dict = obj2.run()
    x, y_gnb = zip(*sorted(plot_curve_GNB_dict.items()))
    x, y_lr = zip(*sorted(plot_curve_LR_dict.items()))
    fig = plt.figure()
    plt.title('Accuracy vs Size of data curve')
    plt.plot(x, y_gnb, 'r', label='Gaussian Naive Bayes')
    plt.plot(x, y_lr, 'b', label='Logistic Regression')
    plt.ylabel('accuracy')
    plt.xlabel('size_of_dataset')
    plt.legend()
    plt.show()
    fig.savefig('Gaussian NB and Logistic Regression combined.png')