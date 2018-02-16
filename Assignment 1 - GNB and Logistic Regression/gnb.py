import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class GNB(object):

    def k_folds_cross_validation(self, X, y, k=3, data_fraction = 1.0):
        X, y = self.shuffle(X, y)
        no_of_examples = len(y)
        delta_remaining = {}
        no_of_delta_remaining = (no_of_examples % k)
        if no_of_delta_remaining != 0:
            X = X[:-no_of_delta_remaining]
            delta_remaining["X"] = X[-no_of_delta_remaining:]
            y = y[:-no_of_delta_remaining]
            delta_remaining["y"] = y[-no_of_delta_remaining:]
        X_split = np.split(X, k)
        y_split = np.split(y, k)
        splits = []
        for i in range(k):
            X_test, y_test = X_split[i], y_split[i]
            X_train = np.concatenate(X_split[:i] + X_split[i+1:], axis=0)
            y_train = np.concatenate(y_split[:i] + y_split[i+1:], axis=0)
            for runs in range(5):
                X_train, y_train = self.shuffle(X_train,y_train,data_fraction)
                splits.append([X_train, X_test, y_train, y_test])
        if no_of_delta_remaining != 0:
            np.append(splits[-1][0], delta_remaining["X"], axis=0)
            np.append(splits[-1][2], delta_remaining["y"], axis=0)

        return np.array(splits)

    def load_file(self):
        df = pd.read_csv('data_banknote_authentication.txt', sep=",", header=None)
        df.columns = ["x1", "x2", "x3", "x4", "y"]
        X = (df.values)[:,[0,1,2,3]]
        y = (df.values)[:,[4]]

        return X,y,df

    def train(self, X_train, y_train):
        negative_matrix = X_train[y_train[:,0] == 0.0]
        positive_matrix = X_train[y_train[:,0] == 1.0]

        learnt_params = {}
        try:
            learnt_params["p_y_1"] = positive_matrix.shape[0] / (positive_matrix.shape[0] + negative_matrix.shape[0])
        except ZeroDivisionError:
            learnt_params["p_y_1"] = 0.5
        learnt_params["mu_pos"] = np.mean(positive_matrix, axis=0)
        learnt_params["var_pos"] = np.var(positive_matrix, axis=0)
        learnt_params["p_y_0"] = 1 - learnt_params["p_y_1"]
        learnt_params["mu_neg"] = np.mean(negative_matrix, axis=0)
        learnt_params["var_neg"] = np.var(negative_matrix, axis=0)

        return learnt_params

    def predict(self, X_test, y_test, learnt_params):
        denom = np.sqrt(2*np.pi*np.square(learnt_params["var_pos"]))
        numer = np.exp(-np.square(X_test - learnt_params["mu_pos"]) / (2*np.square(learnt_params["var_pos"])))
        p_x_y = numer/denom
        p_x_y_pos = p_x_y[:,0]*p_x_y[:,1]*p_x_y[:,2]
        p_y_1_mul_p_x_y_pos = learnt_params["p_y_1"] * p_x_y_pos

        denom = np.sqrt(2 * np.pi * np.square(learnt_params["var_neg"]))
        numer = np.exp(-np.square(X_test - learnt_params["mu_neg"]) / (2 * np.square(learnt_params["var_neg"])))
        p_x_y = numer / denom
        p_x_y_neg = p_x_y[:, 0] * p_x_y[:, 1] * p_x_y[:, 2]
        p_y_0_mul_p_x_y_neg = learnt_params["p_y_0"] * p_x_y_neg

        y_pred = (p_y_1_mul_p_x_y_pos > p_y_0_mul_p_x_y_neg)*1.0
        y_pred = np.reshape(y_pred, (y_pred.shape[0],1))
        accuracy = np.sum(y_test == y_pred, axis=0) / len(y_test)

        return accuracy

    def shuffle(self, X, y, data_fraction=1.0):
        # seed = np.random.random_integers(5,high=1000)
        # np.random.seed(seed)
        rows = np.arange(X.shape[0])
        np.random.shuffle(rows)
        rows = rows[0:int(data_fraction*X.shape[0])]

        return X[rows], y[rows]

    def plot_learning_curve(self, plot_curve_dict):
        x, y = zip(*sorted(plot_curve_dict.items()))
        plt.plot(x, y)
        plt.ylabel('accuracy')
        plt.xlabel('size_of_dataset')
        plt.show()

        return None

    def run(self):
        X, y, df = self.load_file()
        data_fractions_dict = {.01: [], .02: [], .05: [], .1: [], .625: [], 1.0: []}
        for data_fraction in data_fractions_dict.keys():
            sets = self.k_folds_cross_validation(X, y, data_fraction=data_fraction)
            for set in sets:
                learnt_params = self.train(set[0], set[2])
                accuracy = self.predict(set[1], set[3], learnt_params)
                print(accuracy)
                data_fractions_dict[data_fraction].append(accuracy)
        plot_curve_dict = {key: np.mean(values) for key, values in data_fractions_dict.items()}
        self.plot_learning_curve(plot_curve_dict)

        return plot_curve_dict


if __name__ == "__main__":
    obj = GNB()
    obj.run()