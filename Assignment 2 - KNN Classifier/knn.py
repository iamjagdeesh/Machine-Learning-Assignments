import pandas as pd
import numpy as np
import operator
from collections import Counter
import matplotlib.pyplot as plt
from scipy.spatial import distance as dist


class KNN(object):
    def __init__(self):
        self.X_train, self.y_train, self.df_train = self.load_train_data()
        self.X_test, self.y_test, self.df_test = self.load_test_data()
        self.distance_matrix_of_test_to_train = self.get_distance_matrix_of_test_to_train()
        self.distance_matrix_of_train_to_train = self.get_distance_matrix_of_train_to_train()

    def load_train_data(self):
        df_train = pd.read_csv('mnist_train.csv', sep=",", header=None, nrows=6000)
        X_train = df_train.values[:, 1:]
        y_train = df_train.values[:, [0]]

        return X_train,y_train,df_train

    def load_test_data(self):
        df_test = pd.read_csv('mnist_test.csv', sep=",", header=None, skiprows=9000)
        X_test = df_test.values[:, 1:]
        y_test = df_test.values[:, [0]]

        return X_test,y_test,df_test

    def get_distance_matrix_of_train_to_train(self, k=9):
        m_train, n_train = self.X_train.shape
        distance_matrix_of_train_to_train = np.zeros(shape=(m_train, m_train))
        for i in range(m_train):
            u = self.X_train[[i], :]
            squared = np.square(u-self.X_train)
            summed = np.sum(squared, axis=1)
            sqrted = np.sqrt(summed)
            distance_matrix_of_train_to_train[i] = sqrted

        return distance_matrix_of_train_to_train

    def get_distance_matrix_of_test_to_train(self, k=9):
        m_train, n_train = self.X_train.shape
        m_test, n_test = self.X_test.shape
        distance_matrix_of_test_to_train = np.zeros(shape=(m_test, m_train))
        for i in range(m_test):
            u = self.X_test[[i], :]
            squared = np.square(u-self.X_train)
            summed = np.sum(squared, axis=1)
            sqrted = np.sqrt(summed)
            distance_matrix_of_test_to_train[i] = sqrted

        return distance_matrix_of_test_to_train

    def predict_test_data(self, k=9):
        m_train,n_train = self.X_train.shape
        m_test, n_test = self.X_test.shape
        y_pred = np.zeros(shape=(m_test,1))
        for i in range(m_test):
            distances_from_i = self.distance_matrix_of_test_to_train[i]
            distances_from_i_dict = dict(enumerate(distances_from_i))
            distances_from_i_dict_sorted = sorted(distances_from_i_dict.items(), key=operator.itemgetter(1))
            distances_from_i_dict_sorted = distances_from_i_dict_sorted[0:k]
            label_count_dict = Counter()
            for (X_train_index, distance) in distances_from_i_dict_sorted:
                label_count_dict[self.y_train[X_train_index, 0]] += 1

            label_count_dict_sorted = sorted(label_count_dict.items(), key=operator.itemgetter(1), reverse=True)
            (label, count) = label_count_dict_sorted[0]
            y_pred[i] = label
        accuracy = np.sum(self.y_test == y_pred, axis=0) / float(len(self.y_test))

        return 1-accuracy

    def predict_train_data(self, k=9):
        m_train,n_train = self.X_train.shape
        m_test, n_test = m_train,n_train
        y_pred = np.zeros(shape=(m_test,1))
        for i in range(m_test):
            distances_from_i = self.distance_matrix_of_train_to_train[i]
            distances_from_i_dict = dict(enumerate(distances_from_i))
            distances_from_i_dict_sorted = sorted(distances_from_i_dict.items(), key=operator.itemgetter(1))
            distances_from_i_dict_sorted = distances_from_i_dict_sorted[0:k]
            label_count_dict = Counter()
            for (X_train_index, distance) in distances_from_i_dict_sorted:
                label_count_dict[self.y_train[X_train_index, 0]] += 1

            label_count_dict_sorted = sorted(label_count_dict.items(), key=operator.itemgetter(1), reverse=True)
            (label, count) = label_count_dict_sorted[0]
            y_pred[i] = label
        accuracy = np.sum(self.y_train == y_pred, axis=0) / float(len(self.y_train))

        return 1-accuracy

    def plot_learning_curve(self, plot_curve1_dict, plot_curve2_dict):
        x, y_error1 = zip(*sorted(plot_curve1_dict.items()))
        x, y_error2 = zip(*sorted(plot_curve2_dict.items()))
        fig = plt.figure()
        plt.title('Error vs Value of K')
        plt.plot(x, y_error1, 'r', label='Training Set Error')
        plt.plot(x, y_error2, 'b', label='Test Set Error')
        plt.ylabel('Error')
        plt.xlabel('Value of K')
        plt.legend()
        plt.show()
        fig.savefig('KNN_graph.png')

        return None

    def run(self):
        k_training_dict = {1: [], 9: [], 19: [], 29: [], 39: [], 49: [], 59: [], 69: [], 79: [], 89: [], 99: []}
        k_test_dict = {1: [], 9: [], 19: [], 29: [], 39: [], 49: [], 59: [], 69: [], 79: [], 89: [], 99: []}
        for k in k_training_dict.keys():
            training_set_error = self.predict_train_data(k=k)
            test_set_error = self.predict_test_data(k=k)
            k_training_dict[k] = training_set_error
            k_test_dict[k] = test_set_error
            print("K = ", k, " Training set error: ", training_set_error, " Test set error: ", test_set_error)
        self.plot_learning_curve(k_training_dict, k_test_dict)

        return None


if __name__ == "__main__":
    obj = KNN()
    obj.run()