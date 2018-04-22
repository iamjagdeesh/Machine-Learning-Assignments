import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Kmeans(object):
    def __init__(self, iterations=1000):
        self.X = self.load_data()
        self.iterations = iterations

    def load_data(self):
        df = pd.read_csv('bc.txt', sep=",", header=None)
        X = df.values[:, 1:-1]

        return X

    def run(self):
        k_dict = {2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: []}
        for k in k_dict.keys():
            k_dict[k] = self.kmeans(K=k)
            print("K = ", k, " F_mu_c: ", k_dict[k])
        self.plot_learning_curve(k_dict)

        return None

    def plot_learning_curve(self, plot_curve_dict):
        x, y = zip(*sorted(plot_curve_dict.items()))
        fig = plt.figure()
        plt.title('Value of Potential function vs Value of K')
        plt.plot(x, y, 'r')
        plt.ylabel('Value of Potential function')
        plt.xlabel('Value of K')
        plt.show()
        fig.savefig('KMeans_graph.png')

        return None

    def kmeans(self, K=3):
        m,n = self.X.shape
        centroid_indexes, centroids = self.randomly_select_centroid_indexes(K=K)
        old_centroids = None
        data_labels = None
        iterations = 0
        while iterations < self.iterations and (not np.array_equal(old_centroids, centroids)):
            iterations += 1
            old_centroids = centroids
            data_labels = self.assign_centroid(centroids)
            centroids = self.recompute_centroid(old_centroids, data_labels)

        F_mu_c = self.compute_potential_function(centroids=centroids, data_labels=data_labels)

        return F_mu_c

    def compute_potential_function(self, centroids, data_labels):
        F_mu_c = 0
        for i in range(centroids.shape[0]):
            indices_belonging_to_i = np.where(data_labels == i)
            data_belonging_to_i = self.X[indices_belonging_to_i]
            u = centroids[[i], :]
            squared = np.square(u - data_belonging_to_i)
            summed = np.sum(squared) #, axis=1)
            # sqrted = np.sqrt(summed)
            F_mu_c += summed # np.sum(summed)

        return F_mu_c

    def recompute_centroid(self, old_centroids, data_labels):
        new_centroids = np.zeros(shape=old_centroids.shape)
        for i in range(old_centroids.shape[0]):
            indices_belonging_to_i = np.where(data_labels == i)
            data_belonging_to_i = self.X[indices_belonging_to_i]
            new_centroid = np.mean(data_belonging_to_i, axis=0)
            new_centroids[i] = new_centroid

        return new_centroids

    def randomly_select_centroid_indexes(self, K=3):
        """
        Randomly select K centroids from the dataset
        :return:
        """
        np.random.seed(1)
        rows = np.arange(self.X.shape[0])
        np.random.shuffle(rows)
        rows = rows[0:K]

        return rows, self.X[rows]

    def assign_centroid(self, centroids):
        m_data, n_data = self.X.shape
        m_centroid, n_centroid = centroids.shape
        distance_matrix_of_centroids_to_data = np.zeros(shape=(m_centroid, m_data))
        for i in range(m_centroid):
            u = centroids[[i], :]
            squared = np.square(u - self.X)
            summed = np.sum(squared, axis=1)
            sqrted = np.sqrt(summed)
            distance_matrix_of_centroids_to_data[i] = sqrted

        distance_matrix_of_data_to_centroids = distance_matrix_of_centroids_to_data.transpose()
        data_labels = np.argmin(distance_matrix_of_data_to_centroids, axis=1)

        return data_labels



if __name__ == "__main__":
    obj = Kmeans(iterations=1000)
    obj.run()