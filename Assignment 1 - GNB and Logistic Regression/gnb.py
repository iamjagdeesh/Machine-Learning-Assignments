import pandas as pd
import numpy as np

class GNB(object):

    def load_file(self):
        df = pd.read_csv('data_banknote_authentication.txt', sep=",", header=None)
        df.columns = ["x1", "x2", "x3", "x4", "y"]
        return df

    def train(self):
        df = self.load_file()
        positive_matrix = df.loc[df['y'] == 1.0].values
        negative_matrix = df.loc[df['y'] == 0.0].values
        x_pos = np.array(positive_matrix[:,[0,1,2,3]])
        x_neg = np.array(negative_matrix[:,[0,1,2,3]])
        # for y = 1
        p_y_1 = x_pos.size / (x_pos.size + x_neg.size)
        mu_pos = np.mean(x_pos, axis=0)
        var_pos = np.var(x_pos, axis=0)
        p_y_0 = x_neg.size / (x_pos.size + x_neg.size)
        mu_neg = np.mean(x_neg, axis=0)
        var_neg = np.var(x_neg, axis=0)
        return (p_y_1, mu_pos, mu_neg, var_pos, var_neg)

if __name__ == "__main__":
    obj = GNB()
    obj.train()
    print("All set to code!")