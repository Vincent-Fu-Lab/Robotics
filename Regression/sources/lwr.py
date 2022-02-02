import numpy as np
import matplotlib.pyplot as plt
from gaussians import Gaussians


def bar(x):
    """
    Add a one at the end of a vector of x data
    :param x: a single vector of size [d]
    :returns: the same vector with a 1 in the end
    """
    if np.size(x) == 1:
        w = np.vstack(([x], [1]))
    else:
        w = np.vstack((x, np.ones((1, np.size(x)))))
    return w


class LWR(Gaussians):
    def __init__(self, nb_features):
        super().__init__(nb_features)
        self.theta = np.zeros((2, self.nb_features))

    def f(self, x):
        """
        Get the FA output for a given input variable(s)

        :param x: a single or vector of dependent variables with size [d] for which to calculate the features

        :returns: a vector of function approximator outputs with size [d]
        """
        wval = bar(x)
        phi = self.phi_output(x)
        linear_model = (np.dot(wval.transpose(), self.theta)).transpose() # [numFeats x d]
        return np.sum(phi * linear_model, axis=0) / np.sum(phi, axis=0)

    def feature(self, x, idx):
        """
         Get the output of the idx^th feature for a given input variable(s)

         :param x: a single or vector of dependent variables with size [d] for which to calculate the features
         :param idx: index of the feature

         :returns: a vector of values
         """
        return np.dot(bar(x)[:, 0], self.theta[:, idx])

    # ----------------------#
    # # Training Algorithm ##
    # ----------------------#

    def train_lwls(self, x_data, y_data) -> None:
        """
        Locally weighted least square function
        This code is specific to the 1D case
        :param x_data: a vector of x values
        :param y_data: a vector of y values
        :return: nothing (set the self.theta vector)
        """

        for k in range(self.nb_features):
            a = np.zeros(shape=(2, 2))
            b = np.zeros((2, 1))

            for i in range(len(x_data)):
                w = np.matrix(bar(x_data[i]))
                phi = self.phi_output(x_data[i])[k]
                ww = np.dot(w, w.transpose())

                a = a + phi[0] * ww
                b = b + y_data[i] * phi[0] * w

            result = np.linalg.pinv(a) * b
            for i in range(2):
                self.theta[i, k] = result[i, 0]


    def plot(self, x_data, y_data):
        xs = np.linspace(0.0, 1.0, 1000)
        z = self.f(xs)

        plt.plot(x_data, y_data, 'o', markersize=3, color='lightgreen')
        plt.plot(xs, z, lw=2, color='red')
        for i in range(self.nb_features):
            ww = (1.0 - 0.0) / self.nb_features / 2.
            xstmp = np.linspace(self.centers[i] - ww, self.centers[i] + ww, 100)

            z2 = []
            for j in xstmp:
                z2.append(self.feature(j, i))
            plt.plot(xstmp, z2, lw=2, color='blue', ls='-')
        plt.show()
        
        
    def error(self, x_data, y_data):
        
        return np.sum((y_data - self.f(x_data)) ** 2)