import numpy as np


class Gaussians:
    def __init__(self, nb_features):
        self.nb_features = nb_features
        self.centers = np.linspace(0.0, 1.0, self.nb_features)
        width_constant = 0.1 / self.nb_features
        self.sigma = np.ones(self.nb_features, ) * width_constant

    def phi_output(self, x):
        """
        Get the output of the Gaussian features for a given input x of size N
        The output is a vertical vector. If one wants a standard vector,
        one needs to transpose it and take the first element of the result.
        
        :param x: A single or vector of dependent variables of size N
        :returns: A vector of feature outputs of size nb_features
        """
        if not hasattr(x, "__len__"):
            x = np.array([x])
        # number of dimensions in x (=N)
        dim_x = np.shape(x)[0]
        # Repeat vectors to vectorize output calculation
        input_mat = np.array([x, ] * self.nb_features)
        centers_mat = np.array([self.centers, ] * dim_x).transpose()
        widths_mat = np.array([self.sigma, ] * dim_x).transpose()
        phi = np.exp(-np.divide(np.square(input_mat - centers_mat), widths_mat))
        return phi

if __name__ == '__main__':
    g = Gaussians(3)
    x = [0.1]
    phi = g.phi_output(x)
    print(phi)
