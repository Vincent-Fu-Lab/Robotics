import numpy as np
import matplotlib.pyplot as plt
from gaussians import Gaussians


class RBFN(Gaussians):
    def __init__(self, nb_features):
        super().__init__(nb_features)
        self.theta = np.random.random(self.nb_features)
        self.a = np.zeros(shape=(self.nb_features, self.nb_features))
        self.a_inv = np.matrix(np.identity(self.nb_features))
        self.b = np.zeros(self.nb_features)

    def f(self, x, theta=None):
        """
        Get the FA output for a given input vector
    
        :param x: A vector of dependent variables of size N
        :param theta: A vector of coefficients to apply to the features. 
        :If left blank the method will default to using the trained thetas in self.theta.
        
        :returns: A vector of function approximator outputs with size nb_features
        """
        if not hasattr(theta, "__len__"):
            theta = self.theta
        value = np.dot(self.phi_output(x).transpose(), theta.transpose())
        return value

    def feature(self, x, idx):
        """
         Get the output of the idx^th feature for a given input vector
         This is function f() considering only one feature
         Used mainly for plotting the features

         :param x: A vector of dependent variables of size N
         :param idx: index of the feature

         :returns: the value of the feature for x
         """
        phi = self.phi_output(x)
        return phi[idx] * self.theta[idx]

    # ----------------------#
    # # Training Algorithms ##
    # ----------------------#

    # ------ batch least squares (projection approach) ---------
    def train_ls(self, x_data, y_data):
        x = np.array(x_data)
        y = np.array(y_data)
        X = self.phi_output(x)
        
        #TODO: Fill this
        
        self.theta = np.linalg.inv(X @ X.T) @ X @ y
            
    

    # ------ batch least squares (calculation approach) ---------
    def train_ls2(self, x_data, y_data):
        a = np.zeros(shape=(self.nb_features, self.nb_features))
        b = np.zeros(self.nb_features)
        for i in range(len(x_data)):
            
        #TODO: Fill this
        
            C = self.phi_output(x_data[i])
            a += C @ C.T
            b += C.T[0] * y_data[i]
            
        self.theta = np.linalg.solve(a, b)
            
        

    # -------- gradient descent -----------------
    def train_gd(self, x, y, alpha):
            
        #TODO: Fill this
        C = self.phi_output(x)
        self.theta = self.theta + alpha * (y - C.T[0] @ self.theta) * C.T[0]

    # -------- recursive least squares -----------------
    def train_rls(self, x, y):
        phi = self.phi_output(x)
        self.a = self.a + np.dot(phi, phi.transpose())
        self.b = self.b + y * phi.transpose()[0]

        result = np.dot(np.linalg.pinv(self.a), self.b)
        self.theta = np.array(result)

    # -------- recursive least squares (other version) -----------------
    def train_rls2(self, x, y):
        phi = self.phi_output(x)
        self.a = self.a + np.outer(phi,phi)
        self.b = self.b + y * phi.transpose()[0]

        self.theta = np.dot(np.linalg.pinv(self.a), self.b)

    # -------- recursive least squares with Sherman-Morrison -----------------
    def train_rls_sherman_morrison(self, x, y):
        u = self.phi_output(x)
        v = self.phi_output(x).transpose()

        value = (v * self.a_inv * u)[0, 0]
        tmp_mat = self.a_inv * np.dot(u, v)* self.a_inv

        self.a_inv = self.a_inv - (1.0 / (1 + value)) * tmp_mat
        self.b = self.b + y * u.transpose()[0]

        result = np.dot(self.a_inv, self.b)
        self.theta = np.array(result)[0]

    # -----------------#
    # # Plot function ##
    # -----------------#

    def plot(self, x_data, y_data):
        xs = np.linspace(0.0, 1.0, 1000)
        z = []
        for i in xs:
            z.append(self.f(i))

        z2 = []
        for i in range(self.nb_features):
            temp = []
            for j in xs:
                temp.append(self.feature(j, i))
            z2.append(temp)

        plt.plot(x_data, y_data, 'o', markersize=3, color='lightgreen')
        plt.plot(xs, z, lw=3, color='red')
        for i in range(self.nb_features):
            plt.plot(xs, z2[i])
        plt.show()
        
        
    def error(self, x_data, y_data):
        return np.sum((y_data - self.f(x_data)) ** 2)
