#!/usr/local/bin/python

import numpy as np
import time
from rbfn import RBFN
from lwr import LWR
from line import Line
from sample_generator import SampleGenerator
import matplotlib.pyplot as plt


class Main:
    def __init__(self):
        self.x_data = []
        self.y_data = []
        self.batch_size = 50

    def reset_batch(self):
        self.x_data = []
        self.y_data = []

    def make_nonlinear_batch_data(self):
        """ 
        Generate a batch of non linear data and store it into numpy structures
        """
        self.reset_batch()
        g = SampleGenerator()
        for i in range(self.batch_size):
            # Draw a random sample on the interval [0,1]
            x = np.random.random()
            y = g.generate_non_linear_samples(x)
            self.x_data.append(x)
            self.y_data.append(y)

    def make_linear_batch_data(self):
        """ 
        Generate a batch of linear data and store it into numpy structures
        """
        self.reset_batch()
        g = SampleGenerator()
        for i in range(self.batch_size):
            # Draw a random sample on the interval [0,1]
            x = np.random.random()
            y = g.generate_linear_samples(x)
            self.x_data.append(x)
            self.y_data.append(y)

    def approx_linear_batch(self):
        model = Line()

        self.make_linear_batch_data()

        start = time.perf_counter()
        model.train(self.x_data, self.y_data)
        print("LLS time:", time.perf_counter() - start)
        print("LLS error:", model.error(self.x_data, self.y_data))
        model.plot(self.x_data, self.y_data)

        start = time.perf_counter()
        model.train_from_stats(self.x_data, self.y_data)
        print("LLS from scipy stats:", time.perf_counter() - start)
        print("LLS error from scipy stats:", model.error(self.x_data, self.y_data))
        model.plot(self.x_data, self.y_data)

        start = time.perf_counter()
        model.train_regularized(self.x_data, self.y_data, coef=0.01)
        print("regularized LLS :", time.perf_counter() - start)
        print("regularized LLS error:", model.error(self.x_data, self.y_data))
        model.plot(self.x_data, self.y_data)

    def approx_rbfn_batch(self):
        model = RBFN(nb_features=10)
        self.make_nonlinear_batch_data()

        start = time.perf_counter()
        model.train_ls(self.x_data, self.y_data)
        print("RBFN LS time:", time.perf_counter() - start)
        print("RBFN LS error:", model.error(self.x_data, self.y_data))
        model.plot(self.x_data, self.y_data)

        start = time.perf_counter()
        model.train_ls2(self.x_data, self.y_data)
        print("RBFN LS2 time:", time.perf_counter() - start)
        print("RBFN LS2 error:", model.error(self.x_data, self.y_data))
        model.plot(self.x_data, self.y_data)

    def approx_rbfn_iterative(self):
        max_iter = 50
        model = RBFN(nb_features=10)
        model2 = RBFN(nb_features=10)
        model3 = RBFN(nb_features=10)
        timer = 0
        timer2 = 0
        timer3 = 0
        start = time.perf_counter()
        # Generate a batch of data and store it
        self.reset_batch()
        g = SampleGenerator()
        for i in range(max_iter):
            # Draw a random sample on the interval [0,1]
            x = np.random.random()
            y = g.generate_non_linear_samples(x)
            self.x_data.append(x)
            self.y_data.append(y)

            # Comment the ones you don't want to use
            
            temp = time.perf_counter()
            model.train_gd(x, y, alpha=0.5)
            timer += time.perf_counter() - start
            timer2 -= time.perf_counter() - temp
            model2.train_rls(x, y)
            timer2 += time.perf_counter() - start
            timer3 -= time.perf_counter() - temp
            model3.train_rls_sherman_morrison(x, y)
            timer3 += time.perf_counter() - start
            start = time.perf_counter()

        print("RBFN Incr time:", timer)
        print("RBFN Incr error:", model.error(self.x_data, self.y_data))
        model.plot(self.x_data, self.y_data)
        
        print("RBFN Incr time:", timer2)
        print("RBFN Incr error:", model2.error(self.x_data, self.y_data))
        model2.plot(self.x_data, self.y_data)
        
        print("RBFN Incr time:", timer3)
        print("RBFN Incr error:", model3.error(self.x_data, self.y_data))
        model3.plot(self.x_data, self.y_data)
        
    def approx_lwr_batch(self):
        model = LWR(nb_features=200)
        self.make_nonlinear_batch_data()

        start = time.perf_counter()
        model.train_lwls(self.x_data, self.y_data)
        print("LWR time:", time.perf_counter() - start)
        print("LWR error:", model.error(self.x_data, self.y_data))
        model.plot(self.x_data, self.y_data)
        
    def plot_RRLS_error_evolution(self, lambda_max = 10, size = 10000):
        model = Line()
        
        x_ = np.linspace(0, lambda_max, size)
        y_ = []
        
        for i in x_:
            model.train_regularized(self.x_data, self.y_data, i)
            y_.append(model.error(self.x_data, self.y_data))
            
        plt.plot(x_, y_)
        plt.show()
        

if __name__ == '__main__':
    m = Main()
    #m.approx_linear_batch()
    #m.plot_RRLS_error_evolution()
    m.approx_rbfn_batch()
    #m.plot_RBFN_error_evolution()
    #m.approx_rbfn_iterative()
    m.approx_lwr_batch()
