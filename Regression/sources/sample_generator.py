#!/usr/local/bin/python

import math
import numpy as np

class SampleGenerator:
    def __init__(self):
        self.c0 = np.random.random()*2
        self.c1 = -np.random.random()*4
        self.c2 = -np.random.random()*4
        self.c3 = np.random.random()*4

    def generate_non_linear_samples(self, x):
        """
        Generate a noisy nonlinear data sample from a given data point in the range [0,1]

        :param x: A scalar dependent variable for which to calculate the output y_noisy
        
        :returns: The output with Gaussian noise added
            
        """
        y = self.c0 - x - math.sin(self.c1 * math.pi * x ** 3) * math.cos(self.c2 * math.pi * x ** 3) * math.exp(-x ** 4)
        sigma = 2
        noise = sigma * np.random.random()
        y_noisy = y + noise
        return y_noisy

    def generate_linear_samples(self, x):
        """
        Generate a noisy linear data sample from a given data point in the range [0,1]

        :param x: A scalar dependent variable for which to calculate the output y_noisy

        :returns: The output with Gaussian noise added

        """
        y = self.c3 * x + self.c1
        sigma = 0.5
        noise = sigma * np.random.random()
        y_noisy = y + noise
        return y_noisy
