import cma
import gym
from deap import *
import numpy as np
from fixed_structure_nn_numpy import SimpleNeuralControllerNumpy

from deap import algorithms
from deap import base
from deap import benchmarks
from deap import creator
from deap import tools

import array
import random

import math

from nsga2 import my_nsga2

nn=SimpleNeuralControllerNumpy(4,1,2,5)
IND_SIZE=len(nn.get_parameters())

env = gym.make('CartPole-v1')

def eval_nn(genotype, render=False, nbstep=500):
    total_x=0 # l'erreur en x est dans observation[0]
    total_theta=0 #  l'erreur en theta est dans obervation[2]
    nn=SimpleNeuralControllerNumpy(4,1,2,5)
    nn.set_parameters(genotype)

    observation = env.reset()

    # à compléter
    
    episode_xs = []
    episode_thetas = []
    episode_rewards = []

    
    for i in range(10):
        episode_x = 0
        episode_theta = 0
        episode_reward = 0
        observation = env.reset()
        done = False
        
        while not done and episode_reward < nbstep:
            if render:
                env.render()
    
            if nn.predict(observation)[0] >= 0:
                action = 1
            else:
                action = 0
            
            observation, reward, done, info = env.step(action)
            episode_reward += reward
            episode_x += abs(observation[0])
            episode_theta += abs(observation[2])
            
        episode_xs.append(episode_x)
        episode_thetas.append(episode_theta)
        episode_rewards.append(episode_reward)
        
    total_reward = np.median(episode_rewards)
    total_x = np.median(episode_xs) + (nbstep - total_reward) * 5
    total_theta = np.median(episode_thetas) + (nbstep - total_reward) * 5

    # ATTENTION: vous êtes dans le cas d'une fitness à minimiser.
    # Interrompre l'évaluation le plus rapidement possible est donc une stratégie que l'algorithme évolutionniste
    # peut utiliser pour minimiser la fitness. Dans le cas ou le pendule tombe avant la fin, il faut donc ajouter à
    # la fitness une valeur qui guidera l'apprentissage vers les bons comportements. Vous pouvez par exemple ajouter n fois une pénalité,
    # n étant le nombre de pas de temps restant. Cela poussera l'algorithme à minimiser la pénalité et donc à éviter la chute.
    # La pénalité peut être l'erreur au moment de la chute ou l'erreur maximale.
    
    return total_x, total_theta



if (__name__ == "__main__"):

    # à compléter
    
    pop, paretofront, s_hv = my_nsga2(100, 51, eval_nn, IND_SIZE=IND_SIZE, gym = True)
    print("\n", paretofront[0])
    res = eval_nn(paretofront[0], False)

    env.close()
