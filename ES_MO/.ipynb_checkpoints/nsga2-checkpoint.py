import numpy as np
from deap import base, creator, benchmarks

from deap import algorithms
from deap.tools._hypervolume import hv


import random
from deap import tools

# ne pas oublier d'initialiser la grane aléatoire (le mieux étant de le faire dans le main))
random.seed()

def my_nsga2(n, nbgen, evaluate, ref_point=np.array([1,1]), IND_SIZE=5, weights=(-1.0, -1.0)):
    """NSGA-2

    NSGA-2
    :param n: taille de la population
    :param nbgen: nombre de generation 
    :param evaluate: la fonction d'évaluation
    :param ref_point: le point de référence pour le calcul de l'hypervolume
    :param IND_SIZE: la taille d'un individu
    :param weights: les poids à utiliser pour la fitness (ici ce sera (-1.0,) pour une fonction à minimiser et (1.0,) pour une fonction à maximiser)
    """

    creator.create("MaFitness", base.Fitness, weights=weights)
    creator.create("Individual", list, fitness=creator.MaFitness)
    

    toolbox = base.Toolbox()
    paretofront = tools.ParetoFront()


    # à compléter

    # Pour récupérer l'hypervolume, nous nous contenterons de mettre les différentes aleur dans un vecteur s_hv qui sera renvoyé par la fonction.
    pointset=[np.array(ind.fitness.getValues()) for ind in paretofront]
    s_hv=[hv.hypervolume(pointset, ref_point)]

    # Begin the generational process
    for gen in range(1, nbgen):
        if (gen%10==0):
            print("+",end="", flush=True)
        else:
            print(".",end="", flush=True)

        # à completer
            

        pointset=[np.array(ind.fitness.getValues()) for ind in paretofront]
        s_hv.append(hv.hypervolume(pointset, ref_point))
            
    return population, paretofront, s_hv
