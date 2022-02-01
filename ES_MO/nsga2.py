import numpy as np
from deap import base, creator, benchmarks

from deap import algorithms
from deap.tools._hypervolume import hv


import random
from deap import tools

# ne pas oublier d'initialiser la graine aléatoire (le mieux étant de le faire dans le main))
random.seed()

def my_nsga2(n, nbgen, evaluate, ref_point=np.array([1,1]), IND_SIZE=5, weights=(-1.0, -1.0), gym=False):
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
    
    toolbox.register("attribute", random.uniform, -5, 5)                                                  
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n = IND_SIZE)  
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)                             
    toolbox.register("mutation", tools.mutPolynomialBounded, eta = 15, low = -5, up = 5, indpb = 0.5)
    toolbox.register("crossover", tools.cxSimulatedBinary, eta = 15)
    toolbox.register("selection", tools.selNSGA2, k = n)
    toolbox.register("evaluate", evaluate)
    
    population = toolbox.population(n)
    
    for i in range(len(population)):
        population[i].fitness.values = toolbox.evaluate(population[i])
        
    population = toolbox.selection(population)
    paretofront.update(population)

    # Pour récupérer l'hypervolume, nous nous contenterons de mettre les différentes aleur dans un vecteur s_hv qui sera renvoyé par la fonction.
    pointset=[np.array(ind.fitness.getValues()) for ind in paretofront]
    
    if not gym:
        s_hv=[hv.hypervolume(pointset, ref_point)]
    else:
        s_hv=[(np.median([f[0] for f in pointset]), np.median([f[1] for f in pointset]))]

    # Begin the generational process
    for gen in range(1, nbgen):
        if (gen%10==0):
            print("+",end="", flush=True)
        else:
            print(".",end="", flush=True)

        # à completer
            
        offspring = []
        
        # varAnd
        
        for i in range((n + 1) // 2):
            ind1, ind2 = random.randint(0, len(population) - 1), random.randint(0, len(population) - 1)
            
            while ind2 == ind1:
                ind2 = random.randint(0, len(population) - 1)
            
            p1, p2 = toolbox.crossover(toolbox.clone(population[ind1]), toolbox.clone(population[ind2]))
            p1, p2 = toolbox.mutation(p1)[0], toolbox.mutation(p2)[0]
            offspring.append(p1)
            offspring.append(p2)
            
        for i in range(len(offspring)):
            offspring[i].fitness.values = toolbox.evaluate(offspring[i])
        
        population = toolbox.selection(population + offspring)
        paretofront = tools.ParetoFront()
        paretofront.update(population)

        pointset=[np.array(ind.fitness.getValues()) for ind in paretofront]
        
        if not gym:
            s_hv.append(hv.hypervolume(pointset, ref_point))
        else:
            s_hv.append((np.median([f[0] for f in pointset]), np.median([f[1] for f in pointset])))
        
    return population, paretofront, s_hv
