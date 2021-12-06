from deap import creator, gp, base, tools, algorithms
import operator
import math
import matplotlib.pyplot as plt
import numpy as np
import random
import argparse
import pickle
import datetime
import sys
import os
import networkx as nx


def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1

def ru():
    return random.uniform(-1,1)

def load_data(file):
    fdata = open(file)
    state=[]
    action=[]

    for l in fdata.readlines():
        d=l.split(" ")
        #print("d: "+str(d))
        fd = list(map(float, d))
        action.append([fd[0]])
        state.append(fd[1:])

    fdata.close()

    return state,action


def evalSymbReg(individual, input, output, nb_obj=1):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)

    ## à compléter: calcule l'erreur entre l'entrée et la sortie désirée 
    ## remarque: l'entrée est un verteur, vous pouvez appeler la fonction func de la façon suivante: func(*input[i]) pour tester la i-ième entrée
    ## lorsque la fonction renverra l'erreur seule, faire un renvoi sous la forme 'return err,' (ne pas oublier la virgule car DEAP attend un tuple) 
    
    error = 0.0
    n = len(input)
    
    for i in range(n):
        error += abs(func(*input[i]) - output[i])
    
    return error,

def load_data(file, state_dim=0):
    # (option) fonction à utiliser pour lire des données que vous aurez sauvez depuis un simulateur, par exemple le cartpole
    fdata = open(file)
    state=[]
    action=[]

    for l in fdata.readlines():
        d=l.split(" ")
        #print("d: "+str(d))
        fd = list(map(float, d))
        action.append([fd[0]])
        state.append(fd[1:])

    fdata.close()

    input=[]
    output=[]

    for i in range(len(state)-1):
        input.append(state[i]+action[i])
        output.append(state[i+1][state_dim])

    return input, output


NB_VAL = 30
nbi = 0
moy = []
nbf = 0
nbp = 0
NB_CALLS = 20

while nbi < NB_CALLS:
    random.seed()


    parser = argparse.ArgumentParser(description='Launch symbolic regression run.')
    parser.add_argument('--nb_gen', type=int, default=100,
                        help='number of generations')
    parser.add_argument('--mu', type=int, default=400,
                        help='population size')
    parser.add_argument('--lambda_', type=int, default=400,
                        help='number of individuals to generate')
    parser.add_argument('--res_dir', type=str, default="res",
                        help='basename of the directory in which to put the results')
    parser.add_argument('--selection', type=str, default="nsga2", choices=['elitist', 'double_tournament', 'nsga2'],
                        help='selection scheme')
    parser.add_argument('--problem', type=str, default="f1", choices=['f1', 'f2', 'cartpole'], ## à adapter aux problèmes que vous aurez définis
                        help='function to fit')
    parser.add_argument('--state_dim', type=int, default=0,
                        help='dimension to fit (for problems with multiple dimensions, e.g. cartpole)')
    parser.add_argument('--noise', type=float, default="0.0001",
                        help='noise added to the model to fit (gaussian, mean=0, sigma=noise)')
    
    args = parser.parse_args()
    print("Number of generations: "+str(args.nb_gen))
    ngen=args.nb_gen
    print("Population size: "+str(args.mu))
    mu=args.mu
    print("Number of offspring to generate: "+str(args.lambda_))
    lambda_=args.lambda_
    print("Selection scheme: "+str(args.selection))
    sel=args.selection
    if (sel=="nsga2"):
        nb_obj=2
    else:
        nb_obj=1
    print("Basename of the results dir: "+str(args.res_dir))
    name=args.res_dir

    print("State dim: "+str(args.state_dim)+" (ignored on most problems)")
    state_dim=args.state_dim


    noise=args.noise

    problem=args.problem

    if (problem=="f1"):
        nb_dim=2
        def f(x):
            return x[0] * x[1] + math.cos(x[0])

        input_training=[]
        ## à compléter
        
        for i in range(NB_VAL):
            input_training.append([random.random() for j in range(nb_dim)])
            
        output_training=[f(x) for x in input_training]

        input_testing=[]
        ## à compléter
        
        for i in range(NB_VAL):
            input_testing.append([random.random() for j in range(nb_dim)])
        
        output_testing=[f(x) for x in input_testing]
        name_vars={"ARG0": "x1", "ARG1": "x2"}
    elif (problem=="f2"):
        ## à compléter
        
        nb_dim=3
        
        def f(x):
            return (x[0] ** 2) + (x[1] ** 2) + (x[2] ** 2)

        input_training=[]
        
        for i in range(NB_VAL):
            input_training.append([random.random() for j in range(nb_dim)])
            
        output_training=[f(x) for x in input_training]
        input_testing=[]
        
        for i in range(NB_VAL):
            input_testing.append([random.random() for j in range(nb_dim)])
        
        output_testing=[f(x) for x in input_testing]
        name_vars={"ARG0": "x1", "ARG1": "x2", "ARG2" : "x3"}
    elif (problem=="cartpole"):
        ## à compléter (option)
        pass

    ## à compléter: créer pset, l'ensemble de primitives sur lequel la programmation génétique va s'appuyer, cf gp.PrimitiveSet

    pset = gp.PrimitiveSet("main", len(name_vars))
    pset.renameArguments(**name_vars)
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(protectedDiv, 2)
    pset.addPrimitive(math.cos, 1)
    pset.addPrimitive(math.sin, 1)
    pset.addEphemeralConstant("ru%d"%(nbi), ru)
    pset.addTerminal(1)
    

    ## à compléter, définir les fitness (prendre en compte nb_obj) et Individual, cf creator dans DEAP
    
    creator.create("FitnessMin", base.Fitness, weights = tuple([-1.0 for i in range(nb_obj)]))
    creator.create("Individual", gp.PrimitiveTree, fitness = creator.FitnessMin, pset = pset)

    # valeur typiques de mutation et de croisement en GP
    cxpb=0.5
    mutpb=0.1


    d=datetime.datetime.today()
    if(name!=""):
        sep="_"
    else:
        sep=""
    run_name=name+"_"+d.strftime(name+sep+"%Y_%m_%d-%H-%M-%S")
    try:
        os.makedirs(run_name)
    except OSError:
        pass
    print("Putting the results in : "+run_name)

        
    ## à compléter: définir la toolbox à utiliser pour la GP
    ## notamment, la fonction d'évaluation (faire appel à evalSymbReg) et la stratégie de sélection 
    ## en fonction de la valeur de la variable sel.
    
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genFull, pset = pset, min_ = 1, max_ = 10)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset = pset)
    toolbox.register("evaluate", evalSymbReg, input = input_training, output = output_training)
    
    if sel == "elitist":
        toolbox.register("select", tools.selBest)
    elif sel == "double_tournament":
        toolbox.register("select", tools.selDoubleTournament, fitness_size = mu, parsimony_size = 1.5, fitness_first = True)
    elif sel == "nsga2":
        toolbox.register("select", tools.selNSGA2)
        
    toolbox.register("mate", gp.cxOnePointLeafBiased, termpb = 0.2)
    toolbox.register("expr_mut", gp.genFull, min_= 0, max_= 5)
    toolbox.register("mutate", gp.mutEphemeral, mode = "one")
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=10))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=10))
    
    
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    stats_height = tools.Statistics(lambda ind: ind.height)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size, height=stats_height)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)
    
    
    ## à compléter, initialisation de la population: 
    pop = toolbox.population(mu)

    if (nb_obj==1):
        print("Hall-of-fame: best solution")
        hof = tools.HallOfFame(1)
    else:
        print("Hall-of-fame: Pareto front")
        hof=tools.ParetoFront()

    ## à compléter: faire appel à l'algorithme évolutionniste algorithms.eaMuPlusLambda
    
    pop, log = algorithms.eaMuPlusLambda(pop, toolbox, mu, lambda_, cxpb, mutpb, ngen, stats = mstats, halloffame = hof, verbose= False)
    
    
    # Tracés de quelques courbes de résultats
    avg,dmin,dmax=log.chapters['fitness'].select("avg", "min", "max")
    gen=log.select("gen")

    plt.figure()
    plt.yscale("log")
    plt.plot(gen[1:],dmin[1:])
    plt.title("Minimum error")
    plt.savefig(run_name+"/min_error_gen%d.pdf"%(ngen))
    
    plt.figure()
    plt.yscale("log")
    plt.fill_between(gen[1:], dmin[1:], dmax[1:], alpha=0.25, linewidth=0)
    plt.plot(gen[1:],avg[1:])
    plt.title("Average error")
    plt.savefig(run_name+"/avg_error_gen%d.pdf"%(ngen))

    avg,dmin,dmax=log.chapters['size'].select("avg", "min", "max")
    gen=log.select("gen")
    plt.figure()
    plt.yscale("log")
    plt.fill_between(gen[1:], dmin[1:], dmax[1:], alpha=0.25, linewidth=0)
    plt.plot(gen[1:],avg[1:])
    plt.title("Average size")
    plt.savefig(run_name+"/avg_size_gen%d.png"%(ngen))

    avg,dmin,dmax=log.chapters['height'].select("avg", "min", "max")
    gen=log.select("gen")
    plt.figure()
    plt.yscale("log")
    plt.fill_between(gen[1:], dmin[1:], dmax[1:], alpha=0.25, linewidth=0)
    plt.plot(gen[1:],avg[1:])
    plt.title("Average height")
    plt.savefig(run_name+"/avg_height_gen%d.pdf"%(ngen))
        

    # Affichage de quelques résultats sur les meilleurs individus générés ainsi que de l'arbre généré
    # Remarque: pour voir l'affichage de l'arbre, il faut installer pygraphviz (pip install pygraphviz)
    # sur certains OS, il faut au préalable installer graphviz
    for i,ind in enumerate(hof):
        print("=========")
        print("HOF %d, len=%d"%(i,len(ind)))
        print("Error on the training dataset: %f"%(evalSymbReg(ind, input_training, output_training, nb_obj=1)))
        print("Error on the testing dataset: %f"%(evalSymbReg(ind, input_testing, output_testing, nb_obj=1)))

        nodes, edges, labels = gp.graph(ind)


        plt.figure(figsize=(30, 15))

        g = nx.Graph()
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)
        pos = nx.nx_agraph.graphviz_layout(g, prog="dot")
 
        nx.draw_networkx_nodes(g, pos)
        nx.draw_networkx_edges(g, pos)
        nx.draw_networkx_labels(g, pos, labels)
        plt.savefig(run_name+"/hof%d_tree_gen%d.pdf"%(i,ngen))
        plt.show()
        
        moy.append(len(ind))
        
        if abs(evalSymbReg(ind, input_training, output_training, nb_obj=1)[0]) < 0.00001 :
            if (problem == "f1" and len(ind) == 6) or (problem == "f2" and len(ind) == 11):
                nbf += 1
            
            nbp += 1 
                
    
    nbi += 1
    
print("\nAverage size of", NB_CALLS, "calls :", sum(moy) / len(moy))
print("Number of times where the tree is close to f (low error):", nbp)
print("Number of times where f is found and with the right size:", nbf)

