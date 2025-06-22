import random
from deap import tools
from collections import defaultdict, deque
import numpy as np
# from matplotlib import pyplot as plt
from sklearn.svm import LinearSVC
# import pandas as pd
import logging
# from collections import Counter
from sklearn import preprocessing
# import pymoo
import HOF
# from pymoo.algorithms.moo.moead import MOEAD, ParallelMOEAD
# from pymoo.optimize import minimize
# from pymoo.problems import get_problem
from pymoo.util.ref_dirs import get_reference_directions
# from pymoo.visualization.scatter import Scatter
# from pymoo.decomposition.tchebicheff import Tchebicheff
from scipy.spatial.distance import cdist

import math
from collections import Counter
import copy
# pop_size = 100
# ref_dirs = get_reference_directions("uniform", 2, n_partitions=pop_size - 1)
# decomposition = Tchebicheff()
# n_neighbors = 20
# archive = tools.ParetoFront(pop_size)
# ideal_point = 100 * np.ones(2)
# neighbors = np.argsort(cdist(ref_dirs, ref_dirs), axis=1, kind='quicksort')[:, :n_neighbors]
# problem = get_problem("ZDT1")

# ref_dirs = get_reference_directions("uniform", 2, n_partitions=99)
#
# algorithm = MOEAD(
#     ref_dirs,
#     n_neighbors=15,
#     prob_neighbor_mating=0.7,
#     seed=1,
#     verbose=True
# )
#
# res = minimize(problem, algorithm, termination=('n_gen', 50), verbose=True)
# Scatter().add(res.F).show()

# def plot_pop(pop):
#     fitness = [[ind.fitness.values[0], ind.fitness.values[1]] for ind in pop]
#     fitness = np.asarray(fitness)
#     plt.scatter(fitness[:, 0], fitness[:, 1])
#     plt.show()
#     plt.pause(1)

def pop_compare(ind1, ind2):
    # List all available primitive types in each individual
    types1 = defaultdict(list)
    types2 = defaultdict(list)
    for idx, node in enumerate(ind1[1:],1):
        types1[node.ret].append(idx)
    for idx, node in enumerate(ind2[1:],1):
        types2[node.ret].append(idx)
    return types1==types2

def varAnd(population, toolbox, cxpb, mutpb):
    offspring = [toolbox.clone(ind) for ind in population]
    new_cxpb = cxpb/(cxpb+mutpb)
    i = 1
    while i < len(offspring):
        if random.random() < new_cxpb:
            if (offspring[i - 1] == offspring[i]) or pop_compare(offspring[i - 1], offspring[i]):
                offspring[i - 1], = toolbox.mutate(offspring[i - 1])
                offspring[i], = toolbox.mutate(offspring[i])
            else:
                offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1], offspring[i])
            del offspring[i - 1].fitness.values, offspring[i].fitness.values
            i = i + 2
        else:
            offspring[i], = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values
            i = i + 1
    return offspring

def reproduction(population, toolbox, cxpb, mutpb):
    offspring = [toolbox.clone(ind) for ind in population]
    new_cxpb = cxpb/(cxpb+mutpb)
    i = 1
    while i < len(offspring):
        if random.random() < new_cxpb:
            if (offspring[i - 1] == offspring[i]) or pop_compare(offspring[i - 1], offspring[i]):
                offspring[i - 1], = toolbox.mutate(offspring[i - 1])
                offspring[i], = toolbox.mutate(offspring[i])
            else:
                offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1], offspring[i])
            del offspring[i - 1].fitness.values, offspring[i].fitness.values
            i = i + 2
        else:
            offspring[i-1], = toolbox.mutate(offspring[i-1])
            offspring[i], = toolbox.mutate(offspring[i])
            del offspring[i-1].fitness.values, offspring[i].fitness.values
            i = i + 2
    off = [random.choice(offspring)]
    return off


def Tchebicheff(F, weights, ideal_point):
    v = np.abs(F - ideal_point) * weights
    if len(v.shape) < 2:
        tchebi = v.max()
    else:
        tchebi = v.max(axis=1)
    return tchebi


def get_static_directions(n_static=2):
    pi = math.pi
    degree_range = pi/2
    degree_min = degree_range / (n_static-1)
    static_directions = []
    static_degrees = []
    for i in range(n_static):
        degree_dir = i * degree_min
        static_dir = [math.cos(degree_dir)**2, math.sin(degree_dir)**2]
        static_degrees.append(degree_dir)
        static_directions.append(static_dir)
    return np.asarray(static_directions), np.asarray(static_degrees)


def get_dynamic_directions(n_static=2, n_dynamic=2):
    pi = math.pi
    degree_range = (pi/2) / (n_static-1)
    degree_min = degree_range / (n_dynamic+1)
    dynamic_dirs = []
    dynamic_degrees = []
    for i in range(1, n_dynamic+1):
        degree_dir = i * degree_min
        dynamic_dir = [math.cos(degree_dir)**2, math.sin(degree_dir)**2]
        dynamic_dirs.append(dynamic_dir)
        dynamic_degrees.append(degree_dir)
    return np.asarray(dynamic_dirs), np.asarray(dynamic_degrees)


def get_neighbors(degrees, static_degrees, dynamic_degrees, n):
    # degrees = np.concatenate(static_degrees, dynamic_degrees)
    neighbors = []
    degree_min = abs(static_degrees[0] - static_degrees[1])
    range_neighbor = n * degree_min
    for i, degree in enumerate(degrees):
        neighbor = []
        for j, d in enumerate(degrees):
            if abs(degree-d) <= range_neighbor:
                neighbor.append(j)
        neighbors.append(neighbor)
    return neighbors

def get_entropy(data):
    # 统计每种元素的出现次数
    counter = Counter(data)
    # 计算总数
    total_count = len(data)
    # 计算每种元素的概率
    probabilities = [count / total_count for count in counter.values()]
    # 计算信息熵
    entropy = -sum(p * math.log2(p) for p in probabilities)
    return entropy

def get_n_replace(entropy, pop_size, replace_max, gamma):
    # 计算需要替换的个体数量
    # replace_max 是需要替换的最大个体数量
    # entropy 是种群的熵
    # pop_size 是种群规模
    # gamma 是一个阈值
    # 使用sigmoid函数计算需要替换的个体数量，当种群熵小于一定阈值时，需要替换的个体数量较多，反之则较少
    n_replace = int(math.ceil(replace_max / (1 + np.exp(20 * (entropy / math.log(pop_size) - gamma)))))
    return n_replace

def update_n_replace(n_replace, entropy_old, entropy_new, replace_max, replace_min):
    n_replace = n_replace + int(math.ceil(entropy_old - entropy_new))
    n_replace = int(max(replace_min, min(replace_max, n_replace)))
    entropy_old = entropy_new
    return n_replace, entropy_old



def moead(population, toolbox, cxpb, mutpb, elitpb, ngen, pset, stats=None, halloffame=None, verbose=__debug__):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
    pop_size = len(population)
    ref_dirs = get_reference_directions("uniform", 2, n_partitions=pop_size-1)
    # decomposition = Tchebicheff()
    n_neighbors = max(4, int(pop_size // 10))

    # evaluate population
    fitnesses = toolbox.mapp(toolbox.evaluate, population)
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    # store
    pop_store = []
    pop_store.append(copy.deepcopy(population))
    archive = HOF.ParetoFront(len(population))
    archive.update(population)
    ideal_point = 100*np.ones(2)
    neighbors = np.argsort(cdist(ref_dirs, ref_dirs), axis=1, kind='quicksort')[:, :n_neighbors]
    for gen in range(1, ngen + 1):
        for k in np.random.permutation(pop_size):

            # reproduction
            N = neighbors[k]
            parents = np.random.choice(N, size=2, replace=False)
            pop_parents = [population[i] for i in parents]
            off = reproduction(pop_parents, toolbox, cxpb, mutpb)

            # evaluate off
            fitness = toolbox.mapp(toolbox.evaluate, off)
            for ind, fit in zip(off, fitness):
                ind.fitness.values = fit

            # update solutions of neighbors
            pop_neighbors = [population[i] for i in N]
            F_pop_neighbors = [ind.fitness.values for ind in pop_neighbors]
            FV = Tchebicheff(F_pop_neighbors, weights=ref_dirs[N, :], ideal_point=ideal_point)
            F_off = off[0].fitness.values
            off_FV = Tchebicheff(F_off, weights=ref_dirs[N, :], ideal_point=ideal_point)
            I = np.where(off_FV < FV)[0]
            for i in I:
                population[N[i]] = off[0]
        archive.update(population)
        pop_store.append(copy.deepcopy(population))
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(population), **record)
        logging.info(logbook.stream)
    #
    #
    # M = 2 # number of objectives
    # weights = reference_points(M=M, p=references)
    # size = k * weights.shape[0]
    # neighbours = max(4, int(size // 10))
    # population = initial_population(size, min_values, max_values, list_of_functions)
    # print('Total Number of Points on Reference Hyperplane: ', int(weights.shape[0]), ' Population Size: ', int(size))
    # for gen in range(1, ngen + 1):
    #     if (verbose == True):
    #         print('Generation = ', gen)
    #     offspring = breeding(population, neighbours, min_values, max_values, mu, list_of_functions)
    #     offspring = mutation(offspring, mutation_rate, eta, min_values, max_values, list_of_functions)
    #     population = selection(population, offspring, M, weights, theta)

    return population, logbook, archive, pop_store


def moead2(population, toolbox, pset, cxpb=0.8, mutpb=0.2, ngen=50, n_replace=3, stats=None, halloffame=None, verbose=__debug__):
    # moead_2 调整子代替换的邻域大小，匹配最佳权重向量
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
    pop_size = len(population)
    ref_dirs = get_reference_directions("uniform", 2, n_partitions=pop_size-1)
    # decomposition = Tchebicheff()
    n_neighbors = max(4, int(pop_size // 10))
    sigma = 0.85

    # evaluate population
    fitnesses = toolbox.mapp(toolbox.evaluate, population)
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    # store
    pop_store = []
    pop_copy = copy.deepcopy(population)
    pop_store.append(copy.deepcopy(population))
    archive = HOF.ParetoFront(len(population))
    archive.update(population)
    ideal_point = 100*np.ones(2)
    neighbors = np.argsort(cdist(ref_dirs, ref_dirs), axis=1, kind='quicksort')[:, :n_neighbors]
    replace_neighbors = np.argsort(cdist(ref_dirs, ref_dirs), axis=1, kind='quicksort')[:, :n_replace]
    for gen in range(1, ngen + 1):
        for k in np.random.permutation(pop_size):

            # reproduction
            N = neighbors[k]
            # 以一定概率选择邻域内个体作为父代
            if random.random() < sigma:
                parents = np.random.choice(N, size=2, replace=False)
            else:
                parents = np.random.choice(range(pop_size), size=2, replace=False)
            pop_parents = [population[i] for i in parents]
            off = reproduction(pop_parents, toolbox, cxpb, mutpb)

            # evaluate off
            fitness = toolbox.mapp(toolbox.evaluate, off)
            for ind, fit in zip(off, fitness):
                ind.fitness.values = fit

            # find most matched weight vector
            f_w_min = 10000
            w_match = k
            fit_off = off[0].fitness.values
            for i, weight in enumerate(ref_dirs):
                f_w = Tchebicheff(fit_off, weights=weight, ideal_point=ideal_point)
                if f_w < f_w_min:
                    f_w_min = f_w
                    w_match = i

            # replace the matched neighbors
            replace = replace_neighbors[w_match]
            pop_replace = [population[i] for i in replace]
            fit_replace = [ind.fitness.values for ind in pop_replace]
            FW_replace = Tchebicheff(fit_replace, weights=ref_dirs[replace, :], ideal_point=ideal_point)
            FW_off = Tchebicheff(fit_off, weights=ref_dirs[replace, :], ideal_point=ideal_point)
            I = np.where(FW_off < FW_replace)[0]
            for i in I:
                population[replace[i]] = off[0]



            # # update solutions of neighbors
            # pop_neighbors = [population[i] for i in N]
            # F_pop_neighbors = [ind.fitness.values for ind in pop_neighbors]
            # FV = Tchebicheff(F_pop_neighbors, weights=ref_dirs[N, :], ideal_point=ideal_point)
            # F_off = off[0].fitness.values
            # off_FV = Tchebicheff(F_off, weights=ref_dirs[N, :], ideal_point=ideal_point)
            # I = np.where(off_FV < FV)[0]
            # for i in I:
            #     population[N[i]] = off[0]
        archive.update(population)
        # plot_pop(population)
        pop_store.append(copy.deepcopy(population))
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(population), **record)
        logging.info(logbook.stream)

    return population, logbook, archive, pop_store


def prefer_moead2(population, toolbox, cxpb, mutpb, elitpb, ngen, pset, preference, stats=None, halloffame=None, verbose=__debug__):
    # moead_2 调整子代替换的邻域大小，匹配最佳权重向量
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
    pop_size = len(population)
    ref_dirs = get_reference_directions("uniform", 2, n_partitions=pop_size-1)
    # decomposition = Tchebicheff()
    n_neighbors = max(4, int(pop_size // 10))
    n_replace = 3
    sigma = 0.85

    # evaluate population
    fitnesses = toolbox.mapp(toolbox.evaluate, population)
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    # store
    pop_store = []
    pop_copy = copy.deepcopy(population)
    pop_store.append(copy.deepcopy(population))
    archive = HOF.ParetoFront(len(population))
    archive.update(population)
    ideal_point = 100*np.ones(2)
    neighbors = np.argsort(cdist(ref_dirs, ref_dirs), axis=1, kind='quicksort')[:, :n_neighbors]
    replace_neighbors = np.argsort(cdist(ref_dirs, ref_dirs), axis=1, kind='quicksort')[:, :n_replace]
    for gen in range(1, ngen + 1):
        for k in np.random.permutation(pop_size):

            # reproduction
            N = neighbors[k]
            # 以一定概率选择邻域内个体作为父代
            if random.random() < sigma:
                parents = np.random.choice(N, size=2, replace=False)
            else:
                parents = np.random.choice(range(pop_size), size=2, replace=False)
            pop_parents = [population[i] for i in parents]
            off = reproduction(pop_parents, toolbox, cxpb, mutpb)

            # evaluate off
            fitness = toolbox.mapp(toolbox.evaluate, off)
            for ind, fit in zip(off, fitness):
                ind.fitness.values = fit

            # find most matched weight vector
            f_w_min = 10000
            w_match = k
            fit_off = off[0].fitness.values
            for i, weight in enumerate(ref_dirs):
                f_w = Tchebicheff(fit_off, weights=weight*preference, ideal_point=ideal_point)
                if f_w < f_w_min:
                    f_w_min = f_w
                    w_match = i

            # replace the matched neighbors
            replace = replace_neighbors[w_match]
            pop_replace = [population[i] for i in replace]
            fit_replace = [ind.fitness.values for ind in pop_replace]
            FW_replace = Tchebicheff(fit_replace, weights=ref_dirs[replace, :]*preference, ideal_point=ideal_point)
            FW_off = Tchebicheff(fit_off, weights=ref_dirs[replace, :]*preference, ideal_point=ideal_point)
            I = np.where(FW_off < FW_replace)[0]
            for i in I:
                population[replace[i]] = off[0]



            # # update solutions of neighbors
            # pop_neighbors = [population[i] for i in N]
            # F_pop_neighbors = [ind.fitness.values for ind in pop_neighbors]
            # FV = Tchebicheff(F_pop_neighbors, weights=ref_dirs[N, :], ideal_point=ideal_point)
            # F_off = off[0].fitness.values
            # off_FV = Tchebicheff(F_off, weights=ref_dirs[N, :], ideal_point=ideal_point)
            # I = np.where(off_FV < FV)[0]
            # for i in I:
            #     population[N[i]] = off[0]
        archive.update(population)
        # plot_pop(population)
        pop_store.append(copy.deepcopy(population))
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(population), **record)
        logging.info(logbook.stream)

    return population, logbook, archive, pop_store


def adptive_moead2(population, toolbox, cxpb, mutpb, elitpb, ngen, pset, stats=None, halloffame=None, verbose=__debug__):
    # adptive_moead_2 调整子代替换的邻域大小，匹配最佳权重向量
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
    pop_size = len(population)
    ref_dirs = get_reference_directions("uniform", 2, n_partitions=pop_size-1)
    # decomposition = Tchebicheff()
    n_neighbors = max(4, int(pop_size // 10))
    # n_replace = 3
    replace_max = 0.4 * pop_size
    sigma = 0.85
    gamma = 0.25

    # evaluate population
    fitnesses = toolbox.mapp(toolbox.evaluate, population)
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    # store
    pop_store = []
    pop_copy = copy.deepcopy(population)
    pop_store.append(copy.deepcopy(population))
    archive = HOF.ParetoFront(len(population))
    archive.update(population)
    ideal_point = 100*np.ones(2)
    neighbors = np.argsort(cdist(ref_dirs, ref_dirs), axis=1, kind='quicksort')[:, :n_neighbors]
    # replace_neighbors = np.argsort(cdist(ref_dirs, ref_dirs), axis=1, kind='quicksort')[:, :n_replace]

    for gen in range(1, ngen + 1):
        n_replace = int(np.ceil(replace_max/(1+np.exp(-20*(gen/ngen-gamma)))))
        replace_neighbors = np.argsort(cdist(ref_dirs, ref_dirs), axis=1, kind='quicksort')[:, :n_replace]
        for k in np.random.permutation(pop_size):

            # reproduction
            N = neighbors[k]
            # 以一定概率选择邻域内个体作为父代
            if random.random() < sigma:
                parents = np.random.choice(N, size=2, replace=False)
            else:
                parents = np.random.choice(range(pop_size), size=2, replace=False)
            pop_parents = [population[i] for i in parents]
            off = reproduction(pop_parents, toolbox, cxpb, mutpb)

            # evaluate off
            fitness = toolbox.mapp(toolbox.evaluate, off)
            for ind, fit in zip(off, fitness):
                ind.fitness.values = fit

            # find most matched weight vector
            f_w_min = 10000
            w_match = k
            fit_off = off[0].fitness.values
            for i, weight in enumerate(ref_dirs):
                f_w = Tchebicheff(fit_off, weights=weight, ideal_point=ideal_point)
                if f_w < f_w_min:
                    f_w_min = f_w
                    w_match = i

            # replace the matched neighbors
            replace = replace_neighbors[w_match]
            pop_replace = [population[i] for i in replace]
            fit_replace = [ind.fitness.values for ind in pop_replace]
            FW_replace = Tchebicheff(fit_replace, weights=ref_dirs[replace, :], ideal_point=ideal_point)
            FW_off = Tchebicheff(fit_off, weights=ref_dirs[replace, :], ideal_point=ideal_point)
            I = np.where(FW_off < FW_replace)[0]
            for i in I:
                population[replace[i]] = off[0]



            # # update solutions of neighbors
            # pop_neighbors = [population[i] for i in N]
            # F_pop_neighbors = [ind.fitness.values for ind in pop_neighbors]
            # FV = Tchebicheff(F_pop_neighbors, weights=ref_dirs[N, :], ideal_point=ideal_point)
            # F_off = off[0].fitness.values
            # off_FV = Tchebicheff(F_off, weights=ref_dirs[N, :], ideal_point=ideal_point)
            # I = np.where(off_FV < FV)[0]
            # for i in I:
            #     population[N[i]] = off[0]
        archive.update(population)
        # plot_pop(population)
        pop_store.append(copy.deepcopy(population))
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(population), **record)
        logging.info(logbook.stream)

    return population, logbook, archive, pop_store


def entropy_moead2(population, toolbox, cxpb, mutpb, elitpb, ngen, pset, stats=None, halloffame=None, verbose=__debug__):
    # entropy_moead_2 调整子代替换的邻域大小，匹配最佳权重向量
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
    pop_size = len(population)
    ref_dirs = get_reference_directions("uniform", 2, n_partitions=pop_size-1)
    # decomposition = Tchebicheff()
    n_neighbors = max(4, int(pop_size // 10))
    # n_replace = 3
    replace_max = 0.4 * pop_size
    replace_min = 1
    n_replace = replace_min
    entropy_old = 0


    sigma = 0.85
    gamma = 0.5

    # evaluate population
    fitnesses = toolbox.mapp(toolbox.evaluate, population)
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    # store
    pop_store = []
    n_replace_all = []
    entropy_all = []
    pop_copy = copy.deepcopy(population)
    pop_store.append(copy.deepcopy(population))
    archive = HOF.ParetoFront(len(population))
    archive.update(population)
    ideal_point = 100*np.ones(2)
    neighbors = np.argsort(cdist(ref_dirs, ref_dirs), axis=1, kind='quicksort')[:, :n_neighbors]
    # replace_neighbors = np.argsort(cdist(ref_dirs, ref_dirs), axis=1, kind='quicksort')[:, :n_replace]

    for gen in range(1, ngen + 1):
        fit_entropy = [ind.fitness.values[0] for ind in population]
        entropy = get_entropy(fit_entropy)
        # n_replace = int(math.ceil(replace_max/(1+np.exp(20*(entropy/math.log(pop_size)-gamma)))))
        n_replace, entropy_old = update_n_replace(n_replace, entropy_old, entropy, replace_max, replace_min)
        n_replace_all.append(n_replace)
        entropy_all.append(entropy)
        replace_neighbors = np.argsort(cdist(ref_dirs, ref_dirs), axis=1, kind='quicksort')[:, :n_replace]
        for k in np.random.permutation(pop_size):

            # reproduction
            N = neighbors[k]
            # 以一定概率选择邻域内个体作为父代
            if random.random() < sigma:
                parents = np.random.choice(N, size=2, replace=False)
            else:
                parents = np.random.choice(range(pop_size), size=2, replace=False)
            pop_parents = [population[i] for i in parents]
            off = reproduction(pop_parents, toolbox, cxpb, mutpb)

            # evaluate off
            fitness = toolbox.mapp(toolbox.evaluate, off)
            for ind, fit in zip(off, fitness):
                ind.fitness.values = fit

            # find most matched weight vector
            f_w_min = 10000
            w_match = k
            fit_off = off[0].fitness.values
            for i, weight in enumerate(ref_dirs):
                f_w = Tchebicheff(fit_off, weights=weight, ideal_point=ideal_point)
                if f_w < f_w_min:
                    f_w_min = f_w
                    w_match = i

            # replace the matched neighbors
            replace = replace_neighbors[w_match]
            pop_replace = [population[i] for i in replace]
            fit_replace = [ind.fitness.values for ind in pop_replace]
            FW_replace = Tchebicheff(fit_replace, weights=ref_dirs[replace, :], ideal_point=ideal_point)
            FW_off = Tchebicheff(fit_off, weights=ref_dirs[replace, :], ideal_point=ideal_point)
            I = np.where(FW_off < FW_replace)[0]
            for i in I:
                population[replace[i]] = off[0]



            # # update solutions of neighbors
            # pop_neighbors = [population[i] for i in N]
            # F_pop_neighbors = [ind.fitness.values for ind in pop_neighbors]
            # FV = Tchebicheff(F_pop_neighbors, weights=ref_dirs[N, :], ideal_point=ideal_point)
            # F_off = off[0].fitness.values
            # off_FV = Tchebicheff(F_off, weights=ref_dirs[N, :], ideal_point=ideal_point)
            # I = np.where(off_FV < FV)[0]
            # for i in I:
            #     population[N[i]] = off[0]
        archive.update(population)
        # plot_pop(population)
        pop_store.append(copy.deepcopy(population))
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(population), **record)
        logging.info(logbook.stream)

    return population, logbook, archive, pop_store, n_replace_all, entropy_all


def moead3(population, toolbox, cxpb, mutpb, elitpb, ngen, pset, stats=None, halloffame=None, verbose=__debug__):
    # moead_3 调整子代替换的邻域大小，匹配最佳权重向量
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
    pop_size = len(population)
    ref_dirs = get_reference_directions("uniform", 2, n_partitions=pop_size - 1)
    # decomposition = Tchebicheff()
    n_neighbors = max(4, int(pop_size // 10))
    n_replace = 3
    sigma = 0.85

    # evaluate population
    fitnesses = toolbox.mapp(toolbox.evaluate, population)
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    # store
    pop_store = []
    pop_copy = copy.deepcopy(population)
    pop_store.append(copy.deepcopy(population))
    archive = HOF.ParetoFront(len(population))
    archive.update(population)
    ideal_point = 100 * np.ones(2)
    neighbors = np.argsort(cdist(ref_dirs, ref_dirs), axis=1, kind='quicksort')[:, :n_neighbors]
    replace_neighbors = np.argsort(cdist(ref_dirs, ref_dirs), axis=1, kind='quicksort')[:, :n_replace]
    for gen in range(1, ngen + 1):
        # generate the next generation individuals
        offspring = varAnd(population, toolbox, cxpb, mutpb)
        # evaluate offspring
        fitnesses = toolbox.mapp(toolbox.evaluate, offspring)
        for ind, fit in zip(offspring, fitnesses):
            ind.fitness.values = fit

        candidates = population + offspring

        # 为每个子问题挑选最优解
        best_individuals = []
        for weight in ref_dirs:
            f_candidates = []
            for ind in candidates:
                f_candidate = Tchebicheff(ind.fitness.values, weights=weight, ideal_point=ideal_point)
                f_candidates.append(f_candidate)

            i = np.argmin(f_candidates)
            best_individuals.append(candidates[i])
            del candidates[i]

        # 去除重复个体好像不需要了
        # 更新种群
        population = best_individuals
        archive.update(population)

        pop_store.append(copy.deepcopy(population))
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(population), **record)
        logging.info(logbook.stream)

    return population, logbook, archive, pop_store


def entropy_mate1_moead2(population, toolbox, cxpb, mutpb, elitpb, ngen, pset, stats=None, halloffame=None, verbose=__debug__):
    # entropy_mate1_moead2 调整子代替换的邻域大小，匹配最佳权重向量;
    # 基于子问题上精英策略的父代选择（挑子问题上性能好的个体作为父代）按概率从邻域和最优个体选
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
    pop_size = len(population)
    ref_dirs = get_reference_directions("uniform", 2, n_partitions=pop_size-1)
    # decomposition = Tchebicheff()
    n_neighbors = max(4, int(pop_size // 10))
    # n_replace = 3
    replace_max = 0.3 * pop_size
    replace_min = 1
    n_replace = replace_min
    entropy_old = 0


    sigma = 0.5
    gamma = 0.5

    # evaluate population
    fitnesses = toolbox.mapp(toolbox.evaluate, population)
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    # store
    pop_store = []
    n_replace_all = []
    entropy_all = []
    pop_copy = copy.deepcopy(population)
    pop_store.append(copy.deepcopy(population))
    archive = HOF.ParetoFront(len(population))
    archive.update(population)
    ideal_point = 100*np.ones(2)
    neighbors = np.argsort(cdist(ref_dirs, ref_dirs), axis=1, kind='quicksort')[:, :n_neighbors]
    # replace_neighbors = np.argsort(cdist(ref_dirs, ref_dirs), axis=1, kind='quicksort')[:, :n_replace]

    for gen in range(1, ngen + 1):
        fit_entropy = [ind.fitness.values[0] for ind in population]
        entropy = get_entropy(fit_entropy)
        # n_replace = int(math.ceil(replace_max/(1+np.exp(20*(entropy/math.log(pop_size)-gamma)))))
        n_replace, entropy_old = update_n_replace(n_replace, entropy_old, entropy, replace_max, replace_min)
        n_replace_all.append(n_replace)
        entropy_all.append(entropy)
        replace_neighbors = np.argsort(cdist(ref_dirs, ref_dirs), axis=1, kind='quicksort')[:, :n_replace]

        # 获取父代种群的适应度值
        fit_parents = [[ind.fitness.values[0], ind.fitness.values[1]] for ind in population]
        for k in np.random.permutation(pop_size):

            # reproduction
            N = neighbors[k]
            # 以一定概率选择邻域内个体作为父代
            if random.random() < sigma:
                parents = np.random.choice(N, size=2, replace=False)
            else:
                # 计算子问题k的适应度值
                f_k = []
                for i, fit_parent in enumerate(fit_parents):
                    f = Tchebicheff(fit_parent, weights=ref_dirs[k, :], ideal_point=ideal_point)
                    f_k.append(f)
                candidates = np.argsort(f_k)   # 升序排序后的索引
                parents = np.random.choice(candidates[:n_neighbors], size=2, replace=False)  # 选出父代的索引
            pop_parents = [population[i] for i in parents]
            off = reproduction(pop_parents, toolbox, cxpb, mutpb)

            # evaluate off
            fitness = toolbox.mapp(toolbox.evaluate, off)
            for ind, fit in zip(off, fitness):
                ind.fitness.values = fit

            # find most matched weight vector
            f_w_min = 10000
            w_match = k
            fit_off = off[0].fitness.values
            for i, weight in enumerate(ref_dirs):
                f_w = Tchebicheff(fit_off, weights=weight, ideal_point=ideal_point)
                if f_w < f_w_min:
                    f_w_min = f_w
                    w_match = i

            # replace the matched neighbors
            replace = replace_neighbors[w_match]
            pop_replace = [population[i] for i in replace]
            fit_replace = [ind.fitness.values for ind in pop_replace]
            FW_replace = Tchebicheff(fit_replace, weights=ref_dirs[replace, :], ideal_point=ideal_point)
            FW_off = Tchebicheff(fit_off, weights=ref_dirs[replace, :], ideal_point=ideal_point)
            I = np.where(FW_off < FW_replace)[0]
            for i in I:
                population[replace[i]] = off[0]



            # # update solutions of neighbors
            # pop_neighbors = [population[i] for i in N]
            # F_pop_neighbors = [ind.fitness.values for ind in pop_neighbors]
            # FV = Tchebicheff(F_pop_neighbors, weights=ref_dirs[N, :], ideal_point=ideal_point)
            # F_off = off[0].fitness.values
            # off_FV = Tchebicheff(F_off, weights=ref_dirs[N, :], ideal_point=ideal_point)
            # I = np.where(off_FV < FV)[0]
            # for i in I:
            #     population[N[i]] = off[0]
        archive.update(population)
        # plot_pop(population)
        pop_store.append(copy.deepcopy(population))
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(population), **record)
        logging.info(logbook.stream)

    return population, logbook, archive, pop_store, n_replace_all, entropy_all

def aw_moead(population, toolbox, cxpb, mutpb, elitpb, ngen, pset, stats=None, halloffame=None, verbose=__debug__):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
    pop_size = len(population)
    # ref_dirs = get_reference_directions("uniform", 2, n_partitions=pop_size-1)
    n_s = int(pop_size * 0.9)
    # n_d = int(pop_size * 0.1)
    n_d = pop_size - n_s
    static_dirs, static_degrees = get_static_directions(n_s)
    dynamic_dirs, dynamic_degrees = get_dynamic_directions(n_s, n_d)
    n_neighbors = max(4, int(pop_size // 20))
    degrees = np.concatenate((static_degrees, dynamic_degrees))
    neighbors = get_neighbors(degrees, static_degrees, dynamic_degrees, n_neighbors)
    ref_dirs = np.concatenate((static_dirs, dynamic_dirs))
    # evaluate population
    fitnesses = toolbox.mapp(toolbox.evaluate, population)
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    # store
    pop_store = []
    pop_store.append(copy.deepcopy(population))
    archive = HOF.ParetoFront(len(population))
    archive.update(population)
    ideal_point = 100*np.ones(2)
    # neighbors = np.argsort(cdist(ref_dirs, ref_dirs), axis=1, kind='quicksort')[:, :n_neighbors]
    for gen in range(1, ngen + 1):
        for k in np.random.permutation(pop_size):

            # reproduction
            N = neighbors[k]
            parents = np.random.choice(N, size=2, replace=False)
            pop_parents = [population[i] for i in parents]
            off = reproduction(pop_parents, toolbox, cxpb, mutpb)

            # evaluate off
            fitness = toolbox.mapp(toolbox.evaluate, off)
            for ind, fit in zip(off, fitness):
                ind.fitness.values = fit

            # update solutions of neighbors
            pop_neighbors = [population[i] for i in N]
            F_pop_neighbors = [ind.fitness.values for ind in pop_neighbors]
            FV = Tchebicheff(F_pop_neighbors, weights=ref_dirs[N, :], ideal_point=ideal_point)
            F_off = off[0].fitness.values
            off_FV = Tchebicheff(F_off, weights=ref_dirs[N, :], ideal_point=ideal_point)
            I = np.where(off_FV < FV)[0]
            for i in I:
                population[N[i]] = off[0]
        archive.update(population)
        pop_store.append(copy.deepcopy(population))
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(population), **record)
        logging.info(logbook.stream)
    return population, logbook, archive, pop_store


def acc_lsvm(x_train, x_test, y_train, y_test):
    min_max_scaler = preprocessing.MinMaxScaler()
    train_norm = min_max_scaler.fit_transform(np.asarray(x_train))
    test_norm = min_max_scaler.transform(np.asarray(x_test))
    lsvm = LinearSVC()
    lsvm.fit(train_norm, y_train)
    acc = round(100 * lsvm.score(test_norm, y_test), 2)
    return acc

def lsvm_test(ind_global, ind_local, toolbox, x_train, y_train, x_test, y_test, task):
    if task=='task1':
        func_global = toolbox.compile(expr=ind_global)
        func_local = toolbox.compile(expr=ind_local)
    else:
        func_global = toolbox.compile(expr=ind_global)
        func_local = toolbox.compile(expr=ind_local)
    train_g_tf = []
    train_l_tf = []
    print(ind_global,ind_local)
    for i in range(0, len(y_train)):
        train_g_tf.append(np.asarray(func_global(x_train[i, :, :])))
    for i in range(0, len(y_train)):
        train_l_tf.append(np.asarray(func_local(x_train[i, :, :])))
    train_tf = np.concatenate((np.asarray(train_g_tf), np.asarray(train_l_tf)), axis=1)
    test_g_tf = []
    test_l_tf = []
    # print(individual[0], individual[1])
    for i in range(0, len(y_test)):
        test_g_tf.append(np.asarray(func_global(x_test[i, :, :])))
    for i in range(0, len(y_test)):
        test_l_tf.append(np.asarray(func_local(x_test[i, :, :])))
    test_tf = np.concatenate((np.asarray(test_g_tf), np.asarray(test_l_tf)), axis=1)
    acc_global = acc_lsvm(train_g_tf, test_g_tf, y_train, y_test)
    acc_local = acc_lsvm(train_l_tf, test_l_tf, y_train, y_test)
    min_max_scaler = preprocessing.MinMaxScaler()
    train_norm = min_max_scaler.fit_transform(train_tf)
    test_norm = min_max_scaler.transform(test_tf)
    lsvm = LinearSVC()
    lsvm.fit(train_norm, y_train)
    acc = round(100 * lsvm.score(test_norm, y_test), 2)
    y_predict = lsvm.predict(test_norm)
    del lsvm
    return train_norm.shape[1], acc, acc_global, acc_local, y_predict



