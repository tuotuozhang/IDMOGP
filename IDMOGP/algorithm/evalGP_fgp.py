import random
from deap import tools
from collections import defaultdict, deque
import numpy as np
from sklearn.svm import LinearSVC
from sklearn import preprocessing  ##
from sklearn.metrics import confusion_matrix
import pandas as pd
import logging
from collections import Counter


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
    new_cxpb=cxpb/(cxpb+mutpb)
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


def eaSimple(population, toolbox, cxpb, mutpb, elitpb, ngen , stats=None,
             halloffame=None, verbose=__debug__):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.mapp(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    halloffame.update(population)
    hof_store = tools.HallOfFame(5 * len(population))
    hof_store.update(population)
    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(population), **record)
    logging.info(logbook.stream)
    best_ind_over = []
    best_ind_over.append(halloffame[0])
    for gen in range(1, ngen + 1):
        #Select the next generation individuals by elitism
        elitismNum=int(elitpb * len(population))
        population_for_eli=[toolbox.clone(ind) for ind in population]
        offspringE = toolbox.selectElitism(population_for_eli, k=elitismNum)
        # Select the next generation individuals for crossover and mutation
        offspring = toolbox.select(population, len(population)-elitismNum)
        # Vary the pool of individuals
        offspring = varAnd(offspring, toolbox, cxpb, mutpb)
        # add offspring from elitism into current offspring
        #generate the next generation individuals
            
        # Evaluate the individuals with an invalid fitness
        for i in offspring:
            ind = 0
            while ind<len(hof_store):
                if i == hof_store[ind]:
                    i.fitness.values = hof_store[ind].fitness.values
                    ind = len(hof_store)
                else:
                    ind+=1
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.mapp(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        offspring[0:0]=offspringE
            
        # Update the hall of fame with the generated
        halloffame.update(offspring)
        best_ind_over.append(halloffame[0])
        cop_po = offspring.copy()
        hof_store.update(offspring)
        for i in hof_store:
            cop_po.append(i)
        population[:] = offspring
        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(offspring), **record)
        logging.info(logbook.stream)
    return population, logbook, best_ind_over

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


def voting_vector(args):
    # print(args)
    x = Counter(args)
    # print(x.most_common(1)[0][0])
    return x.most_common(1)[0][0]

def voting_array(predictions):
    predict_ensemble = []
    predict= np.asarray(predictions)
    # print(predict.shape)
    for i in range(predict.shape[1]):
        labelss = predict[:, i].tolist()
        # print(labelss, type(labelss))
        label = voting_vector(labelss)
        predict_ensemble.append(label)
    return predict_ensemble

def test_task_all(hof, toolbox, randomseed, dataSet, x_train, y_train, x_test, y_test, task):
    all_results = pd.DataFrame()
    num_features = []
    acc_all=[]
    acc_global = []
    acc_loal =[]
    train_overall=[]
    train_other_task  =[]
    y_predict_all = []
    pop = []
    for i in range(len(hof)):
        if task =='task1':
            pop.append([hof[i][0], hof[i][1]])
            num_f,  acc_overall, acc_g, acc_l , y_predict= lsvm_test(hof[i][1], hof[i][0], toolbox, x_train, y_train, x_test, y_test, task)
            train_overall.append(hof[i].fitness.values[0])
            train_other_task.append(hof[i].fitness.values[1])
            y_predict_all.append(y_predict.tolist())

        else:
            pop.append([hof[i][0], hof[i][2]])
            num_f, acc_overall, acc_g, acc_l, y_predict = lsvm_test(hof[i][1], hof[i][0], toolbox, x_train, y_train,x_test, y_test, task)
            train_overall.append(hof[i].fitness.values[1])
            train_other_task.append(hof[i].fitness.values[0])
            y_predict_all.insert(0, y_predict.tolist())
        # print(y_predict.shape, type(y_predict))
        num_features.append(num_f)
        acc_all.append(acc_overall)
        acc_global.append(acc_g)
        acc_loal.append(acc_l)
    all_results['Train'] = train_overall
    all_results['OtherTrain'] = train_other_task
    all_results['Test'] = acc_all
    all_results['GTest'] = acc_global
    all_results['TTest'] = acc_loal
    all_results['numF'] = num_features
    all_results.to_csv(str(randomseed)+dataSet+'_allResults.csv')
    if  task =='task1':
        acc_single = acc_all[0]
        num_f_single = num_features[0]
    else:
        acc_single = acc_all[-1]
        num_f_single = num_features[-1]
    # print(y_predict_all)
    predict_en = voting_array(y_predict_all)
    acc_en = round(100*np.sum (np.asarray(predict_en)==y_test)/len(y_test),2)
    return num_f_single, acc_single, pop, acc_en
