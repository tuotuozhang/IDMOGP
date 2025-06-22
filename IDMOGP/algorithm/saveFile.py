import pickle
from deap import gp

def saveResults(fileName, *args, **kwargs):
    f = open(fileName, 'w')
    for i in args:
        f.writelines(str(i) + '\n')
    f.close()
    return


def saveLog(fileName, log):
    f = open(fileName, 'wb')
    pickle.dump(log, f)
    f.close()
    return


def bestInd(toolbox, population, number):
    bestInd = []
    best = toolbox.selectElitism(population, k=number)
    for i in best:
        bestInd.append(i)
    return bestInd


def saveAllResults(randomSeeds, dataSetName, best_ind_va, log, hof, num_features, trainTime, testTime, testResults):
    fileName1 = str(randomSeeds) + 'Resultson' + dataSetName + '.txt'
    saveLog(fileName1, log)
    fileName2 = str(randomSeeds) + 'FinalResultson' + dataSetName + '.txt'
    saveResults(fileName2, 'randomSeed', randomSeeds, 'trainTime', trainTime,
                'trainResults', hof[0].fitness, 'Number of features', num_features,
                'testTime', testTime, 'testResults', testResults, 'bestInd in training',
                hof[0], 'Best individual in each run',
                *hof[:], 'final best fitness', hof[-1].fitness,
                'initial fitness', hof[0].fitness)
    return

def saveAllResults_fgp8(randomSeeds, dataSetName, best_ind_va, log, hof, num_features, trainTime, testTime, testResults):
    fileName1 = str(randomSeeds) + 'Resultson' + dataSetName + '.txt'
    saveLog(fileName1, log)
    fileName2 = str(randomSeeds) + 'FinalResultson' + dataSetName + '.txt'
    saveResults(fileName2, 'randomSeed', randomSeeds, 'trainTime', trainTime,
                'trainResults', best_ind_va.fitness, 'Number of features', num_features,
                'testTime', testTime, 'testResults', testResults, 'bestInd in training',
                hof[0], 'Best individual in each run',
                *hof[:], 'final best fitness', hof[-1].fitness,
                'initial fitness', hof[0].fitness)

    return

def savettMT1(randomSeeds,dataSetName,log,best_pop,pop, best_train,num_features,trainTime,testTime,testResults,testResults_Ensemble, testResults_Rank,num_f_rank):
    fileName1= str(randomSeeds)+'Resultson' + dataSetName+ '.txt'
    saveLog(fileName1, log)
    fileName2=str(randomSeeds)+'FinalResultson' + dataSetName+ '.txt'
    saveResults(fileName2, 'randomSeed', randomSeeds, 'trainTime', trainTime,
                         'trainResults', best_train, 'number features', num_features,
                         'testTime', testTime, 'testResults', testResults, 'testResults Ensemble', testResults_Ensemble,'testResults Rank', testResults_Rank,
                'num_features Rank', num_f_rank,'bestInd in training (global and local)',best_pop[0], best_pop[1],'bestInd in training (global )',*pop[:][0],'bestInd in training (local )', *pop[:][1])
    return

def savettMT(randomSeeds,dataSetName,log,best_pop,pop, best_train,num_features,trainTime,testTime,testResults):
    fileName1= str(randomSeeds)+'Resultson' + dataSetName+ '.txt'
    saveLog(fileName1, log)
    fileName2=str(randomSeeds)+'FinalResultson' + dataSetName+ '.txt'
    saveResults(fileName2, 'randomSeed', randomSeeds, 'trainTime', trainTime,
                         'trainResults', best_train, 'number features', num_features,
                         'testTime', testTime, 'testResults', testResults,'bestInd in training (global and local)',
                         best_pop[0], best_pop[1],'bestInd in training (global )',*pop[:][0],'bestInd in training (local )', *pop[:][1])
    return

def savettMT2(randomSeeds,dataSetName,log,best_pop, best_train,num_features,trainTime,testTime,testResults,testResults_global,  testResults_local):
    fileName1= str(randomSeeds)+'Resultson' + dataSetName+ '.txt'
    saveLog(fileName1, log)
    fileName2=str(randomSeeds)+'FinalResultson' + dataSetName+ '.txt'
    saveResults(fileName2, 'randomSeed', randomSeeds, 'trainTime', trainTime,
                         'trainResults', best_train, 'number features', num_features,
                         'testTime', testTime, 'testResults', testResults, 'testResults Global Tree', testResults_global,'testResults Local Tree', testResults_local,
                'bestInd in training (global and local)',best_pop[0], best_pop[1])
    return