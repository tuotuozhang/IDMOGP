# python packages
from matplotlib import pyplot as plt
import evalGP_moead as evalGP
# from scoop import futures
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing  ##
import logging
import gp_tree
from deal_with_data2 import deal_data2

# only for strongly typed GP
# fitness function
from FEVal_norm_fast import evalTest_fromvector_FGP as evalTest
from FEVal_norm_fast import feature_length
##plot tree and save
##image Data
# defined by author

import random
import time
import gp_restrict
import numpy
import numpy as np
from deap import base, creator, tools, gp
from deap.benchmarks.tools import hypervolume
from strongGPDataType_FGP import Int1, Int2, Int3, Float1, Float2, Float3, Img, Img1, Vector, Vector1
import fgp_functions as fe_fs
from sklearn.svm import LinearSVC
from sklearn import preprocessing
import saveFile_FGP as saveFile
import sys
# randomSeeds = int(sys.argv[1])
# dataSetName = str(sys.argv[2])
# num_instance = int(sys.argv[3])

randomSeeds = 2
dataSetName = 'att'
num_instance = 3
# data_path = '/vol/grid-solar/sgeusers/yingbi/dataset_npy/multi_small'
# data_path = '/nesi/nobackup/nesi00416/datasets/multi_small/'
# data_path = '/nfs/scratch/biyi/datasets/'
# data_path = '/Users/vuw/my_code/dataset/popular/'

# data_path = '/vol/grid-solar/sgeusers/yingbi/dataset_npy/popular/'
# save_path = '/vol/grid-solar/sgeusers/yingbi/MOGP_TUO/results/fgp_entropy_R_moead2/'
data_path = '../popular/'
save_path = '../results/fgp_entropy_R_moead2/'

def load_data(dataset_name, path=None):
    if path is not None:
        file = path + '/' + dataset_name + '/' + dataset_name
    else:
        file = dataset_name
    x_train = numpy.load(file + '_train_data.npy') / 255.0
    y_train = numpy.load(file + '_train_label.npy')
    x_test = numpy.load(file + '_test_data.npy') / 255.0
    y_test = numpy.load(file + '_test_label.npy')
    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = load_data(dataSetName, path=data_path)
n_class = numpy.unique(y_train).shape[0]
base_line_acc = 100 / n_class
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
# nui = numpy.unique(y_train)
# num_instance = len(y_train[y_train == nui[0]])
n_cv = min(5, num_instance)
nf_size = x_train.shape[1]*x_train.shape[2]
x_train, x_test, y_train, y_test = deal_data2(x_train, y_train, x_test, y_test, dataSetName, num_instance, n_class)


file_name = str(randomSeeds) + '_n' + str(num_instance) + '_' + str(dataSetName) + '_'
#################################################################################################################################
# parameters:
population = 100
generation = 50
cxProb = 0.8
mutProb = 0.2
elitismProb = 0.0
totalRuns = 1
initialMinDepth = 2
initialMaxDepth = 8
maxDepth = 10

# logging results
logging.basicConfig(level=logging.INFO, filename=file_name + 'fgp_entropy_R_moead2_main.log')
logging.info('#############Start##############')
logging.info('Algorithm name: fgp_entropy_R_moead2.py ')
logging.info('Training set shape: ' + str(x_train.shape))
logging.info('Test set shape: ' + str(x_test.shape))
logging.info('population: ' + str(population))
logging.info('generation: ' + str(generation))
logging.info('cxProb: ' + str(cxProb))
logging.info('mutProb: ' + str(mutProb))
logging.info('elitismProb: ' + str(elitismProb))
logging.info('initialMinDepth: ' + str(initialMinDepth))
logging.info('initialMaxDepth: ' + str(initialMaxDepth))
logging.info('maxDepth: ' + str(maxDepth))
##GP
pset = gp.PrimitiveSetTyped('MAIN', [Img], Vector1, prefix='Image')
#feature concatenation
pset.addPrimitive(fe_fs.root_con, [Vector1, Vector1], Vector1, name='Root')
pset.addPrimitive(fe_fs.root_conVector2, [Img1, Img1], Vector1, name='Root2')
pset.addPrimitive(fe_fs.root_conVector3, [Img1, Img1, Img1], Vector1, name='Root3')
pset.addPrimitive(fe_fs.root_con, [Vector, Vector], Vector1, name='Roots2')
pset.addPrimitive(fe_fs.root_con, [Vector, Vector, Vector], Vector1, name='Roots3')
pset.addPrimitive(fe_fs.root_con, [Vector, Vector, Vector, Vector], Vector1, name='Roots4')
##feature extraction
pset.addPrimitive(fe_fs.global_hog_small, [Img1], Vector, name='Global_HOG')
pset.addPrimitive(fe_fs.all_lbp, [Img1], Vector, name='Global_uLBP')
pset.addPrimitive(fe_fs.all_sift, [Img1], Vector, name='Global_SIFT')
pset.addPrimitive(fe_fs.global_hog_small, [Img], Vector, name='FGlobal_HOG')
pset.addPrimitive(fe_fs.all_lbp, [Img], Vector, name='FGlobal_uLBP')
pset.addPrimitive(fe_fs.all_sift, [Img], Vector, name='FGlobal_SIFT')
# pooling
pset.addPrimitive(fe_fs.maxP, [Img1, Int3, Int3], Img1, name='MaxPF')
#filtering
pset.addPrimitive(fe_fs.gau, [Img1, Int1], Img1, name='GauF')
pset.addPrimitive(fe_fs.gauD, [Img1, Int1, Int2, Int2], Img1, name='GauDF')
pset.addPrimitive(fe_fs.gab, [Img1, Float1, Float2], Img1, name='GaborF')
pset.addPrimitive(fe_fs.laplace, [Img1], Img1, name='LapF')
pset.addPrimitive(fe_fs.gaussian_Laplace1, [Img1], Img1, name='LoG1F')
pset.addPrimitive(fe_fs.gaussian_Laplace2, [Img1], Img1, name='LoG2F')
pset.addPrimitive(fe_fs.sobelxy, [Img1], Img1, name='SobelF')
pset.addPrimitive(fe_fs.sobelx, [Img1], Img1, name='SobelXF')
pset.addPrimitive(fe_fs.sobely, [Img1], Img1, name='SobelYF')
pset.addPrimitive(fe_fs.medianf, [Img1], Img1, name='MedF')
pset.addPrimitive(fe_fs.meanf, [Img1], Img1, name='MeanF')
pset.addPrimitive(fe_fs.minf, [Img1], Img1, name='MinF')
pset.addPrimitive(fe_fs.maxf, [Img1], Img1, name='MaxF')
pset.addPrimitive(fe_fs.lbp, [Img1], Img1, name='LBPF')
pset.addPrimitive(fe_fs.hog_feature, [Img1], Img1, name='HoGF')
pset.addPrimitive(fe_fs.mixconadd, [Img1, Float3, Img1, Float3], Img1, name='W_AddF')
pset.addPrimitive(fe_fs.mixconsub, [Img1, Float3, Img1, Float3], Img1, name='W_SubF')
pset.addPrimitive(fe_fs.sqrt, [Img1], Img1, name='SqrtF')
pset.addPrimitive(fe_fs.relu, [Img1], Img1, name='ReLUF')
# pooling
pset.addPrimitive(fe_fs.maxP, [Img, Int3, Int3], Img1, name='MaxP')
# filtering
pset.addPrimitive(fe_fs.gau, [Img, Int1], Img, name='Gau')
pset.addPrimitive(fe_fs.gauD, [Img, Int1, Int2, Int2], Img, name='GauD')
pset.addPrimitive(fe_fs.gab, [Img, Float1, Float2], Img, name='Gabor')
pset.addPrimitive(fe_fs.laplace, [Img], Img, name='Lap')
pset.addPrimitive(fe_fs.gaussian_Laplace1, [Img], Img, name='LoG1')
pset.addPrimitive(fe_fs.gaussian_Laplace2, [Img], Img, name='LoG2')
pset.addPrimitive(fe_fs.sobelxy, [Img], Img, name='Sobel')
pset.addPrimitive(fe_fs.sobelx, [Img], Img, name='SobelX')
pset.addPrimitive(fe_fs.sobely, [Img], Img, name='SobelY')
pset.addPrimitive(fe_fs.medianf, [Img], Img, name='Med')
pset.addPrimitive(fe_fs.meanf, [Img], Img, name='Mean')
pset.addPrimitive(fe_fs.minf, [Img], Img, name='Min')
pset.addPrimitive(fe_fs.maxf, [Img], Img, name='Max')
pset.addPrimitive(fe_fs.lbp, [Img], Img, name='LBP_F')
pset.addPrimitive(fe_fs.hog_feature, [Img], Img, name='HOG_F')
pset.addPrimitive(fe_fs.mixconadd, [Img, Float3, Img, Float3], Img, name='W_Add')
pset.addPrimitive(fe_fs.mixconsub, [Img, Float3, Img, Float3], Img, name='W_Sub')
pset.addPrimitive(fe_fs.sqrt, [Img], Img, name='Sqrt')
pset.addPrimitive(fe_fs.relu, [Img], Img, name='ReLU')
# Terminals
pset.renameArguments(ARG0='Image')
pset.addEphemeralConstant('Singma', lambda: random.randint(1, 4), Int1)
pset.addEphemeralConstant('Order', lambda: random.randint(0, 3), Int2)
pset.addEphemeralConstant('Theta', lambda: random.randint(0, 8), Float1)
pset.addEphemeralConstant('Frequency', lambda: random.randint(0, 5), Float2)
pset.addEphemeralConstant('n', lambda: round(random.random(), 3), Float3)
pset.addEphemeralConstant('KernelSize', lambda: random.randrange(2, 5, 2), Int3)



##
creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
toolbox.register("expr", gp_restrict.genHalfAndHalfMD, pset=pset, min_=initialMinDepth, max_=initialMaxDepth)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)
# toolbox.register("mapp", futures.map)
toolbox.register("mapp", map)


def between_with_class(transformed_data, y_train):
    n_class = len(numpy.unique(y_train))
    bc = 0
    wc = 0
    global_mean = numpy.mean(transformed_data, axis=0)
    # print(transformed_data.shape)
    for i in range(n_class):
        if transformed_data.shape[1] == 1:
            one_class = transformed_data[y_train == i, :]
            class_mean = numpy.mean(one_class)
            bc += (class_mean - global_mean[0]) ** 2
            # print(class_mean, global_mean)
            # print(bc)
        else:
            one_class = transformed_data[y_train == i, :]
            class_mean = numpy.mean(one_class, axis=0)
            bc += numpy.sum((class_mean - global_mean) ** 2)
        wc += numpy.sum((one_class - class_mean) ** 2)
    bc = bc / n_class
    wc = wc / len(y_train)
    # bc: to be maximised, the mean of each class to the global mean
    # wc: to be minimised: the mean of each instance to each class mean
    return bc, wc


def rademacher_complexity_multiclass(lsvm, X, num_classes):
    n = X.shape[0]
    prob = 1 / num_classes
    sigma = np.random.choice([-1, 1], size=(n, num_classes), p=[1-prob, prob])
    rademacher_vals = []
    for i in range(num_classes):
        f_predict = lsvm.predict(X)
        f_X = np.where(f_predict==i, 1, -1)
        rademacher_val = np.mean(sigma[:, i] * f_X)
        rademacher_vals.append(rademacher_val)
    rademacher_value = np.max(rademacher_vals)

    return rademacher_value


def evalTrain(individual):
    try:
        func = toolbox.compile(expr=individual)
        kf = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=5)
        acc = 0
        nf = 0
        rademacher_vals = []
        for train_index, test_index in kf.split(y_train, y_train):
            x_train_eval, x_test_eval = x_train[train_index], x_train[test_index]
            y_train_eval, y_test_eval = y_train[train_index], y_train[test_index]
            x_train_combine = numpy.concatenate((x_train_eval, x_test_eval), axis=0)
            train_tf = []
            for i in range(0, len(y_train)):
                train_tf.append(numpy.asarray(func(x_train_combine[i, :, :])))
            train_tf = numpy.asarray(train_tf, dtype=float)
            # train_tf = numpy.asarray(func(x_train_combine, y_train_eval))
            train_tf_train = train_tf[:x_train_eval.shape[0]]
            train_tf_test = train_tf[x_train_eval.shape[0]:]
            min_max_scaler = preprocessing.MinMaxScaler()
            train_norm_train = min_max_scaler.fit_transform(train_tf_train)
            train_norm_test = min_max_scaler.transform(train_tf_test)
            lsvm = LinearSVC(max_iter=100, random_state=5)
            lsvm.fit(train_norm_train, y_train_eval)
            acc += 100 * lsvm.score(train_norm_test, y_test_eval)
            nf += train_tf.shape[1]
            f_predict = lsvm.predict(train_norm_train)
            rademacher = rademacher_complexity_multiclass(lsvm, train_norm_train, n_class)
            rademacher_vals.append(rademacher)
        rademacher_val = np.mean(rademacher_vals)
        nf_mean = round(nf / n_cv)
        accuracy = round(acc / n_cv, 2)
        nf_score = 100 * numpy.exp(-nf_mean/100)
        norm_rademacher = 100 * numpy.exp(-rademacher_val)
        f1 = (accuracy + norm_rademacher)/2
        f2 = nf_score
    except:
        f1 = 0
        f2 = 0
    return f1, f2


# genetic operator
toolbox.register("evaluate", evalTrain)
# toolbox.register("select", tools.selTournament, tournsize=5)
# toolbox.register("selectElitism", tools.selBest)
# select by nsga2
toolbox.register("select", tools.selNSGA2)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp_restrict.genFull, min_=1, max_=4)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)


# toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=maxDepth))
# toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=maxDepth))

def GPMain(randomSeeds):
    random.seed(randomSeeds)
    pop = toolbox.population(population)
    log = tools.Logbook()
    stats_fit1 = tools.Statistics(key=lambda ind: ind.fitness.values[0])
    # stats_fit1 = tools.Statistics(ind.fitness.values[0])
    # stats_size_tree = tools.Statistics(key=len)
    stats_nf_score = tools.Statistics(key=lambda ind: ind.fitness.values[1])
    # stats_nf_score = tools.Statistics(ind.fitness.values[1])
    # stats_size_feature = tools.Statistics(key=lambda ind: feature_length(ind, x_train[1,:,:], y_train[1], toolbox))
    mstats = tools.MultiStatistics(acc=stats_fit1, nf_score=stats_nf_score)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)
    log.header = ["gen", "evals"] + mstats.fields

    # pop, log, best_front, hof, pop_store = evalGP.moead(pop, toolbox, cxProb, mutProb, elitismProb, generation, pset,
    #                                               stats=mstats, verbose=True)
    pop, log, hof, pop_store, n_replace_all, entropy_all = evalGP.entropy_moead2(pop, toolbox, cxProb, mutProb, elitismProb, generation, pset,
                                                        stats=mstats, verbose=True)

    # return pop, log, hof, best_front, pop_store
    return pop, log, hof, hof, pop_store, n_replace_all, entropy_all


if __name__ == "__main__":
    beginTime = time.process_time()
    pop, log, hof, best_front, pop_store, n_replace_all, entropy_all = GPMain(randomSeeds)

    endTime = time.process_time()
    trainTime = endTime - beginTime

    numpy.save(save_path + file_name + '_n_replace_all.npy', n_replace_all)
    numpy.save(save_path + file_name + '_entropy_all.npy', entropy_all)
    print('n_replace:', n_replace_all)
    print('entropy:', entropy_all)

    import matplotlib.pyplot as plt
    for p in pop_store:
        fitness = [[ind.fitness.values[0], ind.fitness.values[1]] for ind in p]
        fitness = numpy.asarray(fitness)
        plt.scatter(fitness[:, 0], fitness[:, 1])
        plt.show()
        plt.pause(1)


    logging.info('train time ' + str(trainTime))

    saveFile.saveLog(save_path + file_name + 'final_pop.pickle', pop)
    saveFile.saveLog(save_path + file_name + 'all_pop.pickle', hof)
    saveFile.saveLog(save_path + file_name + 'best_front.pickle', best_front)
    saveFile.saveLog(save_path + file_name + 'pop_store.pickle', pop_store)
    HV_train = hypervolume(best_front, [0, 0])
    train_results = []
    test_acc = numpy.zeros((len(best_front), 1))
    test_nf = numpy.zeros((len(best_front), 1))
    test_nf_score = numpy.zeros((len(best_front), 1))
    for i, ind in enumerate(best_front):
        train_tf, test_tf, trainLabel, testL, testResults = evalTest(toolbox, ind, x_train, y_train, x_test, y_test)
        testTime = time.process_time() - endTime
        print(ind)
        print(testResults)
        logging.info(f'individual{i}: ')
        logging.info(ind)
        logging.info('test acc-' + f'individual{i}: ' + str(testResults))
        print(train_tf.shape, test_tf.shape)
        num_features = train_tf.shape[1]
        nf_score = 100 * numpy.exp(-num_features/100)
        print(nf_score)
        logging.info('Num Features-' + f'individual{i}: ' + str(num_features))
        logging.info('Num Features score-' + f'individual{i}: ' + str(nf_score))
        logging.info('test time-' + f'individual{i}: ' + str(testTime))
        train_results.append(ind.fitness.values)
        test_acc[i] = testResults
        test_nf_score[i] = nf_score
        test_nf[i] = num_features
        ind.fitness.values = numpy.concatenate((test_acc[i], test_nf_score[i]))
        endTime = time.process_time()
    HV_test = hypervolume(best_front, [0, 0])
    logging.info('HV_train:' + str(HV_train))
    logging.info('HV_test:' + str(HV_test))
    test_results = numpy.concatenate((test_acc, test_nf_score, test_nf), axis=1)
    numpy.save(save_path + file_name + '_train_results.npy', train_results)
    numpy.save(save_path + file_name + '_test_results.npy', test_results)
    numpy.save(save_path + file_name + '_HV_train.npy', HV_train)
    numpy.save(save_path + file_name + '_HV_test.npy', HV_test)
    saveFile.saveAllResults(save_path + file_name, dataSetName, hof, log,
                            hof, num_features, trainTime, testTime, testResults)
    print('End')
    logging.info('End')
