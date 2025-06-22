# python packages
import random
import time

# from matplotlib import pyplot as plt

import evalGP_lfgp16 as evalGP
import sys
# only for strongly typed GP
import gp_restrict_con7 as gp_restrict
import numpy
import gp_tree
# deap package
from deap import base, creator, tools, gp
# fitness function
from FEVal_norm_fast import evalTest_fromvector as evalTest
##plot tree and save
import saveFile
##image Data
# defined by author
import fgp_functions_matrix as fs
# from scoop import futures
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing  ##
import logging
from strongGPDataType import region, kernelSize, filterData, coordsX1, coordsX2, windowSize2, windowSize3, poolingType, \
    para_m, selePro, poolingData3, filterData2, histdata, Int1, Int2, Int3
from deal_with_data2 import deal_data, deal_data2

'same as linar_fgp_16_1, but using the classification accuracy as fitness measure'

# randomSeeds = int(sys.argv[1])
# # randomSeeds = sys.argv[1]
# dataSetName = str(sys.argv[2])


randomSeeds = 2
# dataSetName = 'cifar10'
dataSetName = 'jaffe'
# num_instance = 3
# data_path = '/vol/grid-solar/sgeusers/yingbi/dataset_npy/multi_small'
# data_path = '/vol/grid-solar/sgeusers/yingbi/dataset_npy/popular/'


data_path = '../popular/'

# data_path = '/nesi/nobackup/nesi00416/datasets/multi_small/'
# data_path = '/nfs/scratch/biyi/datasets/'
# data_path = '/Users/vuw/my_code/dataset/popular/'

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
nui = numpy.unique(y_train)
num_instance = len(y_train[y_train == nui[0]])
n_cv = min(5, num_instance)
nf_size = x_train.shape[1]*x_train.shape[2]
x_train, x_test, y_train, y_test = deal_data2(x_train, y_train, x_test, y_test, dataSetName, num_instance, n_class)
indicators = ['tchebycheff', 'projection']
save_path = '../results/guiding_mofgp_light/'
file_name = str(randomSeeds) + '_' + indicators[1] + '_' + str(dataSetName) + '_'
#################################################################################################################################
# parameters:
population = 5
generation = 5
cxProb = 0.8
mutProb = 0.19
elitismProb = 0.01
totalRuns = 1
initialMinDepth = 2
initialMaxDepth = 8
maxDepth = 10
# logging results
logging.basicConfig(level=logging.INFO, filename=save_path + file_name + 'guiding_mofgp_main.log')
logging.info('#############Start##############')
logging.info('Algorithm name: Guiding_MOFGP.py ')
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
# GP
pset = gp_tree.PrimitiveSetTyped('MAIN', [filterData, poolingType], filterData2, prefix='Image')
# feature selection (Are you sure it is not a feature concatenation node?)
pset.addPrimitive(fs.FeaCon2, [filterData2, filterData2], filterData2, name='Root2')
# then feature selection
# similarity-based
pset.addPrimitive(fs.relif_fs, [histdata, poolingType, selePro], filterData2, name='RelifF')
pset.addPrimitive(fs.fisher_score_fs, [histdata, poolingType, selePro], filterData2, name='Fisher')
# information theoretical based
# pset.addPrimitive(fs.cmim_fs, [region, poolingType, selePro], region,name='CMIM')
# pset.addPrimitive(fs.disr_fs, [region, poolingType, selePro], region,name='DISR')
# pset.addPrimitive(fs.mrmr_fs, [histdata, poolingType, selePro], filterData2,name='MRMR')
# pset.addPrimitive(fs.mifs_fs, [histdata, poolingType, selePro], filterData2,name='MIFS')
# sparse learning based
# pset.addPrimitive(fs.rfs_fs, [region, poolingType, selePro], poolingData3,name='RFS')
# pset.addPrimitive(fs.mcfs_fs, [region, poolingType, selePro], region,name='MCFS')

# statistical based
# pset.addPrimitive(fs.chi_sq_fs, [histdata, poolingType, selePro], filterData2,name='Chi_s')
pset.addPrimitive(fs.f_score_fs, [histdata, poolingType, selePro], filterData2, name='F_score')
# pset.addPrimitive(fs.gini_index_fs, [histdata, poolingType, selePro], filterData2,name='Gini')
pset.addPrimitive(fs.fs_variance, [histdata, selePro], filterData2, name='Variance')
# pset.addPrimitive(fs.maxP,[region, kernelSize, kernelSize],poolingData3,name='MaxP2')

# feature transformation/rescaling
pset.addPrimitive(fs.log_tf, [poolingData3], histdata, name='LOG_TF')
# pset.addPrimitive(fs.reciprocal_tf,[poolingData3], histdata,name='RE_TF')
pset.addPrimitive(fs.sqrt_tf, [poolingData3], histdata, name='Sqrt_TF')
# pset.addPrimitive(fs.exponential_tf,[poolingData3, para_m], histdata,name='Exp_TF')
# pset.addPrimitive(fs.box_cox_tf,[region],region,name='Box_TF')
# pset.addPrimitive(fs.relu, [poolingData3], poolingData3, name='Relu_TF')
# pset.addPrimitive(fs.yeo_johnson_tf,[region],region,name='Yeo_John_TF')
pset.addPrimitive(fs.MinMax, [poolingData3], histdata, name='MinMax')
pset.addPrimitive(fs.Normal, [poolingData3], histdata, name='Normal')
# feature concatenation after extraction
pset.addPrimitive(fs.FeaCon2, [region, region], poolingData3, name='Roots2')
pset.addPrimitive(fs.FeaCon3, [region, region, region], poolingData3, name='Roots3')
pset.addPrimitive(fs.FeaCon4, [region, region, region, region], poolingData3, name='Roots4')
##feature extraction
pset.addPrimitive(fs.global_hog_small, [filterData], region, name='Global_HOG')
pset.addPrimitive(fs.all_lbp, [filterData], region, name='Global_uLBP')
pset.addPrimitive(fs.all_sift, [filterData], region, name='Global_SIFT')
pset.addPrimitive(fs.all_dif, [filterData], region, name='Global_DIF')
pset.addPrimitive(fs.all_histogram, [filterData], region, name='Global_Hist')

# pooling
# pset.addPrimitive(fs.maxP,[filterData, kernelSize, kernelSize],filterData,name='MaxP')
# aggregation
# pset.addPrimitive(fs.mixconadd, [filterData, float, filterData, float], filterData, name='Mix_ConAdd')
# pset.addPrimitive(fs.mixconsub, [filterData, float, filterData, float], filterData, name='Mix_ConSub')
##pset.addPrimitive(numpy.abs, [filterData], filterData, name='Abs')
pset.addPrimitive(fs.sqrt, [filterData], filterData, name='Sqrt')
pset.addPrimitive(fs.relu, [filterData], filterData, name='Relu')
# edge features
pset.addPrimitive(fs.sobelxy, [filterData], filterData, name='Sobel_XY')
# pset.addPrimitive(fs.sobelx, [filterData], filterData, name='Sobel_X')
# pset.addPrimitive(fs.sobely, [filterData], filterData, name='Sobel_Y')
# Gabor
pset.addPrimitive(fs.gab, [filterData, windowSize2, windowSize3], filterData, name='Gabor2')
pset.addPrimitive(fs.gaussian_Laplace1, [filterData], filterData, name='LoG1')
pset.addPrimitive(fs.gaussian_Laplace2, [filterData], filterData, name='LoG2')
# pset.addPrimitive(fs.laplace,[filterData],filterData, name='Lap')
# pset.addPrimitive(fs.lbp,[filterData],filterData, name='LBP')
# pset.addPrimitive(fs.hog_feature,[filterData],filterData, name='HoG')
# Gaussian features
pset.addPrimitive(fs.gau, [filterData, coordsX2], filterData, name='Gau')
# pset.addPrimitive(fs.gauD, [filterData,coordsX2, coordsX1,coordsX1], filterData, name='GauD')
# general filters
pset.addPrimitive(fs.medianf, [filterData], filterData, name='Med')
pset.addPrimitive(fs.maxf, [filterData], filterData, name='Max')
pset.addPrimitive(fs.minf, [filterData], filterData, name='Min')
pset.addPrimitive(fs.meanf, [filterData], filterData, name='Mean')

# pset.addPrimitive(fs.regionS, [filterData, Int1, Int2, Int3], filterData, name='Region_S')
# pset.addPrimitive(fs.regionR, [filterData, Int1, Int2, Int3, filterData], filterData, name='Region_R')
# pset.addPrimitive(fs.regionS, [filterData, Int1, Int2, Int3], filterData, name='Region_S2')
# pset.addPrimitive(fs.regionR, [filterData, Int1, Int2, Int3, filterData], filterData, name='Region_R2')
# pset.addPrimitive(fs.regionS, [filterData, Int1, Int2, Int3], filterData, name='Region_S3')
# pset.addPrimitive(fs.regionR, [filterData, Int1, Int2, Int3, filterData], filterData, name='Region_R3')
# Terminals
pset.renameArguments(ARG0='grey')
# pset.addEphemeralConstant('randomD', lambda: round(random.random(), 3), float)
# pset.addEphemeralConstant('kernelSize', lambda: random.randrange(2, 5, 2), kernelSize)
pset.addEphemeralConstant('Theta', lambda: random.randint(0, 8), windowSize2)
pset.addEphemeralConstant('Frequency', lambda: random.randint(0, 5), windowSize3)
pset.addEphemeralConstant('Singma', lambda: random.randint(1, 4), coordsX2)
pset.addEphemeralConstant('Order', lambda: random.randint(0, 3), coordsX1)
# pset.addEphemeralConstant('Para_m1', lambda: random.randint(2, 4), para_m)
pset.addEphemeralConstant('select_pro', lambda: random.randint(1, 9), selePro)
# pset.addEphemeralConstant('X', lambda: random.randint(0, bound1 - 10), Int1)
# pset.addEphemeralConstant('Y', lambda: random.randint(0, bound2 - 10), Int2)
# pset.addEphemeralConstant('Size', lambda: random.randint(10, 31), Int3)

##
creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0,))
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


def evalTrain(individual):
    # print(individual)
    # try:
    func = toolbox.compile(expr=individual)
    kf = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=5)
    acc = 0
    for train_index, test_index in kf.split(x_train, y_train):
        x_train_eval, x_test_eval = x_train[train_index], x_train[test_index]
        y_train_eval, y_test_eval = y_train[train_index], y_train[test_index]
        x_train_combine = numpy.concatenate((x_train_eval, x_test_eval), axis=0)
        train_tf = numpy.asarray(func(x_train_combine, y_train_eval))
        train_tf_train = train_tf[:x_train_eval.shape[0]]
        train_tf_test = train_tf[x_train_eval.shape[0]:]
        min_max_scaler = preprocessing.MinMaxScaler()
        train_norm_train = min_max_scaler.fit_transform(train_tf_train)
        train_norm_test = min_max_scaler.transform(train_tf_test)
        lsvm = LinearSVC(max_iter=100, random_state=5)
        lsvm.fit(train_norm_train, y_train_eval)
        acc += 100 * lsvm.score(train_norm_test, y_test_eval)
    train_tf = numpy.asarray(func(x_train, y_train))
    min_max_scaler = preprocessing.MinMaxScaler()
    train_norm = min_max_scaler.fit_transform(train_tf)
    lsvm = LinearSVC(max_iter=100, random_state=5)
    lsvm.fit(train_norm, y_train)
    coef = lsvm.coef_
    b = lsvm.intercept_
    x_transform = numpy.matmul(coef, train_norm.T)
    x_transform = (x_transform.T + b)
    bc, wc = between_with_class(x_transform, y_train)
    # bc: to be maximise, the mean of each class to the global mean
    # wc: to be minimise: the mean of each instance to each class mean
    # print(bc, wc)
    nf = train_tf.shape[1]
    accuracy = round(acc / n_cv, 2)
    nf_mean = train_norm.shape[1]
    nf_score = round(100 / (1 + numpy.exp(-(-nf_mean + nf_size / 4))), 2)  # range: #
    distance = round(100 / (1 + numpy.exp(-(bc - wc))), 2)  # range: #
    # print(accuracy, nf_mean, nf_score, distance)
    # allt = 0.1*accuracy+0.1*nf_score+0.8*distance
    # print(all)
    # except:
    #     accuracy = 0
    #     nf_score = 0
    return accuracy, nf


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
    stats_size_tree = tools.Statistics(key=len)
    # stats_size_feature = tools.Statistics(key= lambda ind: feature_length(ind, x_train[1,:,:], y_train[1], toolbox))
    mstats = tools.MultiStatistics(acc=stats_fit1, size_tree=stats_size_tree)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)
    log.header = ["gen", "evals"] + mstats.fields

    pop, log, best_front, hof, pop_store = evalGP.Guiding_MOFGP(pop, toolbox, cxProb, mutProb, elitismProb, generation, pset,
                                                     indicator=indicators[0], stats=mstats, verbose=True)

    return pop, log, hof, best_front, pop_store


if __name__ == "__main__":
    beginTime = time.process_time()
    pop, log, hof, best_front, pop_store = GPMain(randomSeeds)

    endTime = time.process_time()
    trainTime = endTime - beginTime
    saveFile.saveLog(save_path + file_name + 'final_pop.pickle', pop)
    saveFile.saveLog(save_path + file_name + 'all_pop.pickle', hof)
    saveFile.saveLog(save_path + file_name + 'best_front.pickle', best_front)
    saveFile.saveLog(save_path + file_name + 'generational_pop.pickle', pop_store)
    logging.info('train time ' + str(trainTime))
    for i, ind in enumerate(best_front):
        train_tf, test_tf, trainLabel, testL, testResults = evalTest(toolbox, ind, x_train, y_train, x_test, y_test)
        testTime = time.process_time() - endTime
        print(ind)
        print(testResults)
        logging.info(f'individual_{i}: ')
        logging.info(ind)
        logging.info('test results-' + f'individual_{i}: ' + str(testResults))
        print(train_tf.shape, test_tf.shape)
        num_features = train_tf.shape[1]
        logging.info('Num Features-' + f'individual_{i}: ' + str(num_features))
        logging.info('test time-' + f'individual_{i}: ' + str(testTime))
        endTime = time.process_time()
    saveFile.saveAllResults(save_path + file_name, dataSetName, hof, log,
                            hof, num_features, trainTime, testTime, testResults)
    print('End')
    logging.info('End')
