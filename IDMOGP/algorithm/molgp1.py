#python packages
import random
import time
import evalGP_fgp as evalGP
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
from strongGPDataType import region,kernelSize,histdata,filterData,coordsX1,coordsX2,windowSize2,windowSize3,poolingType, para_m, selePro
# defined by author
import fgp_functions_matrix as fs
# from scoop import futures
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing##
import logging

# randomSeeds=int(sys.argv[1])
# dataSetName=str(sys.argv[2])
randomSeeds = 2
dataSetName = 'cifar10'
# num_instance = 3
data_path = '../popular/'
# randomSeeds = 3
# dataSetName = 'f1'
# data_path = '/vol/grid-solar/sgeusers/yingbi/datasets_npy/popular/'
# data_path = '/vol/grid-solar/sgeusers/yingbi/datasets_npy/multi_small/'
# data_path = '/vol/grid-solar/sgeusers/yingbi/datasets_npy/ssmultiTask/crossD/'

def load_data(dataset_name, path=None):
    if path is not None:
        file = path+dataset_name+'/'+dataset_name
    else: file = dataset_name
    x_train = numpy.load(file+'_train_data.npy')/255.0
    y_train = numpy.load(file+'_train_label.npy')
    x_test = numpy.load(file+'_test_data.npy')/255.0
    y_test = numpy.load(file+'_test_label.npy')
    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = load_data(dataSetName)
# x_train, y_train, x_test, y_test = load_data(dataSetName, path=data_path)
print(x_train.shape,y_train.shape, x_test.shape,y_test.shape)
nui = numpy.unique(y_train)
n_instances = len(y_train[y_train ==nui[0]])
n_cv = min(5, n_instances)
#parameters:
population=100
generation=50
cxProb=0.8
mutProb=0.19
elitismProb=0.01
totalRuns = 1
initialMinDepth=2
initialMaxDepth=6
maxDepth=8
#logging results
logging.basicConfig(level=logging.INFO, filename=str(randomSeeds)+'_'+dataSetName+'_FGP.log')
logging.info('#############Strat##############')
logging.info('Algorithm name: FGP.py ')
logging.info('Training set shape: '+str(x_train.shape))
logging.info('Test set shape: '+str(x_test.shape))
logging.info('population: ' + str(population))
logging.info('generation: ' + str(generation))
logging.info('cxProb: ' + str(cxProb))
logging.info('mutProb: ' + str(mutProb))
logging.info('elitismProb: ' + str(elitismProb))
logging.info('initialMinDepth: ' + str(initialMinDepth))
logging.info('initialMaxDepth: ' + str(initialMaxDepth))
logging.info('maxDepth: ' + str(maxDepth))
##GP
##GP
pset = gp_tree.PrimitiveSetTyped('MAIN',[filterData, poolingType], region, prefix='Image')
pset.addPrimitive(fs.FeaCon2, [region, region],region,name='Roots2')
pset.addPrimitive(fs.FeaCon3, [region, region, region],region,name='Roots3')
pset.addPrimitive(fs.FeaCon4, [region, region, region, region],region,name='Roots4')
#feature selection
# pset.addPrimitive(fs.chi_sq_fs, [region, poolingType, selePro], region,name='Chi_s')
#pset.addPrimitive(fs.f_score_fs, [region, poolingType, selePro], region,name='F_score')
#pset.addPrimitive(fs.gini_index_fs, [region, poolingType, selePro], region,name='Gini')
#pset.addPrimitive(fs.fs_variance, [region, selePro], region,name='Variance')
#pset.addPrimitive(fs.relif_fs, [region, poolingType, selePro], region,name='RelifF')
# pset.addPrimitive(fs.cmim_fs, [region, poolingType, selePro], region,name='CMIM')
# pset.addPrimitive(fs.disr_fs, [region, poolingType, selePro], region,name='DISR')
#feature transformation
#pset.addPrimitive(fs.log_tf,[region],region,name='LOG_TF')
#pset.addPrimitive(fs.reciprocal_tf,[region],region,name='RE_TF')
#pset.addPrimitive(fs.sqrt_tf,[region],region,name='Sqrt_TF')
#pset.addPrimitive(fs.exponential_tf,[region, para_m],region,name='Exp_TF')
# pset.addPrimitive(fs.box_cox_tf,[region],region,name='Box_TF')
#pset.addPrimitive(fs.yeo_johnson_tf,[region],region,name='Yeo_John_TF')
# pset.addPrimitive(fs.MinMax,[region],region,name='MinMax')
# pset.addPrimitive(fs.Normal,[region],region,name='Normal')
##feature extraction
pset.addPrimitive(fs.global_hog_small,[filterData],region,name='Global_HOG')
pset.addPrimitive(fs.all_lbp,[filterData],region,name='Global_uLBP')
pset.addPrimitive(fs.all_sift,[filterData],region,name='Global_SIFT')
#pooling#aggregation
#pooling
pset.addPrimitive(fs.maxP,[filterData, kernelSize, kernelSize],filterData,name='MaxP')
#aggregation
pset.addPrimitive(fs.mixconadd, [filterData, float, filterData, float], filterData, name='Mix_ConAdd')
pset.addPrimitive(fs.mixconsub, [filterData, float, filterData, float], filterData, name='Mix_ConSub')
##pset.addPrimitive(numpy.abs, [filterData], filterData, name='Abs')
pset.addPrimitive(fs.sqrt, [filterData], filterData, name='Sqrt')
pset.addPrimitive(fs.relu, [filterData], filterData, name='Relu')
# edge features
pset.addPrimitive(fs.sobelxy, [filterData], filterData, name='Sobel_XY')
pset.addPrimitive(fs.sobelx, [filterData], filterData, name='Sobel_X')
pset.addPrimitive(fs.sobely, [filterData], filterData, name='Sobel_Y')
#Gabor
pset.addPrimitive(fs.gab, [filterData,windowSize2, windowSize3], filterData, name='Gabor2')
pset.addPrimitive(fs.gaussian_Laplace1,[filterData],filterData, name='LoG1')
pset.addPrimitive(fs.gaussian_Laplace2,[filterData],filterData, name='LoG2')
pset.addPrimitive(fs.laplace,[filterData],filterData, name='Lap')
pset.addPrimitive(fs.lbp,[filterData],filterData, name='LBP')
pset.addPrimitive(fs.hog_feature,[filterData],filterData, name='HoG')
# Gaussian features
pset.addPrimitive(fs.gau, [filterData,coordsX2], filterData, name='Gau')
pset.addPrimitive(fs.gauD, [filterData,coordsX2, coordsX1,coordsX1], filterData, name='GauD')
#general filters
pset.addPrimitive(fs.medianf, [filterData],filterData,name='Med')
pset.addPrimitive(fs.maxf, [filterData],filterData,name='Max')
pset.addPrimitive(fs.minf, [filterData],filterData,name='Min')
pset.addPrimitive(fs.meanf, [filterData],filterData,name='Mean')

#Terminals
pset.renameArguments(ARG0='grey')
pset.addEphemeralConstant('randomD',lambda:round(random.random(),3),float)
pset.addEphemeralConstant('kernelSize',lambda:random.randrange(2,5,2),kernelSize)
pset.addEphemeralConstant('Theta',lambda:random.randint(0,8),windowSize2)
pset.addEphemeralConstant('Frequency',lambda:random.randint(0,5),windowSize3)
pset.addEphemeralConstant('Singma',lambda:random.randint(1,4),coordsX2)
pset.addEphemeralConstant('Order',lambda:random.randint(0,3),coordsX1)
pset.addEphemeralConstant('Para_m1',lambda:random.randint(2,4),para_m)
pset.addEphemeralConstant('select_pro',lambda:random.randint(1,9), selePro)

##
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr", gp_restrict.genHalfAndHalfMD, pset=pset, min_=initialMinDepth, max_=initialMaxDepth)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)
# toolbox.register("mapp", futures.map)
toolbox.register("mapp", map)

def evalTrainb(individual):
    # print(individual)
    func = toolbox.compile(expr=individual)
    train_tf= numpy.asarray(func(x_train, y_train))
    # print(train_tf.shape)
    train_tf = numpy.asarray(train_tf, dtype=float)
    min_max_scaler = preprocessing.MinMaxScaler()
    train_norm = min_max_scaler.fit_transform(train_tf)
    lsvm = LinearSVC()
    accuracy = round(100 * cross_val_score(lsvm, train_norm, y_train, cv=n_cv).mean(), 2)
    return accuracy,

def evalTrain(individual):
    try:
        func = toolbox.compile(expr=individual)
        train_tf= numpy.asarray(func(x_train, y_train))
        train_tf = numpy.asarray(train_tf, dtype=float)
        min_max_scaler = preprocessing.MinMaxScaler()
        train_norm = min_max_scaler.fit_transform(train_tf)
        lsvm = LinearSVC(max_iter=100)
        accuracy = round(100 * cross_val_score(lsvm, train_norm, y_train, cv=n_cv).mean(), 2)
    except:
        accuracy = 0
    return accuracy,

#genetic operator
toolbox.register("evaluate", evalTrain)
toolbox.register("select", tools.selTournament,tournsize=7)
toolbox.register("selectElitism", tools.selBest)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp_restrict.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
#toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=maxDepth))
#toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=maxDepth))

def GPMain(randomSeeds):

    random.seed(randomSeeds)
   
    pop = toolbox.population(population)
    hof = tools.HallOfFame(1)
    log = tools.Logbook()
    stats_fit = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats_size_tree = tools.Statistics(key=len)
    # stats_size_feature = tools.Statistics(key= lambda ind: feature_length(ind, x_train[1,:,:], y_train[1], toolbox))
    mstats = tools.MultiStatistics(fitness=stats_fit,size_tree=stats_size_tree)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)
    log.header = ["gen", "evals"] + mstats.fields

    pop, log, best_ind_over = evalGP.eaSimple(pop, toolbox, cxProb, mutProb, elitismProb, generation,
                                              stats=mstats, halloffame=hof, verbose=True)

    return pop, log, hof, best_ind_over


if __name__ == "__main__":
    beginTime = time.process_time()
    pop, log, hof, best_ind_over = GPMain(randomSeeds)
    endTime = time.process_time()
    trainTime = endTime - beginTime

    train_tf, test_tf, trainLabel, testL, testResults = evalTest(toolbox, hof[0], x_train, y_train, x_test, y_test)

    saveFile.saveLog(str(randomSeeds) + dataSetName + 'all_pop.pickle', pop)
    saveFile.saveLog(str(randomSeeds) + dataSetName + 'best_pop.pickle', hof)
    saveFile.saveLog(str(randomSeeds) + dataSetName + 'best_pop_gen.pickle', best_ind_over)
    
    testTime = time.process_time() - endTime
    print(testResults)
    logging.info('test results ' + str(testResults))
    print(train_tf.shape, test_tf.shape)
    num_features = train_tf.shape[1]
    logging.info('Num Features ' + str(num_features))
##    bestInd=saveFile.bestInd(toolbox,pop,5)
    saveFile.saveAllResults(randomSeeds, dataSetName, hof, log,
                            hof, num_features, trainTime, testTime, testResults)
    print(hof[0])
    print('End')
    logging.info(hof[0])
    logging.info('train time ' + str(trainTime))
    logging.info('test time ' + str(testTime))
    logging.info('End')
