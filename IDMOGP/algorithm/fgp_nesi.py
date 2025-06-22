#python packages
import random
import time
import operator
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
from FEVal_norm_fast import feature_length
##plot tree and save
import  saveFile
##image Data
from strongGPDataType import region,kernelSize,histdata,filterData,coordsX1,coordsX2,windowSize2,windowSize3,poolingType
# defined by author
import functionSet_renew3 as fs
import feature_function as fe_fs
# import scoop
# from scoop import futures
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing##
import logging
##from plot_confusion_matrix import plot_conf_matrix
##import matplotlib.pyplot as plt

randomSeeds=int(sys.argv[1])
dataSetName=str(sys.argv[2])
# randomSeeds = 2
# dataSetName = 'f1'
# data_path = '/vol/grid-solar/sgeusers/yingbi/datasets_npy/ssmultiTask/crossD'
data_path = '/nesi/nobackup/nesi00416/datasets/ssmultiTask/crossD/'

def load_data(dataset_name, path=None):
    if path is not None:
        file = path+'/'+dataset_name+'/'+dataset_name
    else: file = dataset_name
    x_train = numpy.load(file+'_train_data.npy')/255.0
    y_train = numpy.load(file+'_train_label.npy')
    x_test = numpy.load(file+'_test_data.npy')/255.0
    y_test = numpy.load(file+'_test_label.npy')
    return x_train, y_train, x_test, y_test

#x_train, y_train, x_test, y_test = load_data(dataSetName)
x_train, y_train, x_test, y_test = load_data(dataSetName, path=data_path)
print(x_train.shape,y_train.shape, x_test.shape,y_test.shape)
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
pset = gp_tree.PrimitiveSetTyped('MAIN',[filterData], histdata, prefix='Image')
pset.addPrimitive(fe_fs.root_con,[histdata,histdata],histdata,name='Root1')
pset.addPrimitive(fs.root_conVector2,[poolingType, poolingType],histdata,name='Root2')
pset.addPrimitive(fs.root_conVector3,[poolingType, poolingType, poolingType],histdata,name='Root3')
##pset.addPrimitive(fs.root_conVector4,[poolingType, poolingType, poolingType, poolingType],histdata,name='Root4')

pset.addPrimitive(fe_fs.root_con, [region, region],histdata,name='Roots2')
pset.addPrimitive(fe_fs.root_con, [region, region, region],histdata,name='Roots3')
pset.addPrimitive(fe_fs.root_con, [region, region, region, region],histdata,name='Roots4')

##with other features
pset.addPrimitive(fe_fs.global_hog_small,[poolingType],region,name='Global_HOG')
pset.addPrimitive(fe_fs.all_lbp,[poolingType],region,name='Global_uLBP')
pset.addPrimitive(fe_fs.all_sift,[poolingType],region,name='Global_SIFT')
pset.addPrimitive(fe_fs.global_hog_small,[filterData],region,name='FGlobal_HOG')
pset.addPrimitive(fe_fs.all_lbp,[filterData],region,name='FGlobal_uLBP')
pset.addPrimitive(fe_fs.all_sift,[filterData],region,name='FGlobal_SIFT')
#pooling
pset.addPrimitive(fs.maxP,[poolingType, kernelSize, kernelSize],poolingType,name='MaxPf')
#aggregation
pset.addPrimitive(fs.mixconadd, [poolingType, float, poolingType, float], poolingType, name='Mix_ConAddf')
pset.addPrimitive(fs.mixconsub, [poolingType, float, poolingType, float], poolingType, name='Mix_ConSubf')
##pset.addPrimitive(numpy.abs, [filterData], filterData, name='Abs')
pset.addPrimitive(fs.sqrt, [poolingType], poolingType, name='Sqrtf')
pset.addPrimitive(fs.relu, [poolingType], poolingType, name='Reluf')
# edge features
pset.addPrimitive(fs.sobelxy, [poolingType], poolingType, name='Sobel_XYf')
pset.addPrimitive(fs.sobelx, [poolingType], poolingType, name='Sobel_Xf')
pset.addPrimitive(fs.sobely, [poolingType], poolingType, name='Sobel_Yf')
#Gabor
pset.addPrimitive(fs.gab, [poolingType,windowSize2, windowSize3], poolingType, name='Gabor2f')
pset.addPrimitive(fs.gaussian_Laplace1,[poolingType],poolingType, name='LoG1f')
pset.addPrimitive(fs.gaussian_Laplace2,[poolingType],poolingType, name='LoG2f')
pset.addPrimitive(fs.laplace,[poolingType],poolingType, name='Lapf')
pset.addPrimitive(fs.lbp,[poolingType],poolingType, name='LBPf')
pset.addPrimitive(fs.hog_feature,[poolingType],poolingType, name='HoGf')
pset.addPrimitive(fs.gauD, [poolingType,coordsX2, coordsX1,coordsX1], poolingType, name='Gau_Df')
pset.addPrimitive(fs.gau, [poolingType,coordsX2], poolingType, name='Gauf')
pset.addPrimitive(fs.medianf, [poolingType],poolingType,name='Medf')
pset.addPrimitive(fs.maxf, [poolingType],poolingType,name='Maxf')
pset.addPrimitive(fs.minf, [poolingType],poolingType,name='Minf')
pset.addPrimitive(fs.meanf, [poolingType],poolingType,name='Meanf')
#pooling
pset.addPrimitive(fs.maxP,[filterData, kernelSize, kernelSize],poolingType,name='MaxP1')
#aggregation
pset.addPrimitive(fs.mixconadd, [filterData, float, filterData, float], filterData, name='Mix_ConAdd')
pset.addPrimitive(fs.mixconsub, [filterData, float, filterData, float], filterData, name='Mix_ConSub')
##pset.addPrimitive(numpy.abs, [filterData], filterData, name='Abs')
pset.addPrimitive(fs.sqrt, [filterData], filterData, name='Sqrt')
pset.addPrimitive(fs.relu, [filterData], filterData, name='Relu')
# edge features
##pset.addPrimitive(fs.prewittx,[filterData], filterData, name='Prewitt_X')
##pset.addPrimitive(fs.prewitty,[filterData], filterData, name='Prewitt_Y')
##pset.addPrimitive(fs.prewittxy, [filterData], filterData, name='Prewitt_XY')
pset.addPrimitive(fs.sobelxy, [filterData], filterData, name='Sobel_XY')
pset.addPrimitive(fs.sobelx, [filterData], filterData, name='Sobel_X')
pset.addPrimitive(fs.sobely, [filterData], filterData, name='Sobel_Y')
##pset.addPrimitive(fs.scharrxy, [filterData], filterData, name='Scharr_XY')
#Gabor
pset.addPrimitive(fs.gab, [filterData,windowSize2, windowSize3], filterData, name='Gabor2')
pset.addPrimitive(fs.gaussian_Laplace1,[filterData],filterData, name='LoG1')
pset.addPrimitive(fs.gaussian_Laplace2,[filterData],filterData, name='LoG2')
pset.addPrimitive(fs.laplace,[filterData],filterData, name='Lap')
pset.addPrimitive(fs.lbp,[filterData],filterData, name='LBP')
pset.addPrimitive(fs.hog_feature,[filterData],filterData, name='HoG')
# Gaussian features
pset.addPrimitive(fs.gau, [filterData,coordsX2], filterData, name='Gau2')
pset.addPrimitive(fs.gauD, [filterData,coordsX2, coordsX1,coordsX1], filterData, name='Gau_D2')
#general filters
##pset.addPrimitive(fs.gau, [filterData,coordsX2], filterData, name='Gau1')
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

def evalTrain(individual):
    try:
        func = toolbox.compile(expr=individual)
        train_tf = []
        for i in range(0, len(y_train)):
            train_tf.append(numpy.asarray(func(x_train[i, :, :])))
        train_tf = numpy.asarray(train_tf, dtype=float)
        min_max_scaler = preprocessing.MinMaxScaler()
        train_norm = min_max_scaler.fit_transform(train_tf)
        lsvm = LinearSVC()
        accuracy = round(100 * cross_val_score(lsvm, train_norm, y_train, cv=5).mean(), 2)
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
    hof = tools.HallOfFame(10)
    log = tools.Logbook()
    stats_fit = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats_size_tree = tools.Statistics(key=len)
    stats_size_feature = tools.Statistics(key= lambda ind: feature_length(ind, x_train[1,:,:], toolbox))
    mstats = tools.MultiStatistics(fitness=stats_fit,size_tree=stats_size_tree, size_feature = stats_size_feature)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)
    log.header = ["gen", "evals"] + mstats.fields

    pop, log = evalGP.eaSimple(pop, toolbox, cxProb, mutProb,elitismProb, generation,
                    stats=mstats, halloffame=hof, verbose=True)

    return pop,log, hof

if __name__ == "__main__":

    beginTime = time.process_time()
    pop, log, hof = GPMain(randomSeeds)
    endTime = time.process_time()
    trainTime = endTime - beginTime

    train_tf, test_tf, trainLabel, testL, testResults = evalTest(toolbox, hof[0], x_train, y_train,x_test, y_test)

    saveFile.saveLog(str(randomSeeds)+dataSetName+'all_pop.pickle', pop)
    saveFile.saveLog(str(randomSeeds)+dataSetName+'best_pop.pickle', hof)
    
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
