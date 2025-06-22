import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing

def feature_length(ind, instances, y_instances, toolbox):
    func=toolbox.compile(ind)
    try:
        feature_len = len(func(instances, y_instances))
    except: feature_len=0
    return feature_len,

def evalTrain_old(toolbox, individual, hof, trainData, trainLabel):
    if len(hof) != 0 and individual in hof:
        ind = 0
        while ind < len(hof):
            if individual == hof[ind]:
                accuracy, = hof[ind].fitness.values
                ind = len(hof)
            else: ind+=1
    else:
        try:
            func = toolbox.compile(expr=individual)
            train_tf = []
            for i in range(0,len(trainLabel)):
                train_tf.append(np.asarray(func(trainData[i, :, :])))
            train_tf  = np.asarray(train_tf, dtype=float)
            min_max_scaler = preprocessing.MinMaxScaler()
            train_norm = min_max_scaler.fit_transform(train_tf)
            lsvm= LinearSVC()
            accuracy = round(100*cross_val_score(lsvm, train_norm, trainLabel, cv=5).mean(),2)
        except:
            accuracy=0
    return accuracy,


def evalTrainp(toolbox, individual, hof, trainData, trainLabel):
    if individual in hof and hof is not None:
        for i in range(len(hof)):
            if individual==hof[i]:
                ind = i
        accuracy,=hof[ind].fitness.values
    else:
        try:
            func = toolbox.compile(expr=individual)
            train_tf = []
            for i in range(0,len(trainLabel)):
                train_tf.append(np.asarray(func(trainData[i, :, :,0],trainData[i, :, :,1],trainData[i, :, :,2])))
            min_max_scaler = preprocessing.MinMaxScaler()
            train_norm = min_max_scaler.fit_transform(np.asarray(train_tf))
            lsvm= LinearSVC()
            accuracy = round(100*cross_val_score(lsvm, train_norm, trainLabel, cv=5).mean(),2)
        except:
            accuracy=0
    return accuracy,

def evalTest_fromvector(toolbox, individual, trainData, trainLabel, test, testL):
    func = toolbox.compile(expr=individual)
    train_combine = np.concatenate((trainData, test), axis=0)
    label_combine = np.concatenate((trainLabel, testL), axis=0)
    train_all= np.asarray(func(train_combine, trainLabel))
    # train_all = np.asarray(func(train_combine))
    print(train_all.shape)
    train_tf = train_all[0:len(trainLabel), :]
    test_tf = train_all[len(trainLabel):, :]
    # test_tf = np.asarray(func(test))
    # train_tf = np.asarray(func(trainData))
    min_max_scaler = preprocessing.MinMaxScaler()
    train_norm = min_max_scaler.fit_transform(train_tf)
    test_norm = min_max_scaler.transform(test_tf)
    lsvm= LinearSVC()
    lsvm.fit(train_norm, trainLabel)
##    print(np.asarray(train_tf).shape, np.asarray(test_tf).shape)    
    accuracy = round(100*lsvm.score(test_norm, testL), 2)
    return np.asarray(train_tf), np.asarray(test_tf), trainLabel, testL, accuracy

def evalTest_fromvector_FGP(toolbox, individual, trainData, trainLabel, test, testL):
    func = toolbox.compile(expr=individual)
    train_combine = np.concatenate((trainData, test), axis=0)
    label_combine = np.concatenate((trainLabel, testL), axis=0)
    train_all = []
    for i in range(0, len(label_combine)):
        # print(x_train_combine[i, :, :])
        # print(func(x_train_combine[i, :, :]))
        train_all.append(np.asarray(func(train_combine[i, :, :])))
    # train_all= np.asarray(func(train_combine, trainLabel))
    # train_all = np.asarray(func(train_combine))
    # print(train_all.shape)
    train_all = np.asarray(train_all, dtype=float)
    train_tf = train_all[0:len(trainLabel), :]
    test_tf = train_all[len(trainLabel):, :]
    # test_tf = np.asarray(func(test))
    # train_tf = np.asarray(func(trainData))
    min_max_scaler = preprocessing.MinMaxScaler()
    train_norm = min_max_scaler.fit_transform(train_tf)
    test_norm = min_max_scaler.transform(test_tf)
    lsvm= LinearSVC()
    lsvm.fit(train_norm, trainLabel)
##    print(np.asarray(train_tf).shape, np.asarray(test_tf).shape)
    accuracy = round(100*lsvm.score(test_norm, testL), 2)
    return np.asarray(train_tf), np.asarray(test_tf), trainLabel, testL, accuracy
