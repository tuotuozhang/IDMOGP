import sift_features
from pylab import *
from scipy import ndimage
from skimage.filters import gabor
import skimage
from skimage.feature import local_binary_pattern
from skimage.feature import hog
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
import logging
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn import preprocessing
from sklearn.preprocessing import power_transform
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from fisher_score import fisher_score
from chi_square import chi_square
from f_score import f_score
from gini_index import gini_index
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import mutual_info_classif

from CMIM import cmim
from DISR import disr
# from JMI import jmi
from reliefF import reliefF
from MRMR import mrmr
from MIFS import mifs
from RFS import rfs
from MCFS import mcfs

def fisher_score_fs(X, y, precentage):
    # print(X.shape)
    if X.shape[0]==len(y):
        f_scoress = fisher_score(X, y)
    else:
        f_scoress = fisher_score(X[0:len(y),:], y)
    # print(f_score)
    idx = np.argsort(f_scoress)
    f_rank = idx[::-1]
    f_sel = f_rank[0:int(0.1*precentage*len(f_scoress))]
    # print(len(f_sel))
    x_new = X[:, f_sel]
    # print(x_new.shape)
    return x_new

def mutual_info_fs(X, y, precentage):
    # print(X.shape)
    X = (abs(X)+X)/2
    if X.shape[0]==len(y):
        f_scoress = mutual_info_classif(X, y)
    else:
        f_scoress = mutual_info_classif(X[0:len(y),:], y)
    # print(f_score)
    idx = np.argsort(f_scoress)
    f_rank = idx[::-1]
    f_sel = f_rank[0:int(0.1*precentage*len(f_scoress))]
    # print(len(f_sel))
    x_new = X[:, f_sel]
    # print(x_new.shape)
    return x_new

def chi_sq_fs(X, y, precentage):
    # print(X.shape)
    X = (abs(X)+X)/2
    if X.shape[0]==len(y):
        f_scoress = chi_square(X, y)
    else:
        f_scoress = chi_square(X[0:len(y),:], y)
    # print(f_score)
    idx = np.argsort(f_scoress)
    f_rank = idx[::-1]
    f_sel = f_rank[0:int(0.1*precentage*len(f_scoress))]
    # print(len(f_sel))
    x_new = X[:, f_sel]
    # print(x_new.shape)
    return x_new

def f_score_fs(X, y, precentage):
    # print(X.shape)
    if X.shape[0]==len(y):
        f_scoress = f_score(X, y)
    else:
        f_scoress = f_score(X[0:len(y),:], y)
    # print(f_scoress)
    idx = np.argsort(f_scoress)
    f_rank = idx[::-1]
    f_sel = f_rank[0:int(0.1*precentage*len(f_scoress))]
    # print(len(f_sel))
    x_new = X[:, f_sel]
    # print(x_new.shape)
    return x_new

def gini_index_fs(X, y, precentage):
    # print(X.shape)
    if X.shape[0]==len(y):
        f_scoress = gini_index(X, y)
    else:
        f_scoress = gini_index(X[0:len(y),:], y)
    # print(f_scoress)
    idx = np.argsort(f_scoress)
    f_rank = idx[::-1]
    f_sel = f_rank[0:int(0.1*precentage*len(f_scoress))]
    # print(f_rank)
    x_new = X[:, f_sel]
    # print(x_new.shape)
    return x_new

def fs_variance(x_data, para):
    #remove features that more than para% are zero or one
    #need to recale the images into 0 or 1
    try:
        fs = VarianceThreshold(threshold=(0.1*para * (1 - 0.1*para)))
        x_tf = fs.fit_transform(x_data)
    except: x_tf = x_data
    # print(x_tf.shape)
    return x_tf

def relif_fs(X, y, precentage):
    # print(X.shape)
    if X.shape[0]==len(y):
        f_scoress = reliefF(X, y)
    else:
        f_scoress = reliefF(X[0:len(y),:], y)
    # print(f_scoress)
    idx = np.argsort(f_scoress)
    f_rank = idx[::-1]
    f_sel = f_rank[0:int(0.1*precentage*len(f_scoress))]
    # print(f_rank)
    x_new = X[:, f_sel]
    # print(x_new.shape)
    return x_new

def cmim_fs(X, y, precentage):
    # print(X.shape)
    num_f = int(0.1*precentage*X.shape[1])
    if X.shape[0]==len(y):
        f, j_cmim, mify = cmim(X, y, n_selected_features=num_f)
    else:
        f, j_cmim, mify = cmim(X[0:len(y),:], y,n_selected_features=num_f)
    # print(f, j_cmim, mify)
    # print(f)
    x_new = X[:, f]
    # print(x_new.shape)
    return x_new

def disr_fs(X, y, precentage):
    # print(X.shape)
    num_f = int(0.1*precentage*X.shape[1])
    if X.shape[0]==len(y):
        f, j_disr, mify = disr(X, y, n_selected_features=num_f)
    else:
        f, j_disr, mify  = disr(X[0:len(y), :], y, n_selected_features=num_f)
    # print(f_scoress)
    # print(f)
    x_new = X[:, f]
    # print(f_rank)
    # print(x_new.shape)
    return x_new

def mrmr_fs(X, y, precentage):
    # print(X.shape)
    num_f = int(0.1*precentage*X.shape[1])
    if X.shape[0]==len(y):
        f, j_disr, mify = mrmr(X, y, n_selected_features=num_f)
    else:
        f, j_disr, mify  = mrmr(X[0:len(y), :], y, n_selected_features=num_f)
    # print(f_scoress)
    # print(f)
    x_new = X[:, f]
    # print(f_rank)
    # print(x_new.shape)
    return x_new

def mifs_fs(X, y, precentage):
    # print(X.shape)
    num_f = int(0.1*precentage*X.shape[1])
    if X.shape[0]==len(y):
        f, j_disr, mify = mifs(X, y, n_selected_features=num_f)
    else:
        f, j_disr, mify  = mifs(X[0:len(y), :], y, n_selected_features=num_f)
    # print(f_scoress)
    # print(f)
    x_new = X[:, f]
    # print(f_rank)
    # print(x_new.shape)
    return x_new

def rfs_fs(X, y, precentage):
    '''has problem because the output f_scores is a matrix '''
    lb = preprocessing.LabelBinarizer()
    y_binary = lb.fit_transform(y)
    # print(X.shape)
    num_f = int(0.1*precentage*X.shape[1])
    if X.shape[0]==len(y):
        f_scoress = rfs(X, y_binary)
    else:
        f_scoress = rfs(X[0:len(y), :], y_binary)
    print(f_scoress)
    idx = np.argsort(f_scoress)
    f_rank = idx[::-1]
    f_sel = f_rank[0:int(0.1 * precentage * len(f_scoress))]
    # print(f_rank)
    x_new = X[:, f_sel]
    # print(x_new.shape)
    return x_new

def mcfs_fs(X, y, precentage):
    # print(X.shape)
    num_f = int(0.1*precentage*X.shape[1])
    if X.shape[0]==len(y):
        f_scoress = mcfs(X, y)
    else:
        f_scoress = mcfs(X[0:len(y), :], y)
    idx = np.argsort(f_scoress)
    f_rank = idx[::-1]
    f_sel = f_rank[0:int(0.1 * precentage * len(f_scoress))]
    # print(f_rank)
    x_new = X[:, f_sel]
    # print(x_new.shape)
    return x_new

def log_tf(XX):
    XX[XX<=0] = 1
    x = np.log(XX,)
    return x

def reciprocal_tf(XX):
    with np.errstate(divide='ignore', invalid='ignore'):
        x = np.divide(1, XX)
        if isinstance(x, np.ndarray):
            x[np.isinf(x)] = 1
            x[np.isnan(x)] = 1
        elif np.isinf(x) or np.isnan(x):
            x = 1
    return x

def sqrt_tf(XX):
    with np.errstate(divide='ignore', invalid='ignore'):
        x = np.sqrt(XX, )
        if isinstance(x, np.ndarray):
            x[np.isinf(x)] = 1
            x[np.isnan(x)] = 1
        elif np.isinf(x) or np.isnan(x):
            x = 1
    return x


def exponential_tf(XX, m):
    x = XX**m
    return x

def box_cox_tf(XX):
    x = power_transform(XX,method='box-cox')
    return x

def yeo_johnson_tf(XX):
    x = power_transform(XX,method='yeo-johnson')
    return x

def MinMax(XX):
    scale = MinMaxScaler()
    x = scale.fit_transform(X=XX)
    return x

def MaxAbs(XX):
    scale = MaxAbsScaler()
    x = scale.fit_transform(X=XX)
    return x

def Standard(XX):
    scale = StandardScaler()
    x = scale.fit_transform(X=XX)
    return x

def Robust(XX):
    scale = RobustScaler()
    x = scale.fit_transform(X=XX)
    return x

def Normal(XX):
    scale = Normalizer()
    x = scale.fit_transform(X=XX)
    return x

# def Quantile(features):
#     x = QuantileTransformer(features,quantile_range=(25, 75))
#     return x

def combine(*args):
    output = args[0]
    for i in range(1, len(args)):
        output += args[i]
    #print(output.shape)
    return output

def svm_train_model(model, x, y, k=3):
    min_max_scaler = preprocessing.MinMaxScaler()
    x = min_max_scaler.fit_transform(np.asarray(x))
    kf = StratifiedKFold(n_splits=k)
    ni = np.unique(y)
    num_class = ni.shape[0]
    y_predict = np.zeros((len(y), num_class))
    for train_index, test_index in kf.split(x,y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(x_train,y_train)
        y_pred = model.predict(x_test)
        y_label = []
        for i in y_pred:
            binary_label = np.zeros((num_class))
            binary_label[int(i)] = 1
            y_label.append(binary_label)
        y_predict[test_index,:] = np.asarray(y_label)
    return y_predict

def test_function_svm(model, x_train, y_train, x_test):
    min_max_scaler = preprocessing.MinMaxScaler()
    x_train = min_max_scaler.fit_transform(np.asarray(x_train))
    x_test = min_max_scaler.transform(np.asarray(x_test))
    logging.info('Training set shape in testing '+str(x_train.shape))
    logging.info('Test set shape in testing'+str(x_test.shape))
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_label = []
    ni = np.unique(y_train)
    num_class = ni.shape[0]
    for i in y_pred:
        binary_label = np.zeros((num_class))
        binary_label[int(i)] = 1
        y_label.append(binary_label)
    y_predict = np.asarray(y_label)
    return y_predict

def train_model_prob(model, x, y, k=3):
    kf = StratifiedKFold(n_splits=k)
    ni = np.unique(y)
    num_class = ni.shape[0]
    y_predict = np.zeros((len(y), num_class))
    for train_index, test_index in kf.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(x_train,y_train)
        y_predict[test_index,:] = model.predict_proba(x_test)
    return y_predict

def test_function_prob(model, x_train, y_train, x_test):
    logging.info('Training set shape in testing '+str(x_train.shape))
    logging.info('Test set shape in testing'+str(x_test.shape))
    model.fit(x_train, y_train)
    y_pred = model.predict_proba(x_test)
    return y_pred

def linear_svm(x_train, y_train, cm=0):
    #parameters c
    c = 10**(cm)
    classifier = LinearSVC(C=c)
    num_train = y_train.shape[0]
    if num_train == x_train.shape[0]:
        y_labels = svm_train_model(classifier, x_train, y_train)
    else:
        y_labels = test_function_svm(classifier, x_train[0:num_train,:], y_train, x_train[num_train:x_train.shape[0],:])
    return y_labels

def lr(x_train, y_train, cm=0):
    c = 10**(cm)
    classifier = LogisticRegression(C=c, solver='sag', multi_class= 'auto', max_iter=1000)
    num_train = y_train.shape[0]
    if num_train==x_train.shape[0]:
        y_labels = svm_train_model(classifier, x_train, y_train)
    else:
        y_labels = test_function_svm(classifier, x_train[0:num_train,:], y_train, x_train[num_train:x_train.shape[0],:])
    return y_labels

def randomforest(x_train, y_train, n_tree = 500, max_dep = 100):
    classifier = RandomForestClassifier(n_estimators=n_tree, max_depth=max_dep)
    num_train = y_train.shape[0]
    if num_train == x_train.shape[0]:
        y_labels = train_model_prob(classifier, x_train, y_train)
    else:
        y_labels = test_function_prob(classifier, x_train[0:num_train,:], y_train, x_train[num_train:x_train.shape[0],:])
    return y_labels

def erandomforest(x_train, y_train, n_tree = 500, max_dep = 100):
    classifier = ExtraTreesClassifier(n_estimators=n_tree, max_depth=max_dep)
    num_train = y_train.shape[0]
    if num_train == x_train.shape[0]:
        y_labels = train_model_prob(classifier, x_train, y_train)
    else:
        y_labels = test_function_prob(classifier, x_train[0:num_train,:], y_train, x_train[num_train:x_train.shape[0],:])
    return y_labels

def conVector(img):
    try:
        img_vector=np.concatenate((img))
    except ValueError:
        img_vector=img
    return img_vector

def FeaCon2(img1, img2):
    x_features = []
    for i in range(img1.shape[0]):
        image1 = conVector(img1[i, :])
        image2 = conVector(img2[i, :])
        feature_vector = np.concatenate((image1, image2), axis=0)
        x_features.append(feature_vector)
    return np.asarray(x_features)

def FeaCon3(img1, img2, img3):
    x_features = []
    for i in range(img1.shape[0]):
        image1 = conVector(img1[i, :])
        image2 = conVector(img2[i, :])
        image3 = conVector(img3[i, :])
        feature_vector = np.concatenate((image1, image2, image3), axis=0)
        x_features.append(feature_vector)
    return np.asarray(x_features)

def FeaCon4(img1, img2, img3, img4):
    x_features = []
    for i in range(img1.shape[0]):
        image1 = conVector(img1[i, :])
        image2 = conVector(img2[i, :])
        image3 = conVector(img3[i, :])
        image4 = conVector(img4[i, :])
        feature_vector = np.concatenate((image1, image2, image3, image4), axis=0)
        x_features.append(feature_vector)
    return np.asarray(x_features)

def all_lbp(image):
    #uniform_LBP
    # global and local
    feature = []
    n_bins = 59
    for i in range(image.shape[0]):
        lbp = local_binary_pattern(image[i, :, :], P=8, R=1.5, method='nri_uniform')
        hist,ax=np.histogram(lbp,n_bins,[0,59])
        feature.append(hist)
    return np.asarray(feature)

def HoGFeatures(image):
    try:
        img,realImage=hog(image,orientations=9, pixels_per_cell=(8, 8),
                    cells_per_block=(3, 3), block_norm='L2-Hys', visualize=True,
                    transform_sqrt=False, feature_vector=True)
        return realImage
    except:
        return image

def hog_features_patches(image,patch_size,moving_size):
    img=np.asarray(image)
    width, height = img.shape
    w = int(width / moving_size)
    h = int(height / moving_size)
    patch = []
    for i in range(0, w):
        for j in range(0, h):
            patch.append([moving_size * i, moving_size * j])
    hog_features = np.zeros((len(patch)))
    realImage=HoGFeatures(img)
    for i in range(len(patch)):
        hog_features[i] = np.mean(
            realImage[patch[i][0]:(patch[i][0] + patch_size), patch[i][1]:(patch[i][1] + patch_size)])
    return hog_features

def global_hog_small(image):
    feature = []
    for i in range(image.shape[0]):
        feature_vector = hog_features_patches(image[i,:,:], 4, 4)
        feature.append(feature_vector)
    return np.asarray(feature)

def all_sift(image):
    width, height = image[0, :, :].shape
    min_length = np.min((width, height))
    feature = []
    for i in range(image.shape[0]):
        img = np.asarray(image[i, 0:width, 0:height])
        extractor = sift_features.SingleSiftExtractor(min_length)
        feaArrSingle = extractor.process_image(img[0:min_length, 0:min_length])
        # dimension 128 for all images
        # print(feaArrSingle.shape)
        w, h = feaArrSingle.shape
        feature_vector = np.reshape(feaArrSingle, (h,))
        feature.append(feature_vector)
    return np.asarray(feature)

def featureMeanStd(region):
    mean=np.mean(region)
    std=np.std(region)
    return mean,std

def fetureDIF(image):
    feature=np.zeros((20))
    width,height=image.shape
    width1=int(width/2)
    height1=int(height/2)
    width2=int(width/4)
    height2=int(height/4)
    #A1B1C1D1
    feature[0],feature[1]=featureMeanStd(image)
    #A1E1OG1
    feature[2],feature[3]=featureMeanStd(image[0:width1,0:height1])
    #E1B1H1O
    feature[4],feature[5]=featureMeanStd(image[0:width1,height1:height])
    #G1OF1D1
    feature[6],feature[7]=featureMeanStd(image[width1:width,0:height1])
    #OH1C1F1
    feature[8],feature[9]=featureMeanStd(image[width1:width,height1:height])
    #A2B2C2D2
    feature[10],feature[11]=featureMeanStd(image[width2:(width2+width1),height2:(height1+height2)])
    #G1H1
    feature[12],feature[13]=featureMeanStd(image[width1,:])
    #E1F1
    feature[14],feature[15]=featureMeanStd(image[:,height1])
    #G2H2
    feature[16],feature[17]=featureMeanStd(image[width1,height2:(height1+height2)])
    #E2F2
    feature[18],feature[19]=featureMeanStd(image[width2:(width2+width1),height1])
    return feature

def all_dif(image):
    #global and local
    # dimension 20 for all type images
    feature = []
    for i in range(image.shape[0]):
        feature_vector = fetureDIF(image[i])
        feature.append(feature_vector)
    return np.asarray(feature)


def all_histogram(image):
    # global and local
    n_bins = 32
    feature = []
    for i in range(image.shape[0]):
        hist, ax = np.histogram(image[i], n_bins, [0, 1])
        feature.append(hist)
    # dimension 24 for all type images
    return np.asarray(feature)

def gau(left, si):
    img = []
    for i in range(left.shape[0]):
        img.append(ndimage.gaussian_filter(left[i, :, :], sigma=si))
    return np.asarray(img)

def gauD(left, si, or1, or2):
    img  = []
    for i in range(left.shape[0]):
        img.append(ndimage.gaussian_filter(left[i,:,:],sigma=si, order=[or1,or2]))
    return np.asarray(img)

def gab(left,the,fre):
    fmax=np.pi/2
    a=np.sqrt(2)
    freq=fmax/(a**fre)
    thea=np.pi*the/8
    img = []
    for i in range(left.shape[0]):
        filt_real,filt_imag=np.asarray(gabor(left[i,:,:],theta=thea,frequency=freq))
        img.append(filt_real)
    return np.asarray(img)

def laplace(left):
    img = []
    for i in range(left.shape[0]):
        img.append(ndimage.laplace(left[i, :, :]))
    return np.asarray(img)

def gaussian_Laplace1(left):
    img = []
    for i in range(left.shape[0]):
        img.append(ndimage.gaussian_laplace(left[i, :, :], sigma=1))
    return np.asarray(img)

def gaussian_Laplace2(left):
    img = []
    for i in range(left.shape[0]):
        img.append(ndimage.gaussian_laplace(left[i, :, :], sigma=2))
    return np.asarray(img)

def sobelxy(left):
    img = []
    for i in range(left.shape[0]):
        img.append(ndimage.sobel(left[i, :, :]))
    return np.asarray(img)

def sobelx(left):
    img = []
    for i in range(left.shape[0]):
        img.append(ndimage.sobel(left[i,:,:], axis=0))
    return np.asarray(img)

def sobely(left):
    img = []
    for i in range(left.shape[0]):
        img.append(ndimage.sobel(left[i, :, :], axis=1))
    return np.asarray(img)

#max filter
def maxf(image):
    img = []
    size = 3
    for i in range(image.shape[0]):
        img.append(ndimage.maximum_filter(image[i,:,:],size))
    return np.asarray(img)

#median_filter
def medianf(image):
    img = []
    size = 3
    for i in range(image.shape[0]):
        img.append(ndimage.median_filter(image[i,:,:],size))
    return np.asarray(img)

#mean_filter
def meanf(image):
    img = []
    size = 3
    for i in range(image.shape[0]):
        img.append(ndimage.convolve(image[i,:,:], np.full((3, 3), 1 / (size * size))))
    return np.asarray(img)

#minimum_filter
def minf(image):
    img = []
    size = 3
    for i in range(image.shape[0]):
        img.append(ndimage.minimum_filter(image[i,:,:],size))
    return np.asarray(img)

def lbp(image):
    img = []
    all = image.shape[1]*image.shape[2]
    for i in range(image.shape[0]):
        # 'uniform','default','ror','var'
        lbp = local_binary_pattern(image[i,:,:], 8, 1.5, method='nri_uniform')
        img.append(np.divide(lbp,all))
    return np.asarray(img)


def hog_feature(image):
    try:
        img = []
        for i in range(image.shape[0]):
            img1, realImage = hog(image[i, :, :], orientations=9, pixels_per_cell=(8, 8),
                                cells_per_block=(3, 3), block_norm='L2-Hys', visualize=True,
                                transform_sqrt=False, feature_vector=True)
            img.append(realImage)
        data = np.asarray(img)
    except: data = image
    return data


def mis_match(img1,img2):
    n, w1,h1=img1.shape
    n, w2,h2=img2.shape
    w=min(w1,w2)
    h=min(h1,h2)
    return img1[:, 0:w,0:h],img2[:, 0:w,0:h]

def mixconadd(img1, w1,img2, w2):
    img11,img22=mis_match(img1,img2)
    return np.add(img11*w1,img22*w2)

def mixconsub(img1, w1,img2, w2):
    img11,img22=mis_match(img1,img2)
    return np.subtract(img11*w1,img22*w2)

def sqrt(left):
    with np.errstate(divide='ignore',invalid='ignore'):
        x = np.sqrt(left,)
        if isinstance(x, np.ndarray):
            x[np.isinf(x)] = 1
            x[np.isnan(x)] = 1
        elif np.isinf(x) or np.isnan(x):
            x = 1
    return x

def relu(left):
    return (abs(left)+left)/2

def maxP(left, kel1, kel2):
    img = []
    for i in range(left.shape[0]):
        current = skimage.measure.block_reduce(left[i], (kel1,kel2),np.max)
        img.append(current)
    return np.asarray(img)

def regionS(left,x,y,windowSize):
    n_, width,height = left.shape
    x_end = min(width, x+windowSize)
    y_end = min(height, y+windowSize)
    slice = left[:, x:x_end, y:y_end]
    return slice

def regionR(left, x, y, windowSize1,windowSize2):
    n_, width, height = left.shape
    x_end = min(width, x + windowSize1)
    y_end = min(height, y + windowSize2)
    slice = left[:, x:x_end, y:y_end]
    return slice

