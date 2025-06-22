from descriptor_func import histuLBP
from descriptor_func import hog_features_patches as hog_features
import sift_features
import numpy

def root_con(*args):
    feature_vector=numpy.concatenate((args),axis=0)
    #print(feature_vector.shape)
    return feature_vector

def root_fcon2(vector1, f1, vector2, f2):
    feature_vector=numpy.concatenate((vector1*f1, vector2*f2),axis=0)
    #print(feature_vector.shape)
    return feature_vector

def root_fcon3(vector1, f1, vector2, f2, vector3, f3):
    feature_vector=numpy.concatenate((vector1*f1, vector2*f2, vector3*f3),axis=0)
    #print(feature_vector.shape)
    return feature_vector

def root_one(input):
    feature_vector=input
    #print(feature_vector.shape)
    return feature_vector

def global_hog_small(image):
    feature_vector = hog_features(image, 4, 4)
    # dimension 144 for 128*128
    return feature_vector

def global_hog(image):
    feature_vector = hog_features(image, 20, 10)
    # dimension 144 for 128*128
    return feature_vector

def local_hog(image):
    feature_vector=hog_features(image,10,10)
    #dimension don't know
    return feature_vector

def all_lbp(image):
    # global and local
    feature_vector = histuLBP(image, radius=1.5, n_points=8)
    # dimension 59 for all images
    return feature_vector

def all_sift(image):
    # global and local
    width,height=image.shape
    min_length=numpy.min((width,height))
    img=numpy.asarray(image[0:width,0:height])
    extractor = sift_features.SingleSiftExtractor(min_length)
    feaArrSingle = extractor.process_image(img[0:min_length,0:min_length])
    # dimension 128 for all images
    w,h=feaArrSingle.shape
    feature_vector=numpy.reshape(feaArrSingle, (h,))
    return feature_vector
