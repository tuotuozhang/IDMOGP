from skimage.feature import local_binary_pattern
from skimage.feature import hog

def LBP(image,radius,n_points):
    # 'uniform','default','ror','var'
    lbp = local_binary_pattern(image, n_points, radius)
    return lbp

def histLBP(image,radius,n_points):
    #uniform_LBP
    lbp=LBP(image,radius=radius,n_points=n_points)
    n_bins = 256
    hist,ax=numpy.histogram(lbp,n_bins,[0,n_bins])
    return hist
import numpy

def uniform_LBP(image,radius,n_points):
    # 'uniform','default','ror','var'
    lbp = local_binary_pattern(image, n_points, radius, method='nri_uniform')
    return lbp

def histuLBP(image,radius,n_points):
    #uniform_LBP
    lbp=uniform_LBP(image,radius=radius,n_points=n_points)
    n_bins = 59
    hist,ax=numpy.histogram(lbp,n_bins,[0,59])
    return hist
def HoGFeatures(image):
    try:
        img,realImage=hog(image,orientations=9, pixels_per_cell=(8, 8),
                    cells_per_block=(3, 3), block_norm='L2-Hys', visualize=True, visualise=None,
                    transform_sqrt=False, feature_vector=True)
        return realImage
    except:
        return image

def hog_features_patches(image,patch_size,moving_size):
    img=numpy.asarray(image)
    width, height = img.shape
    w = int(width / moving_size)
    h = int(height / moving_size)
    patch = []
    for i in range(0, w):
        for j in range(0, h):
            patch.append([moving_size * i, moving_size * j])
    hog_features = numpy.zeros((len(patch)))
    realImage=HoGFeatures(img)
    for i in range(len(patch)):
        hog_features[i] = numpy.mean(
            realImage[patch[i][0]:(patch[i][0] + patch_size), patch[i][1]:(patch[i][1] + patch_size)])
    #print(hog_features, hog_features.shape)
    return hog_features
