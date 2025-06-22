import numpy
from skimage.filters import gabor


def gabor_features_patches(image,patch_size,moving_size):
    img=numpy.asarray(image)
    width, height = img.shape
    w = int(width / moving_size)
    h = int(height / moving_size)
    patch = []
    for i in range(0, w):
        for j in range(0, h):
            patch.append([moving_size * i, moving_size * j])
    gaobor_features_one = numpy.zeros((len(patch)))
    realImage=img
    for i in range(len(patch)):
        gaobor_features_one[i] = numpy.mean(
            realImage[patch[i][0]:(patch[i][0] + patch_size), patch[i][1]:(patch[i][1] + patch_size)])
    #print(hog_features, hog_features.shape)
    return gaobor_features_one

def gabor_features(image,patch_size,moving_size):
    image=numpy.asarray(image)
    width,height=image.shape
    fmax=numpy.pi/2
    a=numpy.sqrt(2)
    orientation=[0, numpy.pi/8, numpy.pi/4, numpy.pi*3/8, numpy.pi/2,numpy.pi*5/8, numpy.pi*3/4 ,numpy.pi*7/8]
    frequency=[fmax/(a**0), fmax/(a**1),fmax/(a**2),fmax/(a**3),fmax/(a**4)]
    img_filted=numpy.zeros((len(orientation)*len(frequency),width,height))

    for i in range(len(orientation)):
        for j in range(len(frequency)):
            filt_real, filt_imag=numpy.asarray(gabor(image,theta=orientation[i],frequency=frequency[j]))
            #print(filt_real, filt_imag.shape)
            img_filted[len(frequency)*i+j,:]=filt_real
    #print(img_filted.shape)

    gabor_features_all=numpy.asarray([])
    for i in range(len(orientation)*len(frequency)):
        gabor_features=gabor_features_patches(img_filted[i,:,:],patch_size,moving_size)
        gabor_features_all=numpy.concatenate((gabor_features_all,gabor_features))
    #print(gabor_features_all.shape)
    return gabor_features_all
