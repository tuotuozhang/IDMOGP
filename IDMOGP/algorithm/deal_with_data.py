import numpy as np
from sklearn.model_selection import train_test_split
import numpy
from PIL import Image

def deal_data(x_train, y_train, x_test, y_test, dataset_name, num_instance, n_class):
    if y_train.min() == 1:
        y_train -= 1
        y_test -= 1
    if dataset_name=='att':
        x_all = numpy.concatenate((x_train, x_test), axis=0)
        y_all = numpy.concatenate((y_train, y_test), axis=0)
        x_all_new = []
        for i in range(len(x_all)):
            img = Image.fromarray(x_all[i])
            img_new = img.resize((46, 56))
            x_all_new.append(numpy.asarray(img_new))
        x_all = numpy.asarray(x_all_new)
        # print(x_all.shape)
        x_train_select, x_test, y_train_select, y_test = train_test_split(x_all, y_all, train_size=num_instance * n_class, random_state=5, stratify=y_all)
        # print('0', x_train_select.shape, x_test.shape, y_train_select.shape, y_test.shape)
    elif dataset_name =='eyale':
        x_all = numpy.concatenate((x_train, x_test), axis=0)
        y_all = numpy.concatenate((y_train, y_test), axis=0)
        x_all_new = []
        for i in range(len(x_all)):
            img = Image.fromarray(x_all[i])
            img_new = img.resize((32, 32))
            x_all_new.append(numpy.asarray(img_new))
        x_all = numpy.asarray(x_all_new)
        x_train_select, x_test, y_train_select, y_test = train_test_split(x_all, y_all, train_size=num_instance * n_class, random_state=5, stratify=y_all)
    else:
        if len(x_train.shape) == 4:
            x_train = 0.3 * x_train[:, :, :, 0] + 0.59 * x_train[:, :, :, 1] + 0.11 * x_train[:, :, :, 2]
            x_test = 0.3 * x_test[:, :, :, 0] + 0.59 * x_test[:, :, :, 1] + 0.11 * x_test[:, :, :, 2]
        x_train_select, x_test, y_train_select, y_test = train_test_split(x_train, y_train, train_size=num_instance * n_class, random_state=5, stratify=y_train)
    # print('1', x_train_select.shape, x_test.shape, y_train_select.shape, y_test.shape)
    return x_train_select, x_test, y_train_select, y_test


def check_image(x_all, y_all, x_train_select):
    x_test = []
    y_test = []
    for i in range(len(y_all)):
        for j in range(x_train_select.shape[0]):
            if np.array_equal(x_all[i], x_train_select[j], equal_nan=False):
                break
        else:
            x_test.append(x_all[i])
            y_test.append(y_all[i])
    # print(len(x_test))
    return np.asarray(x_test), np.asarray(y_test)


def deal_data_sbgp2(x_train_select, y_train_select, x_train, y_train, x_test, y_test, dataset_name):
    if y_train.min() == 1:
        y_train -= 1
        y_test -= 1
    if dataset_name=='att':
        x_all = numpy.concatenate((x_train, x_test), axis=0)
        y_all = numpy.concatenate((y_train, y_test), axis=0)
        x_all_new = []
        for i in range(len(x_all)):
            img = Image.fromarray(x_all[i])
            img_new = img.resize((46, 56))
            x_all_new.append(numpy.asarray(img_new))
        x_all = numpy.asarray(x_all_new)
        # print( x_all.shape, x_all.min(), x_all.max())
        x_test, y_test = check_image(x_all, y_all, x_train_select)
        # print('0', x_train_select.shape, x_test.shape, y_train_select.shape, y_test.shape)
    elif dataset_name =='eyale':
        x_all = numpy.concatenate((x_train, x_test), axis=0)
        y_all = numpy.concatenate((y_train, y_test), axis=0)
        x_all_new = []
        for i in range(len(x_all)):
            img = Image.fromarray(x_all[i])
            img_new = img.resize((32, 32))
            x_all_new.append(numpy.asarray(img_new))
        x_all = numpy.asarray(x_all_new)
        x_test, y_test = check_image(x_all, y_all, x_train_select)
    else:
        if len(x_test.shape) == 4:
            x_test = 0.3 * x_test[:, :, :, 0] + 0.59 * x_test[:, :, :, 1] + 0.11 * x_test[:, :, :, 2]
    return x_train_select, x_test, y_train_select, y_test