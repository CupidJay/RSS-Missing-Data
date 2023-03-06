from .uci_loader import *
from .imputer import *
import os
import torch
import random
import numpy as np
from copy import deepcopy
import scipy.io as scio
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import datetime

def gaussian_list2file(filename, params):
    '''
    :param l:
    (mean, std) for each item in params
    :return:
    '''
    with open(filename, 'w') as f:
        for (mean, std) in params:
            f.write('{} {}\n'.format(mean, std))

def gaussian_file2list(filename):
    '''

    :param filename:
    :return: params list saved in filename
    '''
    params = []
    with open(filename, 'r') as f:
        filelines = f.readlines()
        for fileline in filelines:
            temp = fileline.strip().split(' ')
            mean, std = float(temp[0]), float(temp[1])
            params.append((mean, std))
    return params

def percentile_list2file(filename, params):
    '''
    :param l:
    percentiles for each item in params
    :return:
    '''
    with open(filename, 'w') as f:
        for percentiles in params:
            for item in percentiles:
                f.write('{} '.format(item))
            f.write('\n')

def percentile_file2list(filename):
    '''

    :param filename:
    :return: params list saved in filename
    '''
    params = []
    with open(filename, 'r') as f:
        filelines = f.readlines()
        for fileline in filelines:
            temp = fileline.strip().split(' ')
            temp = [float(i) for i in temp]
            params.append(temp)
    return params

def histogram_list2file(filename, params):
    '''
    :param l:
     for each item in params
    :return:
    '''
    with open(filename, 'w') as f:
        for param in params:
            probs, means = param['probs'], param['means']
            for prob in probs:
                f.write('{} '.format(prob))
            f.write(',')
            for mean in means:
                f.write(' {}'.format(mean))
            f.write('\n')

def histogram_file2list(filename):
    '''

    :param filename:
    :return: params list saved in filename
    '''
    params = []
    with open(filename, 'r') as f:
        filelines = f.readlines()
        for fileline in filelines:
            probs, means = fileline.strip().split(',')
            probs, means = probs.strip().split(' '), means.strip().split(' ')
            #print(probs, means)
            probs, means = [float(x) for x in probs], [float(x) for x in means]
            params.append({'probs': probs, 'means': means})
    return params

def mle(x):
    '''
    maximum likelihood estimation
    :param x:
    :return mean and std of Gaussian:
    '''
    m = np.mean(x)
    return m, np.sqrt(np.dot(x-m, (x-m.T)) / x.shape[0])

def estimate_gaussian(X):
    '''
    estimate 1-d gaussian distribution for each column in X, i.e., each feature has two params, mean/std
    :param X:
    :return: [(mean_i, std_i) for i in features]
    '''
    n, d = X.shape
    params = []
    for i in range(d):
        feature_i = X[:, i]
        #drop nana
        feature_i_dropnan = feature_i[~np.isnan(feature_i)]
        #print(feature_i, feature_i_dropnan)
        #print(feature_i.shape, feature_i_dropnan.shape)
        assert ~np.isnan(feature_i_dropnan).all()
        params.append((mle(feature_i_dropnan)))
    return params


def estimate_percentile(X):
    '''
    estimate percentile distribution for each column in X, i.e., each feature has 10-bins histogram
    :param X:
    :return:
    '''
    n, d = X.shape
    params = []
    for i in range(d):
        feature_i = X[:, i]
        #drop nana
        feature_i_dropnan = feature_i[~np.isnan(feature_i)]
        #print(feature_i, feature_i_dropnan)
        #print(feature_i.shape, feature_i_dropnan.shape)
        assert ~np.isnan(feature_i_dropnan).all()

        percentiles = np.percentile(feature_i_dropnan, [5*j for j in range(21)])
        params.append(list(percentiles))
    return params

def estimate_histogram(X, num_bins=10):
    '''
    estimate 1-d histogram distribution for each column in X, i.e., each feature has 10-bins histogram
    :param X:
    :return:
    '''
    n, d = X.shape
    params = []
    for i in range(d):
        feature_i = X[:, i]
        #drop nana
        feature_i_dropnan = feature_i[~np.isnan(feature_i)]
        #print(feature_i, feature_i_dropnan)
        #print(feature_i.shape, feature_i_dropnan.shape)
        assert ~np.isnan(feature_i_dropnan).all()
        probs, intervals = np.histogram(feature_i_dropnan, bins=num_bins)
        probs = probs + 1
        probs = probs / np.sum(probs)
        probs, intervals = list(probs), list(intervals)
        means = []
        for k in range(len(intervals)-1):
            means.append((intervals[k]+intervals[k+1])/2.0)
        dict_i = {'probs': probs, 'means': means}
        params.append(dict_i)
    return params

def generate_missing_positions(X, frac):
    n, d = X.shape
    X_missing = deepcopy(X)
    missing_positions = random.sample(range(n*d), int(n*d*frac))
    for idx in missing_positions:
        i = idx//d
        j = idx%d
        X_missing[i][j] = np.nan
    return X_missing


def create_missing_dataset(cfg, idx):
    '''

    :param cfg:
    :return: manually create a missing dataset from a complete one
    '''
    load_data = 'Load_' + cfg.DATASETS.NAME
    x_train, y_train, x_val, y_val, x_test, y_test = eval(load_data)()
    savedir = os.path.join('./datasets/', cfg.DATASETS.NAME, 'missing_processed_speed', 'fraction-'+str(cfg.DATASETS.MISSING_FRAC), 'version-'+str(idx))
    if not os.path.exists(savedir):
        print('generate missing data')
        os.makedirs(savedir)
        x_train_missing = generate_missing_positions(x_train, cfg.DATASETS.MISSING_FRAC)
        x_test_missing = generate_missing_positions(x_test, cfg.DATASETS.MISSING_FRAC)
        np.save(os.path.join(savedir, 'x_train.npy'), x_train)
        np.save(os.path.join(savedir, 'x_test.npy'), x_test)
        np.save(os.path.join(savedir, 'x_train_missing.npy'), x_train_missing)
        np.save(os.path.join(savedir, 'x_test_missing.npy'), x_test_missing)
        np.save(os.path.join(savedir, 'y_train.npy'), y_train)
        np.save(os.path.join(savedir, 'y_test.npy'), y_test)
    else:
        print('load data file')
        x_train = np.load(os.path.join(savedir, 'x_train.npy'))
        x_test = np.load(os.path.join(savedir, 'x_test.npy'))
        x_train_missing = np.load(os.path.join(savedir, 'x_train_missing.npy'))
        x_test_missing = np.load(os.path.join(savedir, 'x_test_missing.npy'))
        y_train = np.load(os.path.join(savedir, 'y_train.npy'))
        y_test = np.load(os.path.join(savedir, 'y_test.npy'))

    return x_train, x_train_missing, y_train, x_test, x_test_missing, y_test

def load_original_missing_data(cfg, idx):
    '''

    :param cfg:
    :param idx: k-fold number
    :return: load originally missing dataset, and we use k-fold cross validation, idx denotes the idx-th fold-number
    '''
    load_data = 'Load_' + cfg.DATASETS.NAME
    x_train_missing, y_train, x_val_missing, y_val, x_test_missing, y_test = eval(load_data)()
    savedir = os.path.join('./datasets/', cfg.DATASETS.NAME, 'missing_processed_speed',
                           'fraction-' + str(cfg.DATASETS.MISSING_FRAC), 'version-' + str(idx))
    if not os.path.exists(savedir):
        print('generate missing data')
        os.makedirs(savedir)
        np.save(os.path.join(savedir, 'x_train_missing.npy'), x_train_missing)
        np.save(os.path.join(savedir, 'x_test_missing.npy'), x_test_missing)
        np.save(os.path.join(savedir, 'y_train.npy'), y_train)
        np.save(os.path.join(savedir, 'y_test.npy'), y_test)
    else:
        print('load data file')
        x_train_missing = np.load(os.path.join(savedir, 'x_train_missing.npy'))
        x_test_missing = np.load(os.path.join(savedir, 'x_test_missing.npy'))
        y_train = np.load(os.path.join(savedir, 'y_train.npy'))
        y_test = np.load(os.path.join(savedir, 'y_test.npy'))

    return x_train_missing, y_train, x_test_missing, y_test

def get_dataset(cfg, mi_idx=-1, cv_idx=-1, train=True):
    '''
    param cfg: dataset_name, cub200 or uci_dataset
    param idx: multiple imputation dataset number
    param cv_idx: k-fold cross_validation fold number
    return: pytorch train_loader, test_loader
    '''

    if cfg.DATASETS.CREATE_MISSING:
        x_train, x_train_missing, y_train, x_test, x_test_missing, y_test = create_missing_dataset(cfg, cv_idx)
    else:
        x_train_missing, y_train, x_test_missing, y_test = load_original_missing_data(cfg, cv_idx)
        x_train, x_test = x_train_missing, x_test_missing

    train_preprocessing_time = 0.
    test_preproscessing_time = 0.

    #preprocessing
    if cfg.PREPROCESSING.IMPUTER is not 'None':
        print('impute using {}'.format(cfg.PREPROCESSING.IMPUTER))
        imputer = cfg.PREPROCESSING.IMPUTER+'_imputer'
        #it means we use very slow multiple imputation
        #if cfg.PREPROCESSING.IMPUTER in ['knn', 'iterative', 'missforest', 'softimpute', 'matrix_factorization']:
        #we cache the multiple imputed datasets cause it is very slow
        savedir = os.path.join('./datasets/', cfg.DATASETS.NAME, 'missing_processed_speed',
                                'fraction-' + str(cfg.DATASETS.MISSING_FRAC), 'version-' + str(cv_idx), cfg.PREPROCESSING.IMPUTER)
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        savename = os.path.join(savedir, cfg.PREPROCESSING.IMPUTER+'_fold_{}'.format(cv_idx)+'_mi_{}'.format(mi_idx+1))
        trainfile = savename + '_train.npy'
        testfile = savename + '_test.npy'
        #if False:
        if os.path.exists(trainfile) and os.path.exists(testfile):
            x_train_missing = np.load(trainfile)
            x_test_missing = np.load(testfile)
        else:
            starttime = datetime.datetime.now()
            if cfg.PREPROCESSING.IMPUTER == 'AFASMC':
                matfile = os.path.join('./datasets', cfg.DATASETS.NAME, 'missing_processed_matlab',
                                       'fraction-'+str(cfg.DATASETS.MISSING_FRAC), 'AFASMC-fold-{}.mat'.format(cv_idx))
                data = scio.loadmat(matfile)
                x_train_missing = np.array(data['Xmc'])
            else:
                x_train_missing, x_test_missing, train_preprocessing_time, test_preproscessing_time = eval(imputer)(x_train_missing, x_test_missing)
            np.save(trainfile, x_train_missing)
            if cfg.PREPROCESSING.IMPUTER in ['softimpute', 'matrix_factorization', 'AFASMC']:
                #cannot handle test missing, avoid NaN error
                x_test_missing = x_test
            np.save(testfile, x_test_missing)
            endtime = datetime.datetime.now()
            seconds = (endtime-starttime).seconds
            minutes = seconds // 60
            second = seconds % 60
            print('{} imputation using time {} min {} seconds'.format(cfg.PREPROCESSING.IMPUTER, minutes, second))
        #else:
        #    x_train_missing, x_test_missing = eval(imputer)(x_train_missing, x_test_missing)
    else:
        print('no imputation')
    #print(x_train.shape, x_test.shape)
    #sklearn freiendly format
    if cfg.MODEL.META_ARCHITECTURE.startswith('sklearn'):
        return x_train, x_train_missing, y_train, x_test, x_test_missing, y_test
    #suppose each dimension of feature is a Gaussian distribution, estimate mean/std from training set, employ it to missing values
    elif cfg.MODEL.META_ARCHITECTURE.startswith('NRSHistogramSample'):
        maskfile = os.path.join(cfg.OUTPUT_DIR, 'sample_mask_fold_{}_mi_{}.npy'.format(cv_idx, mi_idx))
        paramsfile = os.path.join(cfg.OUTPUT_DIR, 'params_fold_{}_mi_{}.txt'.format(cv_idx, mi_idx))

        if train:
            mask = perm_mask(cfg.MODEL.DD, cfg.MODEL.DH, cfg.MODEL.DW, cfg.MODEL.N_MUL)
            np.save(maskfile, mask)
            params = estimate_histogram(x_train_missing, cfg.DATASETS.NUM_BINS)
            # save files
            histogram_list2file(paramsfile, params)
        else:
            mask = np.load(maskfile)
            params = histogram_file2list(paramsfile)

        if train:
            #TrainData = NRSHistogramSampleSpeedupLoader(x_train_missing, y_train, cfg, mask, params)
            TrainData = NRSHistogramSampleLoader(x_train_missing, y_train, cfg, mask, params)
        MissingTestData = NRSHistogramSampleLoader(x_test_missing, y_test, cfg, mask, params)
        FullTestData = NRSHistogramSampleLoader(x_test, y_test, cfg, mask, params)
        #MissingTestData = NRSHistogramSampleSpeedupLoader(x_test_missing, y_test, cfg, mask, params)
        #FullTestData = NRSHistogramSampleSpeedupLoader(x_test, y_test, cfg, mask, params)

    elif cfg.MODEL.META_ARCHITECTURE.startswith('MLPHistogramSample'):
        maskfile = os.path.join(cfg.OUTPUT_DIR, 'sample_mask_fold_{}_mi_{}.npy'.format(cv_idx, mi_idx))
        paramsfile = os.path.join(cfg.OUTPUT_DIR, 'params_fold_{}_mi_{}.txt'.format(cv_idx, mi_idx))

        if train:
            mask = perm_mask_for_fc(cfg.MODEL.DD, cfg.MODEL.DH, cfg.MODEL.DW, cfg.MODEL.N_MUL)
            np.save(maskfile, mask)

            params = estimate_histogram(x_train_missing)
            # save files
            histogram_list2file(paramsfile, params)
        else:
            mask = np.load(maskfile)
            params = histogram_file2list(paramsfile)

        if train:
            TrainData = MLPHistogramSampleLoader(x_train_missing, y_train, cfg, mask, params)
        MissingTestData = MLPHistogramSampleLoader(x_test_missing, y_test, cfg, mask, params)
        FullTestData = MLPHistogramSampleLoader(x_test, y_test, cfg, mask, params)
    elif cfg.MODEL.META_ARCHITECTURE.startswith('NRSPercentileSample'):
        maskfile = os.path.join(cfg.OUTPUT_DIR, 'sample_mask_fold_{}_mi_{}.npy'.format(cv_idx, mi_idx))
        paramsfile = os.path.join(cfg.OUTPUT_DIR, 'params_fold_{}_mi_{}.txt'.format(cv_idx, mi_idx))

        if train:
            mask = perm_mask(cfg.MODEL.DD, cfg.MODEL.DH, cfg.MODEL.DW, cfg.MODEL.N_MUL)
            np.save(maskfile, mask)
            params = estimate_percentile(x_train_missing)
            # save files
            percentile_list2file(paramsfile, params)
        else:
            mask = np.load(maskfile)
            params = percentile_file2list(paramsfile)

        if train:
            TrainData = NRSPercentileSampleLoader(x_train_missing, y_train, cfg, mask, params)
        MissingTestData = NRSPercentileSampleLoader(x_test_missing, y_test, cfg, mask, params, False)
        FullTestData = NRSPercentileSampleLoader(x_test, y_test, )
    #normal pytorch loader
    else:
        TrainData = NormLoader(x_train_missing, y_train)
        MissingTestData = NormLoader(x_test_missing, y_test)
        FullTestData = NormLoader(x_test, y_test)

    missing_test_loader = torch.utils.data.DataLoader(dataset=MissingTestData,
                                                      batch_size=cfg.SOLVER.TEST_PER_BATCH)
    full_test_loader = torch.utils.data.DataLoader(dataset=FullTestData,
                                                   batch_size=cfg.SOLVER.TEST_PER_BATCH)
    if train:
        train_loader = torch.utils.data.DataLoader(dataset=TrainData,
                                                    batch_size=cfg.SOLVER.TRAIN_PER_BATCH,
                                                    shuffle=True)
        return train_loader, missing_test_loader, full_test_loader, train_preprocessing_time, test_preproscessing_time
    else:
        return missing_test_loader, full_test_loader, train_preprocessing_time, test_preproscessing_time

