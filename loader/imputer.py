import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from missingpy import MissForest
#from fancyimpute import SoftImpute, MatrixFactorization
#from impyute import em
import time

def em_imputer(x_train_missing, x_test_missing):
    train_shape = x_train_missing.shape[0]
    x = np.concatenate((x_train_missing, x_test_missing), axis=0)
    x = em(x)
    x_train_processed = x[:train_shape]
    x_test_processed = x[train_shape:]
    print(x_train_processed.shape, x_test_processed.shape)
    return x_train_processed, x_test_processed


def softimpute_imputer(x_train_missing, x_test_missing):
    imp = SoftImpute()
    train_shape = x_train_missing.shape[0]
    print(x_train_missing.shape, x_test_missing.shape)

    '''
    x = np.concatenate((x_train_missing, x_test_missing), axis=0)
    x = imp.fit_transform(x)
    x_train_processed = x[:train_shape]
    x_test_processed = x[train_shape:]
    '''

    x_train_processed = imp.fit_transform(x_train_missing)
    x_test_processed = x_test_missing


    print(x_train_processed.shape, x_test_processed.shape)
    return x_train_processed, x_test_processed

def matrix_factorization_imputer(x_train_missing, x_test_missing):
    imp = MatrixFactorization()
    train_shape = x_train_missing.shape[0]
    print(x_train_missing.shape, x_test_missing.shape)

    '''
    x = np.concatenate((x_train_missing, x_test_missing), axis=0)
    x = imp.fit_transform(x)
    x_train_processed = x[:train_shape]
    x_test_processed = x[train_shape:]
    '''

    x_train_processed = imp.fit_transform(x_train_missing)
    x_test_processed = x_test_missing

    print(x_train_processed.shape, x_test_processed.shape)
    return x_train_processed, x_test_processed


def deletion_imputer(x_train_missing, x_test_missing):
    print(x_train_missing.shape)
    x_train_deletion = x_train_missing[~np.isnan(x_train_missing).any(axis=1)]
    print('before deletion x_train is {}, after deletion x_train is {}'.format(x_train_missing.shape, x_train_deletion.shape))
    return x_train_missing, x_test_missing

def frequent_imputer(x_train_missing, x_test_missing):
    imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    imp.fit(x_train_missing)
    x_train_processed = imp.transform(x_train_missing)
    x_test_processed = imp.transform(x_test_missing)

    return x_train_processed, x_test_processed

def mean_imputer(x_train_missing, x_test_missing):
    starttime = time.time()

    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    # imp = IterativeImputer(max_iter=20)
    imp.fit(x_train_missing)
    x_train_processed = imp.transform(x_train_missing)
    traintime = time.time()

    x_test_processed = imp.transform(x_test_missing)
    testtime = time.time()
    train_seconds = (traintime - starttime)
    train_minutes = train_seconds // 60
    train_second = train_seconds % 60
    print('mice training imputation using time {} min {} seconds'.format(train_minutes, train_second))

    test_seconds = (testtime - traintime)
    test_minutes = test_seconds // 60
    test_second = test_seconds % 60
    print('mice inference imputation using time {} min {} seconds'.format(test_minutes, test_second))

    return x_train_processed, x_test_processed, train_seconds, test_seconds

def median_imputer(x_train_missing, x_test_missing):
    starttime = time.time()

    imp = SimpleImputer(missing_values=np.nan, strategy='median')
    # imp = IterativeImputer(max_iter=20)
    imp.fit(x_train_missing)
    x_train_processed = imp.transform(x_train_missing)
    traintime = time.time()

    x_test_processed = imp.transform(x_test_missing)
    testtime = time.time()
    train_seconds = (traintime - starttime)
    train_minutes = train_seconds // 60
    train_second = train_seconds % 60
    print('mice training imputation using time {} min {} seconds'.format(train_minutes, train_second))

    test_seconds = (testtime - traintime)
    test_minutes = test_seconds // 60
    test_second = test_seconds % 60
    print('mice inference imputation using time {} min {} seconds'.format(test_minutes, test_second))

    return x_train_processed, x_test_processed

def iterative_imputer(x_train_missing, x_test_missing):
    starttime = time.time()

    imp = IterativeImputer(max_iter=20, random_state=42)
    #imp = IterativeImputer(max_iter=20)
    imp.fit(x_train_missing)
    x_train_processed = imp.transform(x_train_missing)
    traintime = time.time()

    x_test_processed = imp.transform(x_test_missing)
    testtime = time.time()
    train_seconds = (traintime - starttime)
    train_minutes = train_seconds // 60
    train_second = train_seconds % 60
    print('mice training imputation using time {} min {} seconds'.format(train_minutes, train_second))

    test_seconds = (testtime - traintime)
    test_minutes = test_seconds // 60
    test_second = test_seconds % 60
    print('mice inference imputation using time {} min {} seconds'.format(test_minutes, test_second))

    return x_train_processed, x_test_processed, train_seconds, test_seconds

def knn_imputer(x_train_missing, x_test_missing, k=2):

    starttime = time.time()

    imp = KNNImputer(n_neighbors=k, weights='uniform')
    # imp = IterativeImputer(max_iter=20)
    imp.fit(x_train_missing)
    x_train_processed = imp.transform(x_train_missing)
    traintime = time.time()

    x_test_processed = imp.transform(x_test_missing)
    testtime = time.time()
    train_seconds = (traintime - starttime)
    train_minutes = train_seconds // 60
    train_second = train_seconds % 60
    print('knn training imputation using time {} min {} seconds'.format(train_minutes, train_second))

    test_seconds = (testtime - traintime)
    test_minutes = test_seconds // 60
    test_second = test_seconds % 60
    print('knn inference imputation using time {} min {} seconds'.format(test_minutes, test_second))

    return x_train_processed, x_test_processed, train_seconds, test_seconds

def missforest_imputer(x_train_missing, x_test_missing):
    imp = MissForest()
    imp.fit(x_train_missing)

    x_train_processed = imp.transform(x_train_missing)
    x_test_processed = imp.transform(x_test_missing)

    print('missforest finish')

    return x_train_processed, x_test_processed
