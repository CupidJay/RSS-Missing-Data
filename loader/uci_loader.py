import os
import random
import numpy as np
import logging
from sklearn.datasets import load_svmlight_files, load_svmlight_file
import torch
import torchvision
import torch.utils.data as data
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

def findSortedPosition(theList, target):
    low = 0
    high = len(theList) - 1
    while low <= high:
        mid = (high + low) // 2
        if theList[mid] == target:
            return mid
        elif target < theList[mid]:
            high = mid -1
        else:
            low = mid + 1
    return low

def perm_mask(dd, dH, dW, nMul):
    #generating a mask for permutation into dH x dW x (dd*nMul) tensor
    m = np.random.permutation(dd)
    #m = np.arange(dd)
    for i in range(1, dH*dW*nMul):
        m = np.concatenate((m, np.random.permutation(dd)))
    return m

def perm_mask_for_fc(dd, dH, dW, nMul):
    #generating a mask for permutation into dH x dW x (dd*nMul) tensor
    m = np.arange(dd)
    for i in range(1, nMul):
        m = np.concatenate((m, np.arange(dd)))
    return m

class NormLoader(data.Dataset):
    def __init__(self, matrix, labels):
        self.matrix = matrix
        self.labels = torch.tensor(labels,dtype=torch.long)

    def __getitem__(self, index):
        item_data = self.matrix[index].astype(np.float32)
        return item_data, self.labels[index]

    def __len__(self):
        return len(self.matrix)

class NRSGuassianSampleLoader(data.Dataset):
    def __init__(self, matrix, labels, cfg, mask, params, train=True):
        self.matrix = matrix
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.dd = cfg.MODEL.DD
        self.dH = cfg.MODEL.DH
        self.dW = cfg.MODEL.DW
        self.nMul = cfg.MODEL.N_MUL
        self.mask = mask
        #self.mask = perm_mask(self.dd, self.dH, self.dW, self.nMul)
        self.params = params

        #np.random.seed()

        self.sample_matrix = np.zeros((self.matrix.shape[0], self.dd*self.nMul*self.dH*self.dW))
        for i in range(self.matrix.shape[0]):
            item_data = self.matrix[i]
            item_data = item_data[self.mask]
            for j in range(len(item_data)):
                if np.isnan(item_data[j]):
                    column = self.mask[j]
                    #if train:
                    item_data[j] = np.random.normal(self.params[column][0], self.params[column][1])
                    #else:
                    #    item_data[j] = self.params[column][0]
            assert ~np.isnan(item_data).all()
            self.sample_matrix[i] = item_data

        print('finish')

    def __getitem__(self, index):
        item_data = self.sample_matrix[index]
        item_data = np.reshape(item_data, (self.dd*self.nMul, self.dH, self.dW))
        item_data = np.transpose(item_data, (1, 2, 0)).astype(np.float32)
        #item_data = np.reshape(item_data, (self.dH, self.dW, self.dd*self.nMul))
        return torchvision.transforms.ToTensor()(item_data), self.labels[index]

    def __len__(self):
        return len(self.matrix)

class NRSPercentileSampleLoader(data.Dataset):
    def __init__(self, matrix, labels, cfg, mask, params, train=True):
        self.matrix = matrix
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.dd = cfg.MODEL.DD
        self.dH = cfg.MODEL.DH
        self.dW = cfg.MODEL.DW
        self.nMul = cfg.MODEL.N_MUL
        self.mask = mask
        #self.mask = perm_mask(self.dd, self.dH, self.dW, self.nMul)
        self.params = params

        #np.random.seed()
        '''
        self.sample_matrix = np.zeros((self.matrix.shape[0], self.dd*self.nMul*self.dH*self.dW))
        for i in range(self.matrix.shape[0]):
            item_data = self.matrix[i]
            item_data = item_data[self.mask]
            for j in range(len(item_data)):
                if np.isnan(item_data[j]):
                    column = self.mask[j]
                    #if train:
                    percentiles = self.params[column]
                    #print(len(percentiles))
                    randomint = np.random.randint(0, 20)
                    gen_value = (percentiles[randomint]+percentiles[randomint+1])/2.0
                    item_data[j] = gen_value

                    #else:
                    #    item_data[j] = self.params[column][0]
            assert ~np.isnan(item_data).all()
            self.sample_matrix[i] = item_data
            '''
        print('finish')

    def __getitem__(self, index):
        # '''
        item_data = self.matrix[index]
        # permute feature vector and concatenate
        # return item_data, self.labels[index]

        item_data = item_data[self.mask]
        for i in range(len(item_data)):
            if np.isnan(item_data[i]):
                column = self.mask[i]
                percentiles = self.params[column]
                # print(len(percentiles))
                randomint = np.random.randint(0, 20)
                gen_value = (percentiles[randomint] + percentiles[randomint + 1]) / 2.0
                item_data[i] = gen_value
        assert ~np.isnan(item_data).all()
        # '''

        #item_data = self.sample_matrix[index]
        item_data = np.reshape(item_data, (self.dd*self.nMul, self.dH, self.dW))
        item_data = np.transpose(item_data, (1, 2, 0)).astype(np.float32)
        #item_data = np.reshape(item_data, (self.dH, self.dW, self.dd*self.nMul))
        return torchvision.transforms.ToTensor()(item_data), self.labels[index]

    def __len__(self):
        return len(self.matrix)
'''
class MLPHistogramSampleLoader(data.Dataset):
    def __init__(self, matrix, labels, cfg, params):
        self.matrix = matrix
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.params = params

        self.fixedtrain = cfg.MODEL.SAMPLE.FIXEDTRAIN
        # np.random.seed()
        if self.fixedtrain:
            for i in range(self.matrix.shape[0]):
                for j in range(self.matrix.shape[1]):
                    if np.isnan(self.matrix[i][j]):
                        column = j
                        probs, means = self.params[column]['probs'], self.params[column]['means']
                        gen_prob = np.random.rand()
                        gen_position = findSortedPosition(probs, gen_prob)
                        # for k in range(len(probs)):
                        #    if gen_prob<=probs[k]:
                        #        gen_position = k
                        #        break
                        gen_value = means[gen_position]
                        # print('probs {}, means {}'.format(probs, means))
                        # print('gen_porb {}. gen_position {}, gen_value {}'.format(gen_prob, gen_position, gen_value))
                        # print('*'*10)
                        self.matrix[i][j] = gen_value

            assert ~np.isnan(self.matrix).all()

    def __getitem__(self, index):
        item_data = self.matrix[index].astype(np.float32)
        for j in range(self.matrix.shape[1]):
            if np.isnan(item_data[j]):
                column = j
                probs, means = self.params[column]['probs'], self.params[column]['means']
                gen_prob = np.random.rand()
                gen_position = findSortedPosition(probs, gen_prob)

                gen_value = means[gen_position]
                item_data[j] = gen_value
        assert ~np.isnan(item_data).all()
        return item_data, self.labels[index]

    def __len__(self):
        return len(self.matrix)
'''


class MLPHistogramSampleLoader(data.Dataset):
    def __init__(self, matrix, labels, cfg, mask, params):
        self.matrix = matrix
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.dd = cfg.MODEL.DD
        self.dH = cfg.MODEL.DH
        self.dW = cfg.MODEL.DW
        self.nMul = cfg.MODEL.N_MUL
        self.mask = mask
        #self.mask = perm_mask(self.dd, self.dH, self.dW, self.nMul)
        self.params = params

        self.fixedtrain = cfg.MODEL.SAMPLE.FIXEDTRAIN
        #np.random.seed()
        if self.fixedtrain:
            self.sample_matrix = np.zeros((self.matrix.shape[0], self.dd*self.nMul*self.dH*self.dW))
            for i in range(self.matrix.shape[0]):
                item_data = self.matrix[i]
                item_data = item_data[self.mask]
                for j in range(len(item_data)):
                    if np.isnan(item_data[j]):
                        column = self.mask[j]
                        #if train:
                        probs, means = self.params[column]['probs'], self.params[column]['means']
                        gen_prob = np.random.rand()
                        gen_position = findSortedPosition(probs, gen_prob)
                        #for k in range(len(probs)):
                        #    if gen_prob<=probs[k]:
                        #        gen_position = k
                        #        break
                        gen_value = means[gen_position]
                        #print('probs {}, means {}'.format(probs, means))
                        #print('gen_porb {}. gen_position {}, gen_value {}'.format(gen_prob, gen_position, gen_value))
                        #print('*'*10)

                        item_data[j] = gen_value
                        #else:
                        #    item_data[j] = self.params[column][0]
                assert ~np.isnan(item_data).all()
                self.sample_matrix[i] = item_data
        #'''
        print('finish')

    def __getitem__(self, index):
        if not self.fixedtrain:
            #'''
            item_data = self.matrix[index]
            #permute feature vector and concatenate
            #return item_data, self.labels[index]
            item_data = item_data[self.mask]
            for i in range(len(item_data)):
                if np.isnan(item_data[i]):
                    column = self.mask[i]
                    #item_data[i] = np.random.normal(self.params[column][0], self.params[column][1])
                    # if train:
                    probs, means = self.params[column]['probs'], self.params[column]['means']
                    gen_prob = np.random.rand()
                    gen_position = findSortedPosition(probs, gen_prob)
                    # for k in range(len(probs)):
                    #    if gen_prob<=probs[k]:
                    #        gen_position = k
                    #        break
                    gen_value = means[gen_position]
                    # print('probs {}, means {}'.format(probs, means))
                    # print('gen_porb {}. gen_position {}, gen_value {}'.format(gen_prob, gen_position, gen_value))
                    # print('*'*10)
                    item_data[i] = gen_value
            assert ~np.isnan(item_data).all()
            #'''
        else:
            item_data = self.sample_matrix[index]
        item_data = np.reshape(item_data, (self.dd*self.nMul, -1)).astype(np.float32)
        #item_data = np.reshape(item_data, (self.dH, self.dW, self.dd*self.nMul))
        return item_data, self.labels[index]

    def __len__(self):
        return len(self.matrix)


class NRSHistogramSampleLoader(data.Dataset):
    def __init__(self, matrix, labels, cfg, mask, params):
        self.matrix = matrix
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.dd = cfg.MODEL.DD
        self.dH = cfg.MODEL.DH
        self.dW = cfg.MODEL.DW
        self.nMul = cfg.MODEL.N_MUL
        self.mask = mask
        #self.mask = perm_mask(self.dd, self.dH, self.dW, self.nMul)
        self.params = params

        self.fixedtrain = cfg.MODEL.SAMPLE.FIXEDTRAIN
        
        self.missing_dict = {}

        for i in range(self.matrix.shape[0]):
            item_data = self.matrix[i]
            for j in range(len(item_data)):
                if np.isnan(item_data[j]):
                    if i not in self.missing_dict.keys():
                        self.missing_dict[i] = []
                    self.missing_dict[i].append(j)

        #np.random.seed()
        if self.fixedtrain:
            self.sample_matrix = np.zeros((self.matrix.shape[0], self.dd*self.nMul*self.dH*self.dW))
            for i in range(self.matrix.shape[0]):
                item_data = self.matrix[i]
                item_data = item_data[self.mask]
                for j in range(len(item_data)):
                    if np.isnan(item_data[j]):
                        column = self.mask[j]
                        #if train:
                        probs, means = self.params[column]['probs'], self.params[column]['means']
                        #        break
                        gen_value = np.random.choice(means, 1, p=probs)
                        #print('probs {}, means {}'.format(probs, means))
                        #print('gen_porb {}. gen_position {}, gen_value {}'.format(gen_prob, gen_position, gen_value))
                        #print('*'*10)

                        item_data[j] = gen_value
                        #else:
                        #    item_data[j] = self.params[column][0]
                assert ~np.isnan(item_data).all()
                self.sample_matrix[i] = item_data
        #'''
        print('finish')

    def __getitem__(self, index):
        if not self.fixedtrain:
            #'''
            item_data = self.matrix[index]
            #permute feature vector and concatenate
            #return item_data, self.labels[index]
            item_data = item_data[self.mask]
            gen_values = {}
            for i in range(len(item_data)):
                if np.isnan(item_data[i]):
                    column = self.mask[i]
                    #item_data[i] = np.random.normal(self.params[column][0], self.params[column][1])
                    # if train:
                    probs, means = self.params[column]['probs'], self.params[column]['means']
                    if i<self.dd:
                        gen_values[column] = np.random.choice(means, self.dH*self.dW*self.nMul, p=probs)
                    # print('probs {}, means {}'.format(probs, means))
                    # print('gen_porb {}. gen_position {}, gen_value {}'.format(gen_prob, gen_position, gen_value))
                    # print('*'*10)
                    item_data[i] = gen_values[column][i//self.dd]
            #assert ~np.isnan(item_data).all()
            #'''
        else:
            item_data = self.sample_matrix[index]
        item_data = np.reshape(item_data, (self.dd*self.nMul, self.dH, self.dW))
        item_data = np.transpose(item_data, (1, 2, 0)).astype(np.float32)
        #item_data = np.reshape(item_data, (self.dH, self.dW, self.dd*self.nMul))
        return torchvision.transforms.ToTensor()(item_data), self.labels[index]

    def __len__(self):
        return len(self.matrix)


from multiprocessing import Pool
class NRSHistogramSampleSpeedupLoader(data.Dataset):
    def __init__(self, matrix, labels, cfg, mask, params):
        self.matrix = matrix
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.dd = cfg.MODEL.DD
        self.dH = cfg.MODEL.DH
        self.dW = cfg.MODEL.DW
        self.nMul = cfg.MODEL.N_MUL
        self.mask = mask
        # self.mask = perm_mask(self.dd, self.dH, self.dW, self.nMul)
        self.params = params

        self.fixedtrain = cfg.MODEL.SAMPLE.FIXEDTRAIN

        self.missing_dict = {}

        for i in range(self.matrix.shape[0]):
            item_data = self.matrix[i]
            for j in range(len(item_data)):
                if np.isnan(item_data[j]):
                    if i not in self.missing_dict.keys():
                        self.missing_dict[i] = []
                    self.missing_dict[i].append(j)

        # np.random.seed()
        if self.fixedtrain:
            self.sample_matrix = np.zeros((self.matrix.shape[0], self.dd * self.nMul * self.dH * self.dW))
            for i in range(self.matrix.shape[0]):
                item_data = self.matrix[i]
                item_data = item_data[self.mask]
                for j in range(len(item_data)):
                    if np.isnan(item_data[j]):
                        column = self.mask[j]
                        # if train:
                        probs, means = self.params[column]['probs'], self.params[column]['means']
                        #        break
                        gen_value = np.random.choice(means, 1, p=probs)
                        # print('probs {}, means {}'.format(probs, means))
                        # print('gen_porb {}. gen_position {}, gen_value {}'.format(gen_prob, gen_position, gen_value))
                        # print('*'*10)

                        item_data[j] = gen_value
                        # else:
                        #    item_data[j] = self.params[column][0]
                assert ~np.isnan(item_data).all()
                self.sample_matrix[i] = item_data
        # '''
        print('finish')

    def RunProcess(self, i):
        if np.isnan(self.item_data[i]):
            column = self.mask[i]
            # item_data[i] = np.random.normal(self.params[column][0], self.params[column][1])
            # if train:
            probs, means = self.params[column]['probs'], self.params[column]['means']
            gen_prob = np.random.rand()
            gen_position = findSortedPosition(probs, gen_prob)
            gen_value = means[gen_position]

            self.item_data[i] = gen_value

    def __getitem__(self, index):
        if not self.fixedtrain:
            # '''
            item_data = self.matrix[index]
            # permute feature vector and concatenate
            # return item_data, self.labels[index]
            self.item_data = item_data[self.mask]
            pool = Pool(3)
            pool.map(self.RunProcess, list(range(len(self.item_data))))
            pool.close()
            pool.join()

            # assert ~np.isnan(item_data).all()
            # '''
        else:
            item_data = self.sample_matrix[index]
        item_data = np.reshape(item_data, (self.dd * self.nMul, self.dH, self.dW))
        item_data = np.transpose(item_data, (1, 2, 0)).astype(np.float32)
        # item_data = np.reshape(item_data, (self.dH, self.dW, self.dd*self.nMul))
        return torchvision.transforms.ToTensor()(item_data), self.labels[index]

    def __len__(self):
        return len(self.matrix)


def Load_gisette():
    logger = logging.getLogger(__name__)
    logger.info("start loading gisette file")
    x_train, y_train, x_test, y_test = load_svmlight_files(("./datasets/gisette/gisette_scale", "./datasets/gisette/gisette_scale.t"))
    x_train = x_train.toarray().astype(np.float32)
    x_test = x_test.toarray().astype(np.float32)
    y_train = (y_train + 1) // 2
    y_test = (y_test + 1) // 2

    x_val, y_val = x_test, y_test
    sc = StandardScaler()
    sc.fit(x_train)
    x_train, x_val, x_test = sc.transform(x_train), sc.transform(x_val), sc.transform(x_test)

    return x_train, y_train, x_val, y_val, x_test, y_test


def Load_dna():
    logger = logging.getLogger(__name__)
    logger.info("start loading DNA file")
    # Download these 2 files from the LIBSVM dataset page
    x_train, y_train, x_val, y_val, x_test, y_test = load_svmlight_files(
        ("./datasets/dna/dna.scale.tr",
         "./datasets/dna/dna.scale.val",
         "./datasets/dna/dna.scale.t"))
    # for simplicity, convert sparse matrices to dense ones. We need to handle sparse input later.
    x_train = x_train.toarray().astype(np.float32)
    x_val = x_val.toarray().astype(np.float32)
    x_test = x_test.toarray().astype(np.float32)
    y_train = y_train - 1
    y_val = y_val-1
    y_test = y_test - 1
    logger.info("x_train size is {}, x_val size is {}, x_test size is {}".format(x_train.shape, x_val.shape, x_test.shape))
    return x_train, y_train, x_val, y_val, x_test, y_test

def Load_phishing():
    logger = logging.getLogger(__name__)
    logger.info("start loading phishing file")
    # Download these 2 files from the LIBSVM dataset page
    x, y = load_svmlight_file(
        ("./datasets/phishing/phishing"))
    # for simplicity, convert sparse matrices to dense ones. We need to handle sparse input later.
    x = x.toarray().astype(np.float32)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    x_val, y_val = x_test, y_test

    sc = StandardScaler()
    sc.fit(x_train)
    x_train, x_val, x_test = sc.transform(x_train), sc.transform(x_val), sc.transform(x_test)

    print("x_train size is {}, x_val size is {}, x_test size is {}".format(x_train.shape, x_val.shape, x_test.shape))
    return x_train, y_train, x_val, y_val, x_test, y_test

def Load_protein():
    logger = logging.getLogger(__name__)
    logger.info("start loading protein file")
    # Download these 2 files from the LIBSVM dataset page
    x_train, y_train, x_val, y_val, x_test, y_test = load_svmlight_files(
        ("./datasets/protein/svm-protein.tr",
         "./datasets/protein/svm-protein.val",
         "./datasets/protein/svm-protein.t"))
    # for simplicity, convert sparse matrices to dense ones. We need to handle sparse input later.
    x_train = x_train.toarray().astype(np.float32)
    x_val = x_val.toarray().astype(np.float32)
    x_test = x_test.toarray().astype(np.float32)

    sc = StandardScaler()
    sc.fit(x_train)
    x_train, x_val, x_test = sc.transform(x_train), sc.transform(x_val), sc.transform(x_test)

    logger.info("x_train size is {}, x_val size is {}, x_test size is {}".format(x_train.shape, x_val.shape, x_test.shape))
    return x_train, y_train, x_val, y_val, x_test, y_test


def Load_satimage():
    logger = logging.getLogger(__name__)
    logger.info("start loadding satimage file")
    # Download these 2 files from the LIBSVM dataset page
    x_train, y_train, x_test, y_test = load_svmlight_files(
        ("./datasets/satimage/satimage.scale", "./datasets/satimage/satimage.scale.t"))
    # for simplicity, convert sparse matrices to dense ones. We need to handle sparse input later.
    x_train = x_train.toarray().astype(np.float32)
    x_test = x_test.toarray().astype(np.float32)
    y_train = y_train - 1
    y_test = y_test - 1

    x_val, y_val = x_test, y_test

    sc = StandardScaler()
    sc.fit(x_train)
    x_train, x_val, x_test = sc.transform(x_train), sc.transform(x_val), sc.transform(x_test)

    logger.info("x_train size is {}, x_test size is {}".format(x_train.shape, x_test.shape))
    return x_train, y_train, x_val, y_val, x_test, y_test

def Load_vehicle():
    logger = logging.getLogger(__name__)
    logger.info("start loading vehicle file")
    # Download these 2 files from the LIBSVM dataset page
    x, y = load_svmlight_file(
        ("./datasets/vehicle/vehicle.scale"))
    # for simplicity, convert sparse matrices to dense ones. We need to handle sparse input later.
    x = x.toarray().astype(np.float32)
    y = y-1

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    x_val, y_val = x_test, y_test

    print("x_train size is {}, x_val size is {}, x_test size is {}".format(x_train.shape, x_val.shape, x_test.shape))
    return x_train, y_train, x_val, y_val, x_test, y_test

def Load_pendigits():
    logger = logging.getLogger(__name__)
    logger.info("start loading pendigits file")
    # Download these 2 files from the LIBSVM dataset page
    x_train, y_train, x_test, y_test = load_svmlight_files(
        ("./datasets/pendigits/pendigits",
         "./datasets/pendigits/pendigits.t"))
    # for simplicity, convert sparse matrices to dense ones. We need to handle sparse input later.
    x_train = x_train.toarray().astype(np.float32)
    x_test = x_test.toarray().astype(np.float32)

    x = np.concatenate((x_train, x_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    x_val, y_val = x_test, y_test

    logger.info("x_train size is {}, x_val size is {}, x_test size is {}".format(x_train.shape, x_val.shape, x_test.shape))
    return x_train, y_train, x_val, y_val, x_test, y_test


def Load_german_numer():
    logger = logging.getLogger(__name__)
    logger.info("start loading german.numer file")
    # Download these 2 files from the LIBSVM dataset page
    x, y = load_svmlight_file(
        ("./datasets/german.number/german.numer_scale"))
    # for simplicity, convert sparse matrices to dense ones. We need to handle sparse input later.
    x = x.toarray().astype(np.float32)
    y = (y + 1) // 2

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    x_val, y_val = x_test, y_test

    print("x_train size is {}, x_val size is {}, x_test size is {}".format(x_train.shape, x_val.shape, x_test.shape))
    return x_train, y_train, x_val, y_val, x_test, y_test

def Load_breast_cancer():
    logger = logging.getLogger(__name__)
    logger.info("start loading breast_cancer file")
    # Download these 2 files from the LIBSVM dataset page
    x, y = load_svmlight_file(
        ("./datasets/breast_cancer/breast-cancer_scale"))
    # for simplicity, convert sparse matrices to dense ones. We need to handle sparse input later.
    x = x.toarray().astype(np.float32)
    y = y // 2 - 1

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    x_val, y_val = x_test, y_test

    print("x_train size is {}, x_val size is {}, x_test size is {}".format(x_train.shape, x_val.shape, x_test.shape))
    return x_train, y_train, x_val, y_val, x_test, y_test

def Load_sonar():
    logger = logging.getLogger(__name__)
    logger.info("start loading sonar file")
    # Download these 2 files from the LIBSVM dataset page
    x, y = load_svmlight_file(
        ("./datasets/sonar/sonar_scale"))
    # for simplicity, convert sparse matrices to dense ones. We need to handle sparse input later.
    x = x.toarray().astype(np.float32)
    y = (y + 1) // 2

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    x_val, y_val = x_test, y_test

    print("x_train size is {}, x_val size is {}, x_test size is {}".format(x_train.shape, x_val.shape, x_test.shape))
    return x_train, y_train, x_val, y_val, x_test, y_test


def Load_spambase():
    logger = logging.getLogger(__name__)
    logger.info("start loading spambase data")
    with open('./datasets/spambase/spambase.data') as f:
        filelines = f.readlines()
        X_data = []
        y_data = []
        for fileline in filelines:
            fileline = fileline.strip()
            row_data = fileline.split(',')
            if len(row_data) < 3:
                continue
            #print(row_data)

            X_data.append(row_data[:-1])
            y_data.append(row_data[-1])
    X_data = np.array(X_data).astype(np.float32)

    y_data = np.array(y_data).astype(np.float32)
    print(X_data.shape)
    print(y_data.shape)
    print(np.max(y_data))

    x_train, x_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3)
    print(x_train.shape, y_train.shape)
    x_val, y_val = x_test, y_test

    return x_train, y_train, x_val, y_val, x_test, y_test


def Load_mushroom():
    logger = logging.getLogger(__name__)
    logger.info("start loading mushroom data")
    with open('./datasets/mushroom/agaricus-lepiota.data') as f:
        filelines = f.readlines()
        X_data = []
        y_data = []
        for fileline in filelines:
            fileline = fileline.strip()
            row_data = fileline.split(',')
            if len(row_data) < 3:
                continue
            #print(row_data)
            #remove row_data[11], which has missing values
            del row_data[11]

            X_data.append(row_data[1:])
            y_data.append(row_data[0])
    X_data = np.array(X_data)
    # print(X_numerical)
    # print(X_data)
    X_onehot = X_data
    # print(X_onehot)
    onehot = OneHotEncoder()
    X_onehot = onehot.fit_transform(X_onehot).toarray()
    X_onehot = X_onehot.astype(np.float32)
    # print(X_onehot.shape)
    # print(X_onehot)

    X_data = X_onehot


    label = LabelEncoder()
    y_data = label.fit_transform(y_data)
    y_data = np.array(y_data)
    print(X_data.shape)
    print(y_data.shape)
    print(X_data)
    print(y_data)
    print(np.max(y_data))

    x_train, x_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3)
    print(x_train.shape, y_train.shape)
    x_val, y_val = x_test, y_test

    return x_train, y_train, x_val, y_val, x_test, y_test


def Load_nursery():
    logger = logging.getLogger(__name__)
    logger.info("start loading connect4 data")
    with open('./datasets/nursery/nursery.data') as f:
        filelines = f.readlines()
        X_data = []
        y_data = []
        for fileline in filelines:
            fileline = fileline.strip()
            row_data = fileline.split(',')
            if len(row_data) < 3:
                continue
            #print(row_data)
            if row_data[3]=='more':
                row_data[3] = 4
            if row_data[-1]=='recommend':
                row_data[-1] = 'very_recom'

            X_data.append(row_data[:-1])
            y_data.append(row_data[-1])
    X_data = np.array(X_data)
    X_numerical = X_data[:, [3]]
    X_numerical = X_numerical.astype(np.float32)
    # print(X_numerical)
    # print(X_data)
    X_onehot = X_data[:, [0,1,2,4,5,6]]
    # print(X_onehot)
    onehot = OneHotEncoder()
    X_onehot = onehot.fit_transform(X_onehot).toarray()
    X_onehot = X_onehot.astype(np.float32)
    # print(X_onehot.shape)
    # print(X_onehot)

    X_data = np.concatenate((X_onehot, X_numerical), axis=1)


    label = LabelEncoder()
    y_data = label.fit_transform(y_data)
    y_data = np.array(y_data)
    print(X_data.shape)
    print(y_data.shape)
    print(X_data)
    print(y_data)
    print(np.max(y_data))

    x_train, x_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3)
    print(x_train.shape, y_train.shape)
    x_val, y_val = x_test, y_test

    return x_train, y_train, x_val, y_val, x_test, y_test

def Load_yeast():
    logger = logging.getLogger(__name__)
    logger.info("start loading yeast data")
    with open('./datasets/yeast/yeast.data') as f:
        filelines = f.readlines()
        X_data = []
        y_data = []
        for fileline in filelines:
            fileline = fileline.strip()
            row_data = fileline.split()
            #print(row_data)
            del row_data[0]
            X_data.append(row_data[:-1])
            y_data.append(row_data[-1])

    X_data = np.array(X_data).astype(np.float32)

    label = LabelEncoder()
    y_data = label.fit_transform(y_data)
    y_data = np.array(y_data)
    #print(X_data)
    #print(y_data)
    print(X_data.shape)
    print(y_data.shape)
    print(np.max(y_data))

    x_train, x_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3)
    print(x_train.shape, y_train.shape)
    x_val, y_val = x_test, y_test

    return x_train, y_train, x_val, y_val, x_test, y_test


def Load_connect4():
    logger = logging.getLogger(__name__)
    logger.info("start loading connect4 data")
    with open('./datasets/connect4/connect-4.data') as f:
        filelines = f.readlines()
        X_data = []
        y_data = []
        for fileline in filelines:
            fileline = fileline.strip()
            row_data = fileline.split(',')

            X_data.append(row_data[:-1])
            y_data.append(row_data[-1])
    X_data = np.array(X_data)
    #print(X_numerical)
    #print(X_data)
    #print(X_onehot)
    onehot = OneHotEncoder()
    X_onehot = onehot.fit_transform(X_data).toarray()
    X_onehot = X_onehot.astype(np.float32)
    #print(X_onehot.shape)
    #print(X_onehot)

    X_data = X_onehot


    label = LabelEncoder()
    y_data = label.fit_transform(y_data)
    y_data = np.array(y_data)
    print(X_data.shape)
    print(y_data.shape)
    print(X_data)
    print(y_data)
    print(np.max(y_data))

    x_train, x_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3)
    print(x_train.shape, y_train.shape)
    x_val, y_val = x_test, y_test

    return x_train, y_train, x_val, y_val, x_test, y_test

def Load_lymphography():
    logger = logging.getLogger(__name__)
    logger.info("start loading lymphography data")
    with open('./datasets/lymphography/lymphography.data') as f:
        filelines = f.readlines()
        X_data = []
        y_data = []
        for fileline in filelines:
            fileline = fileline.strip()
            row_data = fileline.split(',')

            X_data.append(row_data[:-1])
            y_data.append(float(row_data[-1]))
    X_data = np.array(X_data).astype(np.float32)
    y_data = np.array(y_data).astype(np.float32)
    print(y_data)
    y_data = y_data - 1
    print(X_data.shape)
    print(y_data.shape)
    print(np.max(y_data))

    x_train, x_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3)
    print(x_train.shape, y_train.shape)
    x_val, y_val = x_test, y_test

    return x_train, y_train, x_val, y_val, x_test, y_test

def Load_wine():
    logger = logging.getLogger(__name__)
    logger.info("start loading wine data")
    with open('./datasets/wine/wine.data') as f:
        filelines = f.readlines()
        X_data = []
        y_data = []
        for fileline in filelines:
            fileline = fileline.strip()
            row_data = fileline.split(',')

            X_data.append(row_data[1:])
            y_data.append(float(row_data[0]))
    X_data = np.array(X_data).astype(np.float32)
    y_data = np.array(y_data).astype(np.float32)
    print(y_data)
    y_data = y_data - 1
    print(X_data.shape)
    print(y_data.shape)
    print(X_data)
    print(y_data)
    print(np.max(y_data))

    x_train, x_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3)
    print(x_train.shape, y_train.shape)
    x_val, y_val = x_test, y_test

    return x_train, y_train, x_val, y_val, x_test, y_test

def Load_chess_krkopt():
    logger = logging.getLogger(__name__)
    logger.info("start loading chess krkopt data")
    with open('./datasets/chess_krkopt/krkopt.data') as f:
        filelines = f.readlines()
        X_data = []
        y_data = []
        for fileline in filelines:
            fileline = fileline.strip()
            row_data = fileline.split(',')

            X_data.append(row_data[:-1])
            y_data.append(row_data[-1])
    X_data = np.array(X_data)
    X_numerical = X_data[:, [1,3,5]]
    X_numerical = X_numerical.astype(np.float32)
    #print(X_numerical)
    #print(X_data)
    X_onehot = X_data[:, [0,2,4]]
    #print(X_onehot)
    onehot = OneHotEncoder()
    X_onehot = onehot.fit_transform(X_onehot).toarray()
    X_onehot = X_onehot.astype(np.float32)
    #print(X_onehot.shape)
    #print(X_onehot)

    X_data = np.concatenate((X_onehot, X_numerical), axis=1)


    label = LabelEncoder()
    y_data = label.fit_transform(y_data)
    y_data = np.array(y_data)
    print(X_data.shape)
    print(y_data.shape)
    print(X_data)
    print(y_data)
    print(np.max(y_data))

    x_train, x_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3)
    print(x_train.shape, y_train.shape)
    x_val, y_val = x_test, y_test

    return x_train, y_train, x_val, y_val, x_test, y_test



def Load_letter():
    logger = logging.getLogger(__name__)
    logger.info("start loadding letter file")

    with open('./datasets/letter/letter-recognition.data') as f:
        filelines = f.readlines()
        X_data = []
        y_data = []
        for fileline in filelines:
            fileline = fileline.strip()
            row_data = fileline.split(',')

            X_data.append(row_data[1:])
            y_data.append(row_data[0])
    X_data = np.array(X_data).astype(np.float32)

    label = LabelEncoder()
    y_data = label.fit_transform(y_data)
    y_data = np.array(y_data)
    print(X_data.shape)
    print(y_data.shape)
    print(X_data)
    print(y_data)
    print(np.max(y_data))

    x_train, x_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3)
    print(x_train.shape, y_train.shape)
    x_val, y_val = x_test, y_test

    return x_train, y_train, x_val, y_val, x_test, y_test



def Load_chess_krkp():
    logger = logging.getLogger(__name__)
    logger.info("start loading chess kr-vs-kp data")
    with open('./datasets/chess_krkp/kr-vs-kp.data') as f:
        filelines = f.readlines()
        X_data = []
        y_data = []
        for fileline in filelines:
            fileline = fileline.strip()
            row_data = fileline.split(',')
            if len(row_data) < 10:
                continue
            assert len(row_data)==37
            X_data.append(row_data[:-1])
            y_data.append(row_data[-1])
    #X_data = np.array(X_data)
    #print(X_data)
    onehot = OneHotEncoder()
    X_data = onehot.fit_transform(X_data).toarray()
    X_data = X_data.astype(np.float32)
    label = LabelEncoder()
    y_data = label.fit_transform(y_data)
    y_data = np.array(y_data)
    print(X_data.shape)
    print(y_data.shape)
    print(X_data)
    print(y_data)

    x_train, x_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3)
    print(x_train.shape, y_train.shape)
    x_val, y_val = x_test, y_test

    return x_train, y_train, x_val, y_val, x_test, y_test


def Load_Hill_Valley_with_noise():
    logger = logging.getLogger(__name__)
    logger.info("start loading Hill_valley data")

    with open('./datasets/Hill_Valley_with_noise/Hill_Valley_with_noise_Training.data') as f:
        filelines = f.readlines()[1:]
        train_data = []
        for fileline in filelines:
            fileline = fileline.strip()
            row_data = fileline.split(',')
            row_data = [float(item) for item in row_data]
            train_data.append(row_data)
    train_data = np.array(train_data)
    x_train = train_data[:, :-1]
    y_train = train_data[:, -1]
    print(x_train)
    print(x_train.shape)

    with open('./datasets/Hill_Valley_with_noise/Hill_Valley_with_noise_Testing.data') as f:
        filelines = f.readlines()[1:]
        test_data = []
        for fileline in filelines:
            fileline = fileline.strip()
            row_data = fileline.split(',')
            row_data = [float(item) for item in row_data]
            test_data.append(row_data)
    test_data = np.array(test_data)
    x_test = test_data[:, :-1]
    y_test = test_data[:, -1]
    print(x_test.shape)
    print(y_test.shape)

    x_val = x_test
    y_val = y_test
    return x_train, y_train, x_val, y_val, x_test, y_test

def Load_HTRU_2():
    logger = logging.getLogger(__name__)
    logger.info("start loading HTRU_2 data")

    with open('./datasets/HTRU_2/HTRU_2.csv') as f:
        filelines = f.readlines()
        data = []
        for fileline in filelines:
            fileline = fileline.strip()
            row_data = fileline.split(',')
            row_data = [float(item) for item in row_data]
            data.append(row_data)
    data = np.array(data)
    x_data = data[:, :-1]
    y_data = data[:, -1]
    print(x_data.shape)

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3)

    x_val = x_test
    y_val = y_test
    return x_train, y_train, x_val, y_val, x_test, y_test

def Load_abalone():
    logger = logging.getLogger(__name__)
    logger.info("start loading abalone data")
    encode_dict = {'F': 0, 'M': 1, 'I': 2}
    with open('./datasets/abalone/abalone.data') as f:
        filelines = f.readlines()
        X_data = []
        y_data = []
        for fileline in filelines:
            fileline = fileline.strip()
            row_data = fileline.split(',')

            row_data[-1] = float(row_data[-1])
            if row_data[-1] <= 9:
                row_data[-1] = 0
            else:
                row_data[-1] = 1

            X_data.append(row_data[:-1])
            y_data.append(row_data[-1])


    X_data = np.array(X_data)

    X_numerical = X_data[:, 1:]
    X_numerical = X_numerical.astype(np.float32)
    # print(X_numerical)
    # print(X_data)
    X_onehot = X_data[:, 0].reshape(-1, 1)
    # print(X_onehot)
    onehot = OneHotEncoder()
    X_onehot = onehot.fit_transform(X_onehot).toarray()
    X_onehot = X_onehot.astype(np.float32)
    # print(X_onehot.shape)
    # print(X_onehot)

    X_data = np.concatenate((X_onehot, X_numerical), axis=1)
    print(X_data.shape)

    x_train, x_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3)

    x_val = x_test
    y_val = y_test

    return x_train, y_train, x_val, y_val, x_test, y_test

def Load_dna():
    logger = logging.getLogger(__name__)
    logger.info("start loading DNA file")
    # Download these 2 files from the LIBSVM dataset page
    x_train, y_train, x_val, y_val, x_test, y_test = load_svmlight_files(
        ("./datasets/dna/dna.scale.tr",
         "./datasets/dna/dna.scale.val",
         "./datasets/dna/dna.scale.t"))
    # for simplicity, convert sparse matrices to dense ones. We need to handle sparse input later.
    x_train = x_train.toarray().astype(np.float32)
    x_val = x_val.toarray().astype(np.float32)
    x_test = x_test.toarray().astype(np.float32)
    y_train = y_train - 1
    y_val = y_val-1
    y_test = y_test - 1
    logger.info("x_train size is {}, x_val size is {}, x_test size is {}".format(x_train.shape, x_val.shape, x_test.shape))
    return x_train, y_train, x_val, y_val, x_test, y_test

def Load_kidney_disease():
    logger = logging.getLogger(__name__)
    logger.info("start loading kidney_disease data")
    X_data = np.genfromtxt('./datasets/kidney_disease/_data.txt', dtype='f8', delimiter=',')
    y_data = np.genfromtxt('./datasets/kidney_disease/_labels.txt', dtype='f8', delimiter=',')
    y_data = (y_data+1)//2
    print(y_data)

    x_train, x_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3)

    x_val = x_test
    y_val = y_test

    return x_train, y_train, x_val, y_val, x_test, y_test

def Load_horse():
    logger = logging.getLogger(__name__)
    logger.info("start loading horse data")
    X_data = np.genfromtxt('./datasets/horse/_data.txt', dtype='f8', delimiter=',')
    y_data = np.genfromtxt('./datasets/horse/_labels.txt', dtype='f8', delimiter=',')
    y_data = (y_data+1)//2

    x_train, x_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3)
    print(x_train.shape, x_test.shape)
    x_val = x_test
    y_val = y_test

    return x_train, y_train, x_val, y_val, x_test, y_test

def Load_pima():
    logger = logging.getLogger(__name__)
    logger.info("start loading pima data")
    X_data = np.genfromtxt('./datasets/pima/_data.txt', dtype='f8', delimiter=',')
    y_data = np.genfromtxt('./datasets/pima/_labels.txt', dtype='f8', delimiter=',')
    y_data = (y_data+1)//2

    x_train, x_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3)
    print(x_train.shape, x_test.shape)
    x_val = x_test
    y_val = y_test

    return x_train, y_train, x_val, y_val, x_test, y_test

def Load_mammographic_masses():
    logger = logging.getLogger(__name__)
    logger.info("start loading mammographic masses data")
    X_data = np.genfromtxt('./datasets/data_rbfn/mammographics/_data.txt', dtype='f8', delimiter=',')
    y_data = np.genfromtxt('./datasets/data_rbfn/mammographics/_labels.txt', dtype='f8', delimiter=',')
    y_data = (y_data+1)//2

    x_train, x_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3)
    print(x_train.shape, x_test.shape)
    x_val = x_test
    y_val = y_test

    return x_train, y_train, x_val, y_val, x_test, y_test


def Load_bands():
    logger = logging.getLogger(__name__)
    logger.info("start loading bands data")
    X_data = np.genfromtxt('./datasets/data_rbfn/bands/_data.txt', dtype='f8', delimiter=',')
    y_data = np.genfromtxt('./datasets/data_rbfn/bands/_labels.txt', dtype='f8', delimiter=',')
    y_data = (y_data+1)//2

    x_train, x_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3)
    print(x_train.shape, x_test.shape)
    x_val = x_test
    y_val = y_test

    return x_train, y_train, x_val, y_val, x_test, y_test



def Load_hepatitis():
    logger = logging.getLogger(__name__)
    logger.info("start loading hepatitis data")
    X_data = np.genfromtxt('./datasets/data_rbfn/hepatitis/_data.txt', dtype='f8', delimiter=',')
    y_data = np.genfromtxt('./datasets/data_rbfn/hepatitis/_labels.txt', dtype='f8', delimiter=',')
    y_data = (y_data+1)//2

    x_train, x_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3)
    print(x_train.shape, x_test.shape)
    x_val = x_test
    y_val = y_test

    return x_train, y_train, x_val, y_val, x_test, y_test

'''
def Load_mammographic_masses():
    logger = logging.getLogger(__name__)
    logger.info("start loading mammographic_masses data")
    with open('./datasets/mammographic_masses/mammographic_masses.data') as f:
        filelines = f.readlines()
        data = []
        for fileline in filelines:
            fileline = fileline.strip()
            row_data = []
            for item in fileline.split(','):
                if item == '?':
                    row_data.append(np.nan)
                else:
                    row_data.append(float(item))

            data.append(row_data)

    data = np.array(data)
    X_data = data[:, :-1]
    y_data = data[:, -1]

    x_train, x_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3)

    x_val = x_test
    y_val = y_test

    return x_train, y_train, x_val, y_val, x_test, y_test

def Load_hepatitis():
    logger = logging.getLogger(__name__)
    logger.info("start loading hepatitis data")
    with open('./datasets/hepatitis/hepatitis.data') as f:
        filelines = f.readlines()
        data = []
        for fileline in filelines:
            fileline = fileline.strip()
            row_data = []
            for item in fileline.split(','):
                if item == '?':
                    row_data.append(np.nan)
                else:
                    row_data.append(float(item))

            data.append(row_data)

    data = np.array(data)
    X_data = data[:, :-1]
    y_data = data[:, -1]-1
    print(X_data.shape)
    print(X_data)
    x_train, x_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3)

    x_val = x_test
    y_val = y_test

    return x_train, y_train, x_val, y_val, x_test, y_test
'''


def Load_heart():
    logger = logging.getLogger(__name__)
    logger.info("start loading heart file")
    # Download these 2 files from the LIBSVM dataset page
    x, y = load_svmlight_file(
        ("./datasets/heart/heart_scale"))
    # for simplicity, convert sparse matrices to dense ones. We need to handle sparse input later.
    x = x.toarray().astype(np.float32)
    y = (y + 1) // 2

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    x_val, y_val = x_test, y_test

    print("x_train size is {}, x_val size is {}, x_test size is {}".format(x_train.shape, x_val.shape, x_test.shape))
    return x_train, y_train, x_val, y_val, x_test, y_test


def Load_segment():
    logger = logging.getLogger(__name__)
    logger.info("start loading segment file")
    # Download these 2 files from the LIBSVM dataset page
    x, y = load_svmlight_file(
        ("./datasets/segment/segment.scale"))
    # for simplicity, convert sparse matrices to dense ones. We need to handle sparse input later.
    x = x.toarray().astype(np.float32)
    y = y-1

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    x_val, y_val = x_test, y_test
    print("x_train size is {}, x_val size is {}, x_test size is {}".format(x_train.shape, x_val.shape, x_test.shape))

    logger.info("x_train size is {}, x_val size is {}, x_test size is {}".format(x_train.shape, x_val.shape, x_test.shape))
    return x_train, y_train, x_val, y_val, x_test, y_test


def Load_mnist():
    logger = logging.getLogger(__name__)
    logger.info("start loading mnist file")
    x_train, y_train, x_test, y_test = load_svmlight_files(("./datasets/mnist/mnist", "./datasets/mnist/mnist.t"))

    x_train = x_train.toarray().astype(np.float32) / 255.0
    x_test = x_test.toarray().astype(np.float32) / 255.0

    logger.info("x_train size is {}, x_test size is {}".format(x_train.shape, x_test.shape))
    x_val, y_val = x_test, y_test

    return x_train, y_train, x_val, y_val, x_test, y_test

def Load_madelon():
    logger = logging.getLogger(__name__)
    logger.info("start loading madelon file")
    x_train, y_train, x_test, y_test = load_svmlight_files(("./datasets/madelon/madelon", "./datasets/madelon/madelon.t"))

    x_train = x_train.toarray().astype(np.float32) 
    x_test = x_test.toarray().astype(np.float32) 

    y_train = (y_train + 1) // 2
    y_test = (y_test + 1) // 2

    #print(y_train, y_test)

    logger.info("x_train size is {}, x_test size is {}".format(x_train.shape, x_test.shape))
    x_val, y_val = x_test, y_test

    return x_train, y_train, x_val, y_val, x_test, y_test