import os
from yacs.config import CfgNode as CN

#-----------------------------------------------
#Config definition
#-----------------------------------------------
_C = CN()

#Model
_C.MODEL = CN()
_C.MODEL.META_ARCHITECTURE = "UciNet"
_C.MODEL.DH = 2
_C.MODEL.DW = 2
_C.MODEL.N_MUL = 5
_C.MODEL.DD = 36
_C.MODEL.N_PER_GROUP = 1


#Model FC
_C.MODEL.FC = CN()
#FC units number
_C.MODEL.FC.N_FC = 128
#dense or one_fc
_C.MODEL.FC.B_FC = True
#
_C.MODEL.FC.NUM_LAYERS = 2

#train policy
_C.MODEL.ENSEMBLEIMP = 1

_C.MODEL.SAMPLE = CN()
#train policy
_C.MODEL.SAMPLE.FIXEDTRAIN = False
#test policy
_C.MODEL.SAMPLE.ENSEMBLETEST = 1



#used for resume from checkpoint
_C.MODEL.RESUME = ""


#imputer strategies
_C.PREPROCESSING = CN()
_C.PREPROCESSING.IMPUTER = 'None'

#DataSet
_C.DATASETS = CN()
_C.DATASETS.NAME = 'mnist'
_C.DATASETS.CLASS = 10
#missing value fraction
_C.DATASETS.MISSING_FRAC = 0.
#if create missing is true, it means we manually create missing values on a complete dataset
_C.DATASETS.CREATE_MISSING = True
#K fold number
_C.DATASETS.K_FOLD_NUMBER = 1
#num_bins in histogram
_C.DATASETS.NUM_BINS = 10

#Solver
_C.SOLVER = CN()
_C.SOLVER.NUM_ROUNDS = 5
_C.SOLVER.MAX_EPOCHS = 50
_C.SOLVER.TRAIN_PER_BATCH = 128
_C.SOLVER.TEST_PER_BATCH = 128
_C.SOLVER.LR_SCHEDULER_ON = False
_C.SOLVER.LR_SCHEDULER_GAMMA = 0.1
_C.SOLVER.LR_SCHEDULER_MILESTONE = [100, 150]

_C.SOLVER.OPTIMIZER = 'SGD'

_C.SOLVER.LR = 0.001
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.WEIGHT_DECAY = 2e-4

_C.TRAIN = True


#Misc options
_C.OUTPUT_DIR = "."
_C.LOG_EPOCHS = 1
