'''
Created on 2018.03.05

@author: caoyh
'''
import os
import torch
import torchvision
import numpy as np
from model import *
from utils.config import cfg
from utils.logger import setup_logger
from trainer import train, ensemble_test, cal_ensemble_accuracy
import argparse
from copy import deepcopy

os.environ['OMP_NUM_THREADS'] = "10"
os.environ['MKL_NUM_THREADS'] = "10"


def get_model(cfg):
    '''
    param cfg:
    return model meta_architecture
    '''

    model_type = cfg.MODEL.META_ARCHITECTURE
    if 'Sample' in model_type:
        if 'NRS' in model_type:
            model = UciSampleNet(cfg)
        else:
            model = UciFCNet(cfg)
    else:
        model = eval(model_type)(cfg)

    return model

def main():
    parser = argparse.ArgumentParser(description="PyTorch Missing Feature Experiment")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    print(args.opts)
    cfg.merge_from_list(args.opts)
    fixedtrain  = 'fixed_train' if cfg.MODEL.SAMPLE.FIXEDTRAIN else 'not_fixed_train'
    
    cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, 'frac_{}'.format(float(cfg.DATASETS.MISSING_FRAC)), cfg.MODEL.META_ARCHITECTURE+cfg.PREPROCESSING.IMPUTER, fixedtrain, 'new_compare')
    #if not cfg.TRAIN:
    #    cfg.MODEL.RESUME = os.path.join(cfg.OUTPUT_DIR, 'checkpoint.pth.tar')
    #cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    print(output_dir)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("NRS", output_dir)
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n"+cf.read()
        logger.info(config_str)

    logger.info("Runnig with config:\n{}".format(cfg))

    if cfg.TRAIN:
        missing_accs = []
        full_accs = []
        training_times = []
        test_times = []
        #cv_idx for fold number
        for cv_idx in range(cfg.DATASETS.K_FOLD_NUMBER):
            logger.info("Start training on FOLD-{} datasets".format(cv_idx+1))
            #mi_idx for multiple dataset index
            individual_missing_accs = []
            missing_outputs = []
            for mi_idx in range(cfg.MODEL.ENSEMBLEIMP):
                logger.info("Start training on Imputation-{} datasets".format(mi_idx + 1))
                model = get_model(cfg)
                _, full_acc, missing_acc, missing_output, targets, training_time, test_time = train(model, cfg, logger, mi_idx, cv_idx)
                individual_missing_accs.append(missing_acc)
                missing_outputs.append(deepcopy(missing_output))
            #print(missing_outputs)
            #print(len(missing_outputs))
            ensemble_missing_acc, _ = cal_ensemble_accuracy(missing_outputs, targets)
            missing_accs.append(ensemble_missing_acc)
            full_accs.append(full_acc)
            training_times.append(training_time)
            test_times.append(test_time)

            if cfg.MODEL.ENSEMBLEIMP > 1:
                logger.info('individual missing test acc is {}, mean/std is {:.2f}/{:.2f}'.format(individual_missing_accs, np.mean(individual_missing_accs),
                                                                                np.std(individual_missing_accs)))
            logger.info('Fold-{} finish, final missing test acc is {:.2f}'.format(cv_idx+1, ensemble_missing_acc))
            logger.info('Fold-{} finish, final full test acc is {:.2f}'.format(cv_idx+1, full_acc))

        logger.info('missing accs: {:.2f}+-{:.2f}, {}'.format(np.mean(missing_accs), np.std(missing_accs), list(np.round(missing_accs, 2))))
        logger.info('full accs: {:.2f}+-{:.2f}, {}'.format(np.mean(full_accs), np.std(full_accs), list(np.round(full_accs, 2))))

        logger.info('training times: {:.2f}+-{:.2f}'.format(np.mean(training_times), np.std(training_times)))
        logger.info('test times: {:.2f}+-{:.2f}'.format(np.mean(test_times), np.std(test_times)))

        logger.info(list(np.round(missing_accs,2)))
        logger.info("Missing Mean(acc)={:.2f}%, Std(acc)={:.2f}%".format(np.mean(missing_accs), np.std(missing_accs)))

        logger.info(list(np.round(full_accs, 2)))
        logger.info("Full Mean(acc)={:.2f}%, Std(acc)={:.2f}%".format(np.mean(full_accs), np.std(full_accs)))

    else:
        missing_accs = []
        full_accs = []
        # cv_idx for fold number
        for cv_idx in range(cfg.DATASETS.K_FOLD_NUMBER):
            logger.info("Start evaluating on FOLD-{} datasets".format(cv_idx + 1))
            # mi_idx for multiple dataset index
            individual_missing_accs = []
            missing_outputs = []
            for mi_idx in range(cfg.MODEL.ENSEMBLEIMP):
                logger.info("Start evaluating on Imputation-{} datasets".format(mi_idx + 1))
                model = get_model(cfg)
                full_acc, missing_acc, missing_output, targets = ensemble_test(model, cfg, logger, mi_idx, cv_idx)
                individual_missing_accs.append(missing_acc)
                missing_outputs.append(deepcopy(missing_output))
            # print(missing_outputs)
            # print(len(missing_outputs))
            ensemble_missing_acc, _ = cal_ensemble_accuracy(missing_outputs, targets)
            missing_accs.append(ensemble_missing_acc)
            full_accs.append(full_acc)

            if cfg.MODEL.ENSEMBLEIMP > 1:
                logger.info(
                    'individual missing test acc is {}, mean/std is {:.2f}/{:.2f}'.format(individual_missing_accs,
                                                                                          np.mean(individual_missing_accs),
                                                                                          np.std(individual_missing_accs)))
            logger.info('Fold-{} finish, final missing test acc is {:.2f}'.format(cv_idx + 1, ensemble_missing_acc))
            logger.info('Fold-{} finish, final full test acc is {:.2f}'.format(cv_idx + 1, full_acc))

        logger.info('missing accs: {:.2f}+-{:.2f}, {}'.format(np.mean(missing_accs), np.std(missing_accs), list(np.round(missing_accs, 2))))
        logger.info('full accs: {:.2f}+-{:.2f}, {}'.format(np.mean(full_accs), np.std(full_accs), list(np.round(full_accs, 2))))

        logger.info(list(np.round(missing_accs, 2)))
        logger.info("Missing Mean(acc)={:.2f}%, Std(acc)={:.2f}%".format(np.mean(missing_accs), np.std(missing_accs)))

        logger.info(list(np.round(full_accs, 2)))
        logger.info("Full Mean(acc)={:.2f}%, Std(acc)={:.2f}%".format(np.mean(full_accs), np.std(full_accs)))

if __name__=="__main__":
    main()
