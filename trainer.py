import os
import shutil
import logging
import time
import torch
import numpy as np
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import MultiStepLR, StepLR
from torch.nn.functional import softmax
from loader import get_dataset


def train(model, cfg, logger, mi_idx=-1, cv_idx=-1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cfg.SOLVER.OPTIMIZER=='Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.SOLVER.LR,
                                   momentum=cfg.SOLVER.MOMENTUM, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    scheduler = MultiStepLR(optimizer, milestones=cfg.SOLVER.LR_SCHEDULER_MILESTONE,
                            gamma=cfg.SOLVER.LR_SCHEDULER_GAMMA)
    #scheduler = StepLR(optimizer, step_size=30, gamma=cfg.SOLVER.LR_SCHEDULER_GAMMA)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    start_epoch = 0
    best_acc = 0.

    if cfg.MODEL.RESUME:
        if os.path.isfile((cfg.MODEL.RESUME)):
            logger.info("=> loading checkpoint '{}'".format(cfg.MODEL.RESUME))
            checkpoint = torch.load(cfg.MODEL.RESUME)
            model.load_state_dict(checkpoint['state_dict'])
            start_epoch = checkpoint['epoch']
            best_acc = checkpoint['acc']
            #scheduler.load_state_dict(checkpoint['scheduler'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(cfg.MODEL.RESUME, start_epoch))
    else:
        logger.info("=> no checkpoint found at '{}'".format(cfg.MODEL.RESUME))

    cudnn.benchmark = True

    train_loader, missing_test_loader, full_test_loader, train_preprocessing_time, test_preprocessing_time = get_dataset(cfg, mi_idx, cv_idx)
    logger.info("finish build model\n{}".format(model))
    loss_func = torch.nn.CrossEntropyLoss()
    loss_func.to(device)

    logger.info("start training for {} epochs".format(cfg.SOLVER.MAX_EPOCHS))

    is_best = False
    best_epoch = start_epoch
    eval_acc = 0.

    train_losses = []
    eval_losses = []
    train_accs = []
    eval_accs = []
    train_data_times = []
    train_total_times = []
    test_data_times = []
    test_total_times = []

    for epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCHS):
        # train per epoch
        train_loss, train_acc, train_data_time, train_total_time = train_per_epoch(model, train_loader, device, loss_func, optimizer)

        #adjust learning rate
        if cfg.SOLVER.LR_SCHEDULER_ON:
            scheduler.step()

        #test per epoch
        eval_loss, eval_acc, _, _, test_data_time, test_total_time = eval_per_epoch(model, missing_test_loader, device, loss_func)
        
        full_eval_loss, full_eval_acc, _, _, _, _ = eval_per_epoch(model, full_test_loader, device, loss_func)

        is_best = False
        if eval_acc>best_acc:
            is_best = True
            best_acc = eval_acc
            best_epoch = epoch+1

        filename = []
        filename.append(os.path.join(cfg.OUTPUT_DIR, 'checkpoint_fold_{}_mi_{}.pth.tar'.format(cv_idx, mi_idx)))
        filename.append(os.path.join(cfg.OUTPUT_DIR, 'model_best_fold_{}_mi_{}.pth.tar').format(cv_idx, mi_idx))

        save_checkpoint({
            'epoch': epoch+1,
            'arch': cfg.MODEL.META_ARCHITECTURE,
            'state_dict': model.state_dict(),
            'acc': eval_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best, filename)

        if (epoch + 1) % cfg.LOG_EPOCHS == 0:
            logger.info("epoch {}".format(epoch + 1))
            logger.info("Train Loss: {:.6f}, Acc: {:.6f}%".format(train_loss, train_acc))
            logger.info("Test Loss: {:.6f}, Acc: {:.6f}%".format(eval_loss, eval_acc))
            logger.info("Full Test Loss: {:.6f}, Full Acc: {:.6f}%".format(full_eval_loss, full_eval_acc))
            logger.info('save model into {}'.format(filename[0]))
        train_losses.append(train_loss)
        eval_losses.append(eval_loss)
        train_accs.append(train_acc)
        eval_accs.append(eval_acc)

        with open(os.path.join(cfg.OUTPUT_DIR, 'train_acc.txt'), 'a') as f:
            f.write(str(train_acc))
            f.write('\n')
        with open(os.path.join(cfg.OUTPUT_DIR, 'missing_test_acc.txt'), 'a') as f:
            f.write(str(eval_acc))
            f.write('\n')
        with open(os.path.join(cfg.OUTPUT_DIR, 'full_test_acc.txt'), 'a') as f:
            f.write(str(full_eval_acc))
            f.write('\n')

        train_data_times.append(train_data_time)
        train_total_times.append(train_total_time)
        test_data_times.append(test_data_time)
        test_total_times.append(test_data_time)
    #logger.info('Train losses {}'.format(train_losses))
    #logger.info('Test losses {}'.format(eval_losses))
    #logger.info('Train accs {}'.format(train_accs))
    #logger.info('Test accs {}'.format(eval_accs))

    logger.info('Average training time per epoch, data: {:.3f}, total: {:.3f}'.format(np.mean(train_data_times), np.mean(train_total_times)))
    logger.info('Average inference time per epoch, data: {:.3f}, total: {:.3f}'.format(np.mean(test_data_times), np.mean(test_total_times)))

    logger.info('Best at epoch %d, missing_test_accuracy %f' % (best_epoch, best_acc))

    logger.info('start final evaluate on both missing_testset and full_testset')

    #_, missing_eval_acc, missing_outputs, targets = eval_per_epoch(model, missing_test_loader, device, loss_func)
    #_, full_eval_acc, full_outputs, targets = eval_per_epoch(model, full_test_loader, device, loss_func)

    full_eval_acc, missing_eval_acc, missing_outputs, targets = ensemble_test(model, cfg, logger, mi_idx, cv_idx)

    #logger.info('Accuracy on missing testset is %f, on full test is %f' % (missing_eval_acc, full_eval_acc))

    return model, full_eval_acc, missing_eval_acc, missing_outputs, targets, train_preprocessing_time+np.sum(train_total_times), test_preprocessing_time+np.mean(test_total_times)


def cal_ensemble_accuracy(outputs, targets):
    '''

    :param outputs: probability list with each item corresponds to output from a model
    :param targets:
    :return: the combined accuracy from these outputs
    '''
    ensemble_output = np.zeros(outputs[0].shape)
    for item in outputs:
        ensemble_output += item

    ensemble_pred = np.argmax(ensemble_output, axis=1)
    ensemble_eval_acc = 100 * np.sum(ensemble_pred == targets) / ensemble_pred.shape[0]
    return ensemble_eval_acc, ensemble_output


def ensemble_test(model, cfg, logger, mi_idx=-1, cv_idx=-1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not cfg.TRAIN:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)
        cfg.MODEL.RESUME = os.path.join(cfg.OUTPUT_DIR, 'checkpoint_fold_{}_mi_{}.pth.tar'.format(cv_idx, mi_idx))

        if os.path.isfile((cfg.MODEL.RESUME)):
            logger.info("=> loading checkpoint '{}'".format(cfg.MODEL.RESUME))
            checkpoint = torch.load(cfg.MODEL.RESUME)
            model.load_state_dict(checkpoint['state_dict'])
            start_epoch = checkpoint['epoch']
            best_acc = checkpoint['acc']
            logger.info("=> loaded checkpoint '{}' (epoch {}, acc {})".format(cfg.MODEL.RESUME, start_epoch, best_acc))
        else:
            logger.info("=> no checkpoint found at '{}'".format(cfg.MODEL.RESUME))

    model.eval()
    cudnn.benchmark = True
    loss_func = torch.nn.CrossEntropyLoss()

    outputs = []
    accs = []
    for i in range(cfg.MODEL.SAMPLE.ENSEMBLETEST):
        missing_test_loader, full_test_loader,_,_ = get_dataset(cfg, mi_idx, cv_idx, train=False)
        _, eval_acc, outputs_i, targets,_, _ = eval_per_epoch(model, missing_test_loader, device, loss_func)
        outputs.append(outputs_i)
        accs.append(eval_acc)

    _, full_eval_acc, _, _,_,_ = eval_per_epoch(model, full_test_loader, device, loss_func)

    missing_eval_acc, missing_outputs = cal_ensemble_accuracy(outputs, targets)

    if 'Sample' in cfg.MODEL.META_ARCHITECTURE:
        for i in [1,3,5,10,15,20]:
            temp_missing_eval_acc, _ = cal_ensemble_accuracy(outputs[:i], targets)
            logger.info('Multiple test at ensemble_size {} for UciSampleNet acc is {:.2f}'.format(i, temp_missing_eval_acc))
    logger.info('Accuracy on ensemble missing testset is {:.2f}'.format(missing_eval_acc))
    logger.info('Accuracy on full testset is {:.2f}'.format(full_eval_acc))

    return full_eval_acc, missing_eval_acc, missing_outputs, targets


def train_per_epoch(model, data_loader, device, loss_func, optimizer):
    train_loss = 0.
    train_acc = 0.
    train_total = 0.
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()
    torch.cuda.synchronize()
    end = time.time()
    for batch_x, batch_y in data_loader:
        torch.cuda.synchronize()
        data_time.update(time.time() - end)
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        out = model(batch_x)

        loss = loss_func(out, batch_y)
        train_loss += loss.item()
        pred = torch.max(out, 1)[1]
        train_correct = (pred == batch_y).sum().item()
        train_acc += train_correct
        train_total += batch_y.size(0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

    epoch_data_time = data_time.sum
    epoch_batch_time = batch_time.sum

    train_loss = train_loss / train_total
    train_acc = 100 * float(train_acc) / float(train_total)

    return train_loss, train_acc, epoch_data_time, epoch_batch_time



def eval_per_epoch(model, data_loader, device, loss_func):
    '''

    :param model:
    :param data_loader:
    :param device:
    :param loss_func:
    :return: eval_loss on data_loader, eval_acc on data_loader,output probabilities, targets
    '''
    model.eval()
    eval_loss = 0.
    eval_acc = 0.
    eval_total = 0.
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    for i, (batch_x, batch_y) in enumerate(data_loader):
        data_time.update(time.time() - end)
        with torch.no_grad():
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            out = model(batch_x)
            loss = loss_func(out, batch_y)
            eval_loss += loss.item()
            pred = torch.max(out, 1)[1]
            num_correct = (pred == batch_y).sum().item()
            eval_acc += num_correct
            eval_total += batch_y.size(0)

            #concat predicts
            out = softmax(out, dim=1)
            out = out.cpu().numpy()
            target = batch_y.cpu().numpy()
            batch_time.update(time.time() - end)
            end = time.time()
            if i==0:
                outs = out
                targets = target
            else:
                outs = np.concatenate((outs, out), axis=0)
                targets = np.concatenate((targets, target), axis=0)

    epoch_data_time = data_time.sum
    epoch_batch_time = batch_time.sum
    eval_loss = eval_loss / eval_total
    eval_acc = 100 * float(eval_acc) / float(eval_total)
    return eval_loss, eval_acc, outs, targets, epoch_data_time, epoch_batch_time


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename[0])
    if is_best:
        shutil.copyfile(filename[0], filename[1])

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch, cfg):
    lr = cfg.SOLVER.LR * (0.1**(epoch//30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr