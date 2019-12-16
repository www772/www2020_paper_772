class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

import os, sys, shutil

from time import time
import numpy as np
import torch

def start(conf, data, model, record_id=0, pre_model='', bias_epoch=0):
    # start to prepare data for training and evaluating
    ###============================== TASK FLAG ==============================###
    #data.initializeTask3Handle()
    exec('data.initialize%sHandle()' % conf.task)
    d_train, d_val, d_test = data.train, data.val, data.test

    print('System start to load data...')
    t0 = time()
    #d_train.task3InitializeData()
    exec('d_train.%sInitializeData(model)' % conf.task.lower())
    #d_val.task3InitializeData()
    exec('d_val.%sInitializeData(model)' % conf.task.lower())
    #d_test.task3InitializeData()
    exec('d_test.%sInitializeData(model)' % conf.task.lower())
    t1 = time()
    print('Data has been loaded successfully, cost:%.4fs' % (t1 - t0))
    ###============================== TASK FLAG ==============================###

    model.initOptimizer()
    
    model.load_state_dict(torch.load('/home/sunpeijie/files/task/pyrec/log/dual_learning/amazon_electronics/dual_learning_v4_132/tmp_model/dual_learning_v4_e2r_loss_#0.9283#_r2e_loss_#1.9390#epoch_#1'))

    r2e_val_loss_dict, e2r_val_loss_dict = {}, {}

    # Start Training !!!
    tmp_epoch, epoch = 0, bias_epoch
    while epoch < conf.epochs:
        tmp_epoch += 1
        epoch = tmp_epoch + bias_epoch

        # optimize model with training data and compute train loss
        tmp_r2e_train_loss, tmp_r2e_val_loss, tmp_r2e_test_loss = [], [], []
        tmp_e2r_train_loss, tmp_e2r_val_loss, tmp_e2r_test_loss = [], [], []

        tmp_train_duality_loss, tmp_val_duality_loss, tmp_test_duality_loss = [], [], []
        t0 = time()
        
        train_epoch_loss, val_epoch_loss, test_epoch_loss = 0, 0, 0
        train_sample_count, val_sample_count, test_sample_count = 0, 0, 0

        # Following is the first training epoch, no optimization
        while d_train.terminal_flag and epoch == 0:
            #d_train.task3GetBatch()
            exec('d_train.%sGetBatch()' % conf.task.lower())

            train_feed_dict = {}
            for (key, value) in model.map_dict['input'].items():
                train_feed_dict[key] = d_train.data_dict[value]
            
            model.train(train_feed_dict)

            sub_r2e_train_loss = model.map_dict['out']['r2e_opt_loss']
            sub_e2r_train_loss = model.map_dict['out']['e2r_opt_loss']
            sub_train_duality_loss = model.map_dict['out']['duality_loss']
            #torch.cuda.empty_cache()

            train_epoch_loss += sub_e2r_train_loss
            train_sample_count += len(train_feed_dict['RATING_INPUT'])

            tmp_r2e_train_loss.append(sub_r2e_train_loss)
            tmp_e2r_train_loss.append(sub_e2r_train_loss)
            tmp_train_duality_loss.append(sub_train_duality_loss)
        r2e_train_loss = np.mean(tmp_r2e_train_loss)
        e2r_train_loss = np.mean(tmp_e2r_train_loss)
        train_duality_loss = np.mean(tmp_train_duality_loss)
        t1 = time()
        d_train.terminal_flag = 1

        train_epoch_loss = 0
        train_sample_count = 0

        # Following is the training process
        while d_train.terminal_flag and epoch >= 1:
            #import cProfile
            #cProfile.runctx('d_train.task3GetBatch()', globals(), locals())
            #d_train.task3GetBatch()
            exec('d_train.%sGetBatch()' % conf.task.lower())

            train_feed_dict = {}
            for (key, value) in model.map_dict['input'].items():
                train_feed_dict[key] = d_train.data_dict[value]
            
            model.train(train_feed_dict)
            model.optimizeE2R()
            sub_r2e_train_loss = model.map_dict['out']['r2e_opt_loss']
            sub_e2r_train_loss = model.map_dict['out']['e2r_opt_loss']
            sub_train_duality_loss = model.map_dict['out']['duality_loss']
            #torch.cuda.empty_cache()

            train_epoch_loss += sub_e2r_train_loss
            train_sample_count += len(train_feed_dict['RATING_INPUT'])

            tmp_r2e_train_loss.append(sub_r2e_train_loss)
            tmp_e2r_train_loss.append(sub_e2r_train_loss)
            tmp_train_duality_loss.append(sub_train_duality_loss)
        r2e_train_loss = np.mean(tmp_r2e_train_loss)
        e2r_train_loss = np.mean(tmp_e2r_train_loss)
        train_duality_loss = np.mean(tmp_train_duality_loss)
        t1 = time()
        d_train.terminal_flag = 1

        train_epoch_loss = 0
        train_sample_count = 0

        # Following is the training process
        while d_train.terminal_flag and epoch >= 1:
            #import cProfile
            #cProfile.runctx('d_train.task3GetBatch()', globals(), locals())
            #d_train.task3GetBatch()
            exec('d_train.%sGetBatch()' % conf.task.lower())

            train_feed_dict = {}
            for (key, value) in model.map_dict['input'].items():
                train_feed_dict[key] = d_train.data_dict[value]
            
            model.train(train_feed_dict)
            model.optimizeR2E()
            sub_r2e_train_loss = model.map_dict['out']['r2e_opt_loss']
            sub_e2r_train_loss = model.map_dict['out']['e2r_opt_loss']
            sub_train_duality_loss = model.map_dict['out']['duality_loss']
            #torch.cuda.empty_cache()

            train_epoch_loss += sub_e2r_train_loss
            train_sample_count += len(train_feed_dict['RATING_INPUT'])

            tmp_r2e_train_loss.append(sub_r2e_train_loss)
            tmp_e2r_train_loss.append(sub_e2r_train_loss)
            tmp_train_duality_loss.append(sub_train_duality_loss)
        r2e_train_loss = np.mean(tmp_r2e_train_loss)
        e2r_train_loss = np.mean(tmp_e2r_train_loss)
        train_duality_loss = np.mean(tmp_train_duality_loss)
        t1 = time()
        d_train.terminal_flag = 1

        # Following is used to compute the loss of val and test dataset
        while d_val.terminal_flag:
            #d_val.task3GetBatch()
            exec('d_val.%sGetBatch()' % conf.task.lower())

            val_feed_dict = {}
            for (key, value) in model.map_dict['input'].items():
                val_feed_dict[key] = d_val.data_dict[value]
            
            model.train(val_feed_dict)
            sub_r2e_val_loss = model.map_dict['out']['r2e_opt_loss']
            sub_e2r_val_loss = model.map_dict['out']['e2r_opt_loss']
            sub_val_duality_loss = model.map_dict['out']['duality_loss']
            #torch.cuda.empty_cache()

            val_epoch_loss += sub_e2r_val_loss
            val_sample_count += len(val_feed_dict['RATING_INPUT'])

            tmp_r2e_val_loss.append(sub_r2e_val_loss)
            tmp_e2r_val_loss.append(sub_e2r_val_loss)
            tmp_val_duality_loss.append(sub_val_duality_loss)
        r2e_val_loss = np.mean(tmp_r2e_val_loss)
        e2r_val_loss = np.mean(tmp_e2r_val_loss)
        val_duality_loss = np.mean(tmp_val_duality_loss)

        r2e_val_loss_dict[tmp_epoch] = r2e_val_loss
        e2r_val_loss_dict[tmp_epoch] = e2r_val_loss
        #print(np.sum(tmp_val_loss))
        d_val.terminal_flag = 1

        while d_test.terminal_flag:
            #d_test.task3GetBatch()
            exec('d_test.%sGetBatch()' % conf.task.lower())

            test_feed_dict = {}
            for (key, value) in model.map_dict['input'].items():
                test_feed_dict[key] = d_test.data_dict[value]
            
            model.train(test_feed_dict)
            sub_r2e_test_loss = model.map_dict['out']['r2e_opt_loss']
            sub_e2r_test_loss = model.map_dict['out']['e2r_opt_loss']
            sub_test_duality_loss = model.map_dict['out']['duality_loss']

            test_epoch_loss += sub_e2r_test_loss
            test_sample_count += len(test_feed_dict['RATING_INPUT'])

            tmp_r2e_test_loss.append(sub_r2e_test_loss)
            tmp_e2r_test_loss.append(sub_e2r_test_loss)
            tmp_test_duality_loss.append(sub_test_duality_loss)
        r2e_test_loss = np.mean(tmp_r2e_test_loss)
        e2r_test_loss = np.mean(tmp_e2r_test_loss)
        test_duality_loss = np.mean(tmp_test_duality_loss)
        d_test.terminal_flag = 1
        t2 = time()

        #print(np.sum(tmp_test_loss))
        
        e2r_train_loss = np.sqrt(train_epoch_loss / train_sample_count)
        e2r_val_loss = np.sqrt(val_epoch_loss / val_sample_count)
        e2r_test_loss = np.sqrt(test_epoch_loss / test_sample_count)
      
        # print log to console and log_file
        print('Epoch:%s' % (epoch))
        print('Duality train Loss:%.4f, val loss:%.4f, test loss:%.4f' % (train_duality_loss, val_duality_loss, test_duality_loss))
        print('E2R: train loss:%.4f, val loss:%.4f, test loss:%.4f' % (e2r_train_loss, e2r_val_loss, e2r_test_loss))
        print('R2E: train loss:%.4f, val loss:%.4f, test loss:%.4f' % (r2e_train_loss, r2e_val_loss, r2e_test_loss))
        print('Time: %.4fs, train cost:%.4fs, validation cost:%.4fs' % ((t2-t0), (t1-t0), (t2-t1)))