################################################################################
## import all required packages
import sys
import numpy as np
import random
import tensorflow as tf
import torchtuples as tt
from torchvision import datasets, transforms
import os
import time
import pandas as pd
import pycox as Pycox
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pycox.models import MTLR, PMF
from torch.utils.data.sampler import SubsetRandomSampler

from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
import torchtuples as tt


import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects import pandas2ri
pandas2ri.activate()
# import R's "base" package
base = rpackages.importr('base')
# import R's utility package
utils = rpackages.importr('utils')

# Install R packages needed for coxph(), RSF, and bigSurvSGD
"""
utils.install_packages('survival')
utils.install_packages('randomForestSRC')
utils.install_packages('parallel')
utils.install_packages('doParallel')
utils.install_packages('bigmemory')
utils.install_packages('bigSurvSGD')
"""
## Load R packages needed for coxph(), RSF, and bigSurvSGD
survival = rpackages.importr('survival')
randomForestSRC = rpackages.importr('randomForestSRC')
bigSurvSGD = rpackages.importr('bigSurvSGD')
survival = rpackages.importr('survival')
bigmemory = rpackages.importr('bigmemory')
doParallel = rpackages.importr('doParallel')
parallel = rpackages.importr('parallel')
################################################################################




################################################################################
## parameters
# Specify survival dataset you want to do analysis
Data = "nwtco"

# hyperparameters for Random Survival Forest
num_trees = [100, 500]
num_trees = [10, 10]
num_nodes_rsf = [5, 15]
num_nodes_rsf = [5]
m_tries = [2, 3, 4]
m_tries = [2]

# hyperparameters for BigSurvSGD
stata_sizes_bigSurvSGD = [2, 5, 10, 20, 50]

# hyperparameters for MLP-based models
# Network architecture
num_nodes = [32, 32]
out_features = 1
BATCH_NORM = False
ACTIVATION = 'relu'
# dropout rates
dropouts = [0.0, 0.1, 0.2]  # 0.0, 0.1, 0.2, 0.3, 0.4, 0.5
dropouts = [0.1]

output_bias = False
# batch_sizes
batch_sizes = [1, 32]
batch_sizes = [64]

# Maximum number of epochs
epochs = 10

# Specify if you want to use early stop with bigSurvMLP
if_early_stop = True

# number of patience epochs
Patience = 5

# strata size for bigSurvMLP
strata_size = 2

# Learning rates
LRs = [0.0001, 0.001, 0.01]  # 0.1, 0.01, 0.001, 0.0001
LRs = [0.0001]  # 0.1, 0.01, 0.001, 0.0001

# epoch step for calculating testing concordance
epoch_test = 5
verbose = False
# number of bins for discrete-time models
num_durations = 100

# Number of training/testing splits
num_rep = 1

# Number of folds for cross-validation
num_folds = 5

# Number of initial repetitions to tune the hyperparameters
num_rep_hyper = 1
################################################################################




################################################################################
## data preprocessing and preparation ##
################################################################################
def initializer_Data(Data):
    if Data == "metabric":
        ## METABRIC
        df_all = Pycox.datasets.metabric.read_df()
        df_all.columns = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6',
                          'x7', 'x8', 'time', 'status']
        print('metabric censoring: ', 1 - np.sum(df_all['status'] / len(df_all['status'])))

        cols_standardize = ['x0', 'x1', 'x2', 'x3', 'x8']
        cols_leave = ['x4', 'x5', 'x6', 'x7']
        standardize = [([col], StandardScaler()) for col in cols_standardize]
        leave = [(col, None) for col in cols_leave]
        x_mapper = DataFrameMapper(standardize + leave)
        x_all = x_mapper.fit_transform(df_all).astype('float32')
    elif Data == "flchain":
        ## FLCHAIN
        df_all = Pycox.datasets.flchain.read_df()
        df_all['sample.yr'] = pd.to_numeric(df_all['sample.yr'])
        df_all = pd.get_dummies(df_all)
        df_all.columns = ['age', 'sex', 'sample.yr', 'kappa', 'lambda', 'creatinine', 'mgus', 'time',
                          'status', 'flc.grp_1', 'flc.grp_2', 'flc.grp_3', 'flc.grp_4', 'flc.grp_5', 'flc.grp_6',
                          'flc.grp_7',
                          'flc.grp_8', 'flc.grp_9', 'flc.grp_10']
        df_all = df_all[['age', 'sex', 'sample.yr', 'kappa', 'lambda', 'creatinine', 'mgus', 'time',
                         'status', 'flc.grp_1', 'flc.grp_2', 'flc.grp_3', 'flc.grp_4', 'flc.grp_5', 'flc.grp_6',
                         'flc.grp_7',
                         'flc.grp_8', 'flc.grp_9']]
        print('Cesnsoring for ', Data, ' is: ', 1 - np.sum(df_all['status'] / len(df_all['status'])))

        for i in range(9, len(df_all.columns)):
            df_all[df_all.columns[i]] = df_all[df_all.columns[i]].astype('int64')

        cols_standardize = ['age', 'kappa', 'sample.yr', 'lambda', 'creatinine', 'mgus']
        cols_leave = ['status', 'flc.grp_1', 'flc.grp_2', 'flc.grp_3', 'flc.grp_4',
                      'flc.grp_5', 'flc.grp_6', 'flc.grp_7', 'flc.grp_8', 'flc.grp_9']
        standardize = [([col], StandardScaler()) for col in cols_standardize]
        leave = [(col, None) for col in cols_leave]
        x_mapper = DataFrameMapper(standardize + leave)
        x_all = x_mapper.fit_transform(df_all).astype('float32')
    elif Data == "gbsg":
        ## GBSG
        df_all = Pycox.datasets.gbsg.read_df()
        df_all.columns = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'time', 'status']
        print('Cesnsoring for ', Data, ' is: ', 1 - np.sum(df_all['status'] / len(df_all['status'])))
        cols_standardize = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6']
        cols_leave = []
        standardize = [([col], StandardScaler()) for col in cols_standardize]
        leave = [(col, None) for col in cols_leave]
        x_mapper = DataFrameMapper(standardize + leave)
        x_all = x_mapper.fit_transform(df_all).astype('float32')
    elif Data == "nwtco":
        ## NWTCO
        df_all = Pycox.datasets.nwtco.read_df()
        df_all.columns = ['stage', 'age', 'in.subcohort', 'instit_2', 'histol_2',
                          'study_4', 'time', 'status']
        print('Cesnsoring for ', Data, ' is: ', 1 - np.sum(df_all['status'] / len(df_all['status'])))

        df_all = pd.get_dummies(df_all)
        df_all = df_all[['age', 'in.subcohort', 'instit_2', 'histol_2', 'study_4', 'time', 'status',
                         'stage_1', 'stage_2', 'stage_3']]
        for i in range(7, len(df_all.columns)):
            df_all[df_all.columns[i]] = df_all[df_all.columns[i]].astype('int64')
        cols_standardize = ['age']
        cols_leave = ['in.subcohort', 'instit_2', 'histol_2', 'study_4', 'stage_1', 'stage_2', 'stage_3']
        standardize = [([col], StandardScaler()) for col in cols_standardize]
        leave = [(col, None) for col in cols_leave]
        x_mapper = DataFrameMapper(standardize + leave)
        x_all = x_mapper.fit_transform(df_all).astype('float32')
    else:
        ## SUPPORT
        df_all = Pycox.datasets.support.read_df()
        df_all.columns = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6',
                          'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13',
                          'time', 'status']
        print('Cesnsoring for ', Data, ' is: ', 1 - np.sum(df_all['status'] / len(df_all['status'])))
        cols_standardize = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6',
                            'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13']
        cols_leave = []
        standardize = [([col], StandardScaler()) for col in cols_standardize]
        leave = [(col, None) for col in cols_leave]
        x_mapper = DataFrameMapper(standardize + leave)
        x_all = x_mapper.fit_transform(df_all).astype('float32')

    return df_all, x_all
################################################################################


################################################################################
################################################################################
# We call methods to be analyzed
################################################################################
################################################################################


################################################################################
## RSF: Calling randomForestSRC package and claculate its concordance
def rsfConcPy(x):
    num_tree = x[0]
    num_node = x[1]
    m_try = x[2]
    return_conc = x[3]
    df_train = x[4]
    if return_conc == 'valid':
        df_test = x[4]
    else:
        df_test = x[5]

    rsfConc = robjects.r('''
                    # A function to estimate concordance using random forest
                    concRandomForest <- function(df_trainCV, df_testCV, num_tree, num_node, m_try){
                        for (i in 1:ncol(df_testCV)){
                            df_testCV[,i] <- as.numeric(as.character(df_testCV[,i]))
                            df_trainCV[,i] <- as.numeric(as.character(df_trainCV[,i]))
                        }
                        modRFSRC <- rfsrc(Surv(time, status) ~ ., data = df_trainCV, ntree = num_tree, nodesize=num_node, mtry=m_try)
                        survival.results <- predict(modRFSRC, newdata = df_testCV)
                        as.numeric(1-survival.results
                        $err.rate[length(survival.results$err.rate)])
                    }
        ''')
    return (rsfConc(df_train, df_test, num_tree, num_node, m_try))


################################################################################


################################################################################
## Calling coxph package and claculate its concordance
def coxPHConcPy(x):
    df_train = x[0]
    df_test = x[1]

    coxConc = robjects.r('''
        # A function to estimate beta and concordance using bigSurvSGD
        concCox <- function(df_train, df_test){
            for (i in 1:ncol(df_test)){
                df_test[,i] <- as.numeric(as.character(df_test[,i]))
                df_train[,i] <- as.numeric(as.character(df_train[,i]))
            }
            beta <- coxph(formula = Surv(time, status)~., data=df_train)$coef 
            f_beta <- as.matrix(df_test[, -which(!is.na(match(colnames(df_test), c("time", "status"))))]) %*% matrix(beta, ncol=1)
            orderedIndices <- order(df_test$time, decreasing = F)
            f_beta = f_beta[orderedIndices]
            Times = df_test$time[orderedIndices] 
            Events = df_test$status[orderedIndices]  
            list(f_beta=f_beta, Times=Times, Events=Events, beta=beta)
        }
    ''')
    f_beta_time_event = coxConc(df_train, df_test)
    f_beta = np.squeeze(f_beta_time_event[0])
    Times = f_beta_time_event[1]
    Events = f_beta_time_event[2]
    return (calc_conc(f_beta, Times, Events))


################################################################################


################################################################################
## Calling bigSurvSGD package and claculate its concordance
def bigSurvConcPy(x):
    df_train = x[0]
    df_test = x[1]
    strata_size = x[2]

    bigSurvConc = robjects.r('''
        # A function to estimate beta and concordance using bigSurvSGD
        concBigSurvSGD <- function(df_train, df_test, strata_size){
            for (i in 1:ncol(df_test)){
                df_test[,i] <- as.numeric(as.character(df_test[,i]))
                df_train[,i] <- as.numeric(as.character(df_train[,i]))
            }
        beta <- bigSurvSGD(formula = Surv(time, status)~., data=df_train, strata.size = strata_size, 
                            inference.method = "none")$coef 
        f_beta <- as.matrix(df_test[, -which(!is.na(match(colnames(df_test), c("time", "status"))))]) %*% matrix(beta, ncol=1)
        orderedIndices <- order(df_test$time, decreasing = F)
        f_beta = f_beta[orderedIndices] 
        Times = df_test$time[orderedIndices] 
        Events = df_test$status[orderedIndices]  
        list(f_beta=f_beta, Times=Times, Events=Events, beta=beta)        
    }
    ''')
    f_beta_time_event = bigSurvConc(df_train, df_test, strata_size)
    f_beta = np.squeeze(f_beta_time_event[0])
    Times = f_beta_time_event[1]
    Events = f_beta_time_event[2]
    return (calc_conc(f_beta, Times, Events))


################################################################################


################################################################################
## Make the MLP neural network
def whole_model(inputDim, num_nodes, BATCH_NORM, ACTIVATION, DROPOUT):
    inputs = tf.keras.Input(shape=(inputDim))
    x = tf.keras.layers.Dense(num_nodes[0], input_dim=inputDim,
                              kernel_initializer='uniform')(inputs)
    if BATCH_NORM:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(ACTIVATION)(x)
    x = tf.keras.layers.Dropout(DROPOUT)(x)
    if len(num_nodes) > 1:
        for l in range(1, len(num_nodes)):
            x = tf.keras.layers.Dense(num_nodes[l], input_dim=num_nodes[l - 1],
                                      kernel_initializer='uniform')(x)
            if BATCH_NORM:
                x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation(ACTIVATION)(x)
            x = tf.keras.layers.Dropout(DROPOUT)(x)

    outputs = tf.keras.layers.Dense(units=1)(x)
    inputs1 = tf.keras.Input(shape=(inputDim))
    inputs2 = tf.keras.Input(shape=(inputDim))
    singlemodel = tf.keras.Model(inputs, outputs)
    outputs1 = singlemodel(inputs1)
    outputs2 = singlemodel(inputs2)
    return tf.keras.Model([inputs1, inputs2], [outputs1, outputs2])


################################################################################


################################################################################
def eval(x, y, mnist_model):
    batch_size = 1
    strata_size = 2
    f_beta = np.zeros((len(y)), dtype="float16")
    for i in range(np.int(len(y) / 2)):
        outs = mnist_model([np.expand_dims(x[y[i * batch_size * strata_size], :], axis=0),
                            np.expand_dims(x[y[i * batch_size * strata_size + 1], :], axis=0)], training=False)
        f_beta[i * batch_size * strata_size] = np.squeeze(outs[0])
        f_beta[(i * batch_size * strata_size + 1)] = np.squeeze(outs[1])

    if len(y) % 2 == 1:
        Index1 = y[(len(y) - 1)]
        outs = mnist_model([np.expand_dims(x[Index1, :], axis=0),
                            np.expand_dims(x[Index1, :], axis=0)], training=False)
        f_beta[(len(y) - 1)] = np.squeeze(outs[0])
    return (f_beta)


################################################################################


################################################################################
## Calculate testing concordance index using estimated risk score
def calc_conc(f_beta, Timess, Eventss):
    orderedIndices = np.argsort(Timess)
    Eventss = np.array(Eventss)[orderedIndices]
    f_beta = np.array(f_beta)[orderedIndices]
    Timess = np.array(Timess)[orderedIndices]
    conc_Bin = 0
    k = 0
    for i in range(len(Eventss) - 1):
        # t1 < t2 and risk1 > risk2, d1 = 1
        Time1 = Timess[i]
        Time2 = Timess[(i + 1):len(Eventss)]
        Event1 = Eventss[i]
        Event2 = Eventss[(i + 1):len(Eventss)]
        f_1 = f_beta[i]
        f_2 = f_beta[(i + 1):len(Eventss)]
        num1 = np.sum(((Time1 != Time2) & (f_1 > f_2) & ((Event1 == 1))))
        k += num1
        conc_Bin += num1 * 1
        # t1 < t2 and risk1 = risk2, d1 = 1
        num1 = np.sum(((Time1 != Time2) & (f_1 == f_2) & ((Event1 == 1))))
        k += num1
        conc_Bin += num1 * 0.5
        # t1 < t2 and risk1 < risk2, d1 =1
        num1 = np.sum(((Time1 != Time2) & (f_1 < f_2) & ((Event1 == 1))))
        k += num1
        conc_Bin += num1 * 0.0
        # t1 = t2 and risk1 = risk2, d1=d2=1
        num1 = np.sum(((Time1 == Time2) & (f_1 == f_2) & ((Event1 == 1) & (Event2 == 1))))
        k += num1
        conc_Bin += num1 * 1
        # t1 = t2 and risk1 != risk2, d1 = d2 = 1
        num1 = np.sum(((Time1 == Time2) & (f_1 != f_2) & ((Event1 == 1) & (Event2 == 1))))
        k += num1
        conc_Bin += num1 * 0.5

        # t1 = t2 and risk1 > risk2, d1 = 1, d2=0
        num1 = np.sum(((Time1 == Time2) & (f_1 > f_2) & ((Event1 == 1) & (Event2 == 0))))
        k += num1
        conc_Bin += num1 * 1
        # t1 = t2 and risk1 = risk2, d1 = 1, d2=0
        num1 = np.sum(((Time1 == Time2) & (f_1 == f_2) & ((Event1 == 1) & (Event2 == 0))))
        k += num1
        conc_Bin += num1 * 0.5
        # t1 = t2 and risk1 < risk2, d1 = 1, d2=0
        num1 = np.sum(((Time1 == Time2) & (f_1 < f_2) & ((Event1 == 1) & (Event2 == 0))))
        k += num1
        conc_Bin += num1 * 0.0
        # t1 = t2 and risk1 > risk2, d1 = 0, d2=1
        num1 = np.sum(((Time1 == Time2) & (f_1 > f_2) & ((Event1 == 0) & (Event2 == 1))))
        k += num1
        conc_Bin += num1 * 0.0
        # t1 = t2 and risk1 = risk2, d1 = 0, d2=1
        num1 = np.sum(((Time1 == Time2) & (f_1 == f_2) & ((Event1 == 0) & (Event2 == 1))))
        k += num1
        conc_Bin += num1 * 0.5
        # t1 = t2 and risk1 < risk2, d1 = 0, d2=1
        num1 = np.sum(((Time1 == Time2) & (f_1 < f_2) & ((Event1 == 0) & (Event2 == 1))))
        k += num1
        conc_Bin += num1 * 1.0
    return conc_Bin / k



################################################################################


################################################################################
def concDisc(surv, Times, Events):
    times = list(np.array(surv.head(n=len(surv)).index, dtype="float64"))
    columns = list(surv.columns)
    median_surv_time = np.array([np.interp(0.5, xp=np.flip(np.array(surv[columns[i]], dtype="float64")),
                                           fp=np.sort(times)[::-1]) for i in range(len(columns))], dtype="float64")
    orderedIndices = np.argsort(Times)
    Eventss = np.array(Events)[orderedIndices]
    median_surv_time = median_surv_time[orderedIndices]
    Timess = np.array(Times)[orderedIndices]
    con_return = calc_conc(-median_surv_time, Timess, Eventss)
    return (con_return)


def concCont(surv, Times, Events):
    times = list(np.array(surv.head(n=len(surv)).index, dtype="float64"))
    columns = list(surv.columns)
    median_surv_time = np.array([np.interp(0.5, xp=np.flip(np.array(surv[columns[i]], dtype="float64")),
                                           fp=np.sort(times)[::-1]) for i in range(len(columns))], dtype="float64")
    orderedIndices = np.argsort(Times)
    Eventss = np.array(Events)[orderedIndices]
    median_surv_time = median_surv_time[orderedIndices]
    Timess = np.array(Times)[orderedIndices]
    con_return = calc_conc(-median_surv_time, Timess, Eventss)
    return (con_return)


################################################################################
## train the model using mini-batches of pair of patients
def train_step(image1, image2, mnist_model, optimizer):
    with tf.GradientTape() as tape:
        outs = mnist_model([image1, image2], training=True)
        loss_all = loss_object(outs[1] - outs[0])
        conc1 = np.nanmean(np.exp(-loss_all))
        grads = tape.gradient(loss_all, mnist_model.trainable_variables)
        optimizer.apply_gradients(zip(grads, mnist_model.trainable_variables))
    return (conc1)


def loss_object(outDiff):
    return (tf.keras.backend.log(1 + tf.keras.backend.exp(outDiff)))


################################################################################


################################################################################
## The main fucntion to read data and train network
def bigSurvSGDMLP(x):
    batch_size = x[0]
    DROPOUT = x[1]
    LR = x[2]
    num_nodes = x[3]
    BATCH_NORM = x[4]
    ACTIVATION = x[5]
    epochs = x[6]
    strata_size = x[7]
    epoch_test = x[8]
    if_early_stop = x[9]
    Patience = x[10]
    return_conc = x[11]
    df_train = x[12]
    x_train = x[13]
    df_valid = x[14]
    x_valid = x[15]
    if return_conc == 'valid':
        df_test = x[14]
        x_test = x[15]
    else:
        df_test = x[16]
        x_test = x[17]

    optimizer = tf.keras.optimizers.Adam(learning_rate=LR, amsgrad=True)
    x_test = np.expand_dims(x_test, axis=2)
    x_valid = np.expand_dims(x_valid, axis=2)

    times_test = df_test['time'].to_numpy(dtype="float64")
    events_test = df_test['status'].to_numpy(dtype="int64")

    times_valid = df_valid['time'].to_numpy(dtype="float64")
    events_valid = df_valid['status'].to_numpy(dtype="int64")

    times_train = df_train['time'].to_numpy(dtype="float64")
    events_train = df_train['status'].to_numpy(dtype="int64")

    inputDim = x_trainCV.shape[1]
    indices_test = np.arange(0, len(times_test))
    indices_train = np.arange(0, len(times_train))
    indices_valid = np.arange(0, len(times_valid))
    # Initialize model
    mnist_model = whole_model(inputDim, num_nodes, BATCH_NORM, ACTIVATION, DROPOUT)
    conc_test = []
    conc_valid = []
    f_beta_test_all = []
    conc_trainAll = []

    for i_e in range(epochs):
        indices = np.random.choice(indices_train,
                                   size=np.int(strata_size * np.floor(len(indices_train) / strata_size)),
                                   replace=False, p=None)
        conc_train = []
        times = times_train[indices].reshape(np.int(len(indices) / strata_size), strata_size)
        events = events_train[indices].reshape(np.int(len(indices) / strata_size), strata_size)
        indices = indices.reshape(np.int(len(indices) / strata_size), strata_size)
        indSorted = np.argsort(times, axis=1)
        events = np.take_along_axis(events, indSorted, axis=1)
        indices = np.take_along_axis(indices, indSorted, axis=1)
        times = np.take_along_axis(times, indSorted, axis=1)
        firstEventNot0EqualTimes = (events[:, 0] == 1) & (times[:, 0] != times[:, 1])
        indices = indices[firstEventNot0EqualTimes, :]
        if indices.shape[0] >= batch_size:
            for b in range(np.int(np.floor(indices.shape[0] / batch_size))):
                Index1 = indices[(b * batch_size):((b + 1) * batch_size), 0]
                Index2 = indices[(b * batch_size):((b + 1) * batch_size), 1]
                results = train_step(x_train[Index1, :],
                                     x_train[Index2, :],
                                     mnist_model, optimizer)
            conc_train.append(results)
        ## consider the remaining strata that are not a complete batch
        if (indices.shape[0] % batch_size) > 0:
            Index1 = indices[(batch_size *
                              np.int(np.floor(indices.shape[0] / batch_size))):indices.shape[0], 0]
            Index2 = indices[(batch_size *
                              np.int(np.floor(indices.shape[0] / batch_size))):indices.shape[0], 1]
            results = train_step(x_train[Index1, :],
                                 x_train[Index2, :],
                                 mnist_model, optimizer)
            conc_train.append(results)
        conc_trainAll.append(np.nanmean(conc_train))

        if (i_e + 1) % epoch_test == 0:
            f_beta_valid = eval(x_valid, indices_valid, mnist_model)
            conc_valid.append(calc_conc(f_beta_valid, times_valid, events_valid))

            if return_conc == "test":
                f_beta_test = eval(x_test, indices_test, mnist_model)
                f_beta_test_all.append(f_beta_test)
                conc_test.append(calc_conc(f_beta_test, times_test, events_test))
            if (if_early_stop & (i_e > (Patience * epoch_test))):
                conc_validMA = np.empty((len(conc_valid)))
                conc_validMA[:] = np.nan
                conc_validMA[0] = conc_valid[0]
                for k in range(1, len(conc_valid)):
                    weights = np.power(0.8, k - np.arange(k + 1))
                    conc_validMA[k] = np.ma.average(conc_valid[0:(k + 1)], weights=weights)
                if np.prod(conc_validMA[len(conc_validMA) - 1] <
                           conc_validMA[(len(conc_validMA) - Patience - 1):(len(conc_validMA) - 1)]):
                    break
    if return_conc == "test":
        conc_return = conc_test[np.nanargmax(conc_valid)]
    else:
        conc_return = np.nanmax(conc_valid)
    return conc_return


################################################################################



################################################################################
## CoxCC method
def CoxCC_met(x):
    batch_size = x[0]
    DROPOUT = x[1]
    LR = x[2]
    num_nodes = x[3]
    BATCH_NORM = x[4]
    ACTIVATION = x[5]
    epochs = x[6]
    Patience = x[7]
    return_conc = x[8]
    df_train = x[9]
    x_train = x[10]
    df_valid = x[11]
    x_valid = x[12]
    if return_conc == 'valid':
        df_test = x[11]
        x_test = x[12]
    else:
        df_test = x[13]
        x_test = x[14]

    out_features = 1
    output_bias = False
    verbose = False
    get_target = lambda df: (df['time'].values, df['status'].values)
    y_train = get_target(df_train)
    y_val = get_target(df_valid)
    durations_test, events_test = get_target(df_test)
    val = tt.tuplefy(x_valid, y_val)
    in_features = x_train.shape[1]
    net = tt.practical.MLPVanilla(in_features, num_nodes, out_features,
                                  BATCH_NORM, DROPOUT, output_bias=False)
    model_CoxCC = Pycox.models.CoxCC(net, tt.optim.Adam)
    model_CoxCC.optimizer.set_lr(LR)
    callbacks = [tt.callbacks.EarlyStopping(patience=Patience)]
    log = model_CoxCC.fit(x_train, y_train, batch_size, epochs, callbacks, verbose,
                          val_data=val.repeat(10).cat())
    _ = model_CoxCC.compute_baseline_hazards()
    surv = model_CoxCC.predict_surv_df(x_test)
    return (concDisc(surv, np.array(durations_test), np.array(events_test)))


################################################################################



################################################################################
## DeepSurv method
def DeepSurv_met(x):
    batch_size = x[0]
    DROPOUT = x[1]
    LR = x[2]
    num_nodes = x[3]
    BATCH_NORM = x[4]
    ACTIVATION = x[5]
    epochs = x[6]
    Patience = x[7]
    return_conc = x[8]
    df_train = x[9]
    x_train = x[10]
    df_valid = x[11]
    x_valid = x[12]
    if return_conc == 'valid':
        df_test = x[11]
        x_test = x[12]
    else:
        df_test = x[13]
        x_test = x[14]

    get_target = lambda df: (df['time'].values, df['status'].values)
    y_train = get_target(df_train)
    y_val = get_target(df_valid)
    durations_test, events_test = get_target(df_test)
    val = tt.tuplefy(x_valid, y_val)
    in_features = x_train.shape[1]

    out_features = 1
    output_bias = False
    verbose = False
    net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, BATCH_NORM,
                                  DROPOUT, output_bias=output_bias)
    model_DeepSurv = Pycox.models.CoxPH(net, tt.optim.Adam)
    model_DeepSurv.optimizer.set_lr(LR)

    callbacks = [tt.callbacks.EarlyStopping(patience=Patience)]
    log = model_DeepSurv.fit(x_train, y_train, batch_size, epochs, callbacks, verbose,
                             val_data=val, val_batch_size=batch_size)
    _ = model_DeepSurv.compute_baseline_hazards()
    surv = model_DeepSurv.predict_surv_df(x_test)
    return (concDisc(surv, np.array(durations_test), np.array(events_test)))


################################################################################


################################################################################
## CoxTime method
def CoxTime_met(x):
    batch_size = x[0]
    DROPOUT = x[1]
    LR = x[2]
    num_nodes = x[3]
    BATCH_NORM = x[4]
    ACTIVATION = x[5]
    epochs = x[6]
    Patience = x[7]
    return_conc = x[8]
    df_train = x[9]
    x_train = x[10]
    df_valid = x[11]
    x_valid = x[12]
    if return_conc == 'valid':
        df_test = x[11]
        x_test = x[12]
    else:
        df_test = x[13]
        x_test = x[14]
    verbose = False
    labtrans_CoxTime = Pycox.models.CoxTime.label_transform()
    get_target = lambda df: (df['time'].values, df['status'].values)
    y_train = labtrans_CoxTime.fit_transform(*get_target(df_train))
    y_val = labtrans_CoxTime.transform(*get_target(df_valid))
    durations_test, events_test = get_target(df_test)
    val = tt.tuplefy(x_valid, y_val)
    in_features = x_train.shape[1]
    net = Pycox.models.cox_time.MLPVanillaCoxTime(in_features=in_features, num_nodes=num_nodes, batch_norm=BATCH_NORM,
                                                  dropout=DROPOUT)
    model_CoxTime = Pycox.models.CoxTime(net, tt.optim.Adam, labtrans=labtrans_CoxTime)
    model_CoxTime.optimizer.set_lr(LR)
    callbacks = [tt.callbacks.EarlyStopping(patience=Patience)]
    log = model_CoxTime.fit(x_train, y_train, batch_size, epochs, callbacks,
                            verbose=verbose, val_data=val.repeat(10).cat())
    _ = model_CoxTime.compute_baseline_hazards()
    surv = model_CoxTime.predict_surv_df(x_test)
    return (concDisc(surv, np.array(durations_test), np.array(events_test)))


################################################################################
## DeepHit method
def DeepHit_met(x):
    batch_size = x[0]
    DROPOUT = x[1]
    LR = x[2]
    num_nodes = x[3]
    BATCH_NORM = x[4]
    ACTIVATION = x[5]
    epochs = x[6]
    Patience = x[7]
    num_durations = x[8]
    return_conc = x[9]
    df_train = x[10]
    x_train = x[11]
    df_valid = x[12]
    x_valid = x[13]

    if return_conc == 'valid':
        df_test = x[12]
        x_test = x[13]
    else:
        df_test = x[14]
        x_test = x[15]

    verbose = False

    labtrans_DeepHit = Pycox.models.DeepHitSingle.label_transform(num_durations)
    get_target = lambda df: (df['time'].values, df['status'].values)
    y_train = labtrans_DeepHit.fit_transform(*get_target(df_train))
    y_val = labtrans_DeepHit.transform(*get_target(df_valid))
    train = (x_train, y_train)
    val = (x_valid, y_val)
    # We don't need to transform the test labels
    durations_test, events_test = get_target(df_test)
    in_features = x_train.shape[1]
    out_features = labtrans_DeepHit.out_features

    net = tt.practical.MLPVanilla(in_features=in_features, num_nodes=num_nodes,
                                  out_features=out_features, batch_norm=BATCH_NORM,
                                  dropout=DROPOUT)
    model_DeepHit = Pycox.models.DeepHitSingle(net, tt.optim.Adam, alpha=0.2, sigma=0.1,
                                               duration_index=labtrans_DeepHit.cuts)
    model_DeepHit.optimizer.set_lr(LR)
    callbacks = [tt.callbacks.EarlyStopping(patience=Patience)]
    log = model_DeepHit.fit(x_train, y_train, batch_size, epochs, callbacks,
                            val_data=val, verbose=verbose)
    surv = model_DeepHit.interpolate(10).predict_surv_df(x_test)
    return (concDisc(surv, np.array(durations_test), np.array(events_test)))


################################################################################


################################################################################
## MTLR method
def MTLR_met(x):
    batch_size = x[0]
    DROPOUT = x[1]
    LR = x[2]
    num_nodes = x[3]
    BATCH_NORM = x[4]
    ACTIVATION = x[5]
    epochs = x[6]
    Patience = x[7]
    num_durations = x[8]
    return_conc = x[9]
    df_train = x[10]
    x_train = x[11]
    df_valid = x[12]
    x_valid = x[13]

    if return_conc == 'valid':
        df_test = x[12]
        x_test = x[13]
    else:
        df_test = x[14]
        x_test = x[15]

    verbose = False

    labtrans_MTLR = Pycox.models.MTLR.label_transform(num_durations)
    get_target = lambda df: (df['time'].values, df['status'].values)
    y_train = labtrans_MTLR.fit_transform(*get_target(df_train))
    y_val = labtrans_MTLR.transform(*get_target(df_valid))
    train = (x_train, y_train)
    val = (x_valid, y_val)
    # We don't need to transform the test labels
    durations_test, events_test = get_target(df_test)
    in_features = x_train.shape[1]
    out_features = labtrans_MTLR.out_features
    net = tt.practical.MLPVanilla(in_features=in_features, num_nodes=num_nodes, out_features=out_features,
                                  batch_norm=BATCH_NORM, dropout=DROPOUT)
    model_MTLR = Pycox.models.MTLR(net, tt.optim.Adam, duration_index=labtrans_MTLR.cuts)
    model_MTLR.optimizer.set_lr(LR)
    callbacks = [tt.callbacks.EarlyStopping(patience=Patience)]
    log = model_MTLR.fit(x_train, y_train, batch_size, epochs, callbacks,
                         val_data=val, verbose=verbose)
    surv = model_MTLR.predict_surv_df(x_test)
    return (concDisc(surv, np.array(durations_test), np.array(events_test)))


################################################################################


################################################################################
## PCHazard method
def PCHazard_met(x):
    batch_size = x[0]
    DROPOUT = x[1]
    LR = x[2]
    num_nodes = x[3]
    BATCH_NORM = x[4]
    ACTIVATION = x[5]
    epochs = x[6]
    Patience = x[7]
    num_durations = x[8]
    return_conc = x[9]
    df_train = x[10]
    x_train = x[11]
    df_valid = x[12]
    x_valid = x[13]

    if return_conc == 'valid':
        df_test = x[12]
        x_test = x[13]
    else:
        df_test = x[14]
        x_test = x[15]

    verbose = False

    labtrans_PCHazard = Pycox.models.PCHazard.label_transform(num_durations)
    get_target = lambda df: (df['time'].values, df['status'].values)
    y_train = labtrans_PCHazard.fit_transform(*get_target(df_train))
    y_val = labtrans_PCHazard.transform(*get_target(df_valid))
    val = (x_valid, y_val)
    durations_test, events_test = get_target(df_test)
    in_features = x_train.shape[1]
    out_features = labtrans_PCHazard.out_features
    net = tt.practical.MLPVanilla(in_features=in_features, num_nodes=num_nodes, out_features=out_features,
                                  batch_norm=BATCH_NORM, dropout=DROPOUT)
    model_PCHazard = Pycox.models.PCHazard(net, tt.optim.Adam, duration_index=labtrans_PCHazard.cuts)
    model_PCHazard.optimizer.set_lr(LR)
    callbacks = [tt.callbacks.EarlyStopping(patience=Patience)]
    log = model_PCHazard.fit(x_train, y_train, batch_size, epochs,
                             callbacks, val_data=val, verbose=verbose)
    model_PCHazard.sub = 10
    surv = model_PCHazard.predict_surv_df(x_test)
    return (concDisc(surv, np.array(durations_test), np.array(events_test)))


################################################################################


################################################################################
## PCHazard method
def PMF_met(x):
    batch_size = x[0]
    DROPOUT = x[1]
    LR = x[2]
    num_nodes = x[3]
    BATCH_NORM = x[4]
    ACTIVATION = x[5]
    epochs = x[6]
    Patience = x[7]
    num_durations = x[8]
    return_conc = x[9]
    df_train = x[10]
    x_train = x[11]
    df_valid = x[12]
    x_valid = x[13]

    if return_conc == 'valid':
        df_test = x[12]
        x_test = x[13]
    else:
        df_test = x[14]
        x_test = x[15]

    verbose = False

    labtrans_PMF = Pycox.models.PMF.label_transform(num_durations)
    get_target = lambda df: (df['time'].values, df['status'].values)
    y_train = labtrans_PMF.fit_transform(*get_target(df_train))
    y_val = labtrans_PMF.transform(*get_target(df_valid))
    val = (x_valid, y_val)
    durations_test, events_test = get_target(df_test)
    in_features = x_train.shape[1]
    out_features = labtrans_PMF.out_features
    net = tt.practical.MLPVanilla(in_features=in_features, num_nodes=num_nodes, out_features=out_features,
                                  batch_norm=BATCH_NORM, dropout=DROPOUT)
    model_PMF = Pycox.models.PMF(net, tt.optim.Adam, duration_index=labtrans_PMF.cuts)
    model_PMF.optimizer.set_lr(LR)
    callbacks = [tt.callbacks.EarlyStopping(patience=Patience)]
    log = model_PMF.fit(x_train, y_train, batch_size, epochs, callbacks,
                        val_data=val, verbose=verbose)
    surv = model_PMF.predict_surv_df(x_test)
    return (concDisc(surv, np.array(durations_test), np.array(events_test)))


################################################################################



################################################################################
## Load the tabular dataset
df_all, x_all = initializer_Data(Data)


################################################################################




################################################################################
#############################################
## random splits of training/validation with CV t0 tune hyper-parameters
conc_RSF_hyper = np.empty((num_rep_hyper, num_folds, len(num_trees), len(num_nodes_rsf), len(m_tries)), dtype="float64")
conc_RSF_hyper[:] = np.nan
conc_bigSurvMLP_hyper = np.empty((num_rep_hyper, num_folds, len(batch_sizes), len(dropouts), len(LRs)), dtype="float64")
conc_bigSurvMLP_hyper[:] = np.nan
conc_CoxCC_hyper = np.empty((num_rep_hyper, num_folds, len(batch_sizes), len(dropouts), len(LRs)), dtype="float64")
conc_CoxCC_hyper[:] = np.nan
conc_CoxTime_hyper = np.empty((num_rep_hyper, num_folds, len(batch_sizes), len(dropouts), len(LRs)), dtype="float64")
conc_CoxTime_hyper[:] = np.nan
conc_DeepSurv_hyper = np.empty((num_rep_hyper, num_folds, len(batch_sizes), len(dropouts), len(LRs)), dtype="float64")
conc_DeepSurv_hyper[:] = np.nan
conc_DeepHit_hyper = np.empty((num_rep_hyper, num_folds, len(batch_sizes), len(dropouts), len(LRs)), dtype="float64")
conc_DeepHit_hyper[:] = np.nan
conc_MTLR_hyper = np.empty((num_rep_hyper, num_folds, len(batch_sizes), len(dropouts), len(LRs)), dtype="float64")
conc_MTLR_hyper[:] = np.nan
conc_PMF_hyper = np.empty((num_rep_hyper, num_folds, len(batch_sizes), len(dropouts), len(LRs)), dtype="float64")
conc_PMF_hyper[:] = np.nan
conc_PCHazard_hyper = np.empty((num_rep_hyper, num_folds, len(batch_sizes), len(dropouts), len(LRs)), dtype="float64")
conc_PCHazard_hyper[:] = np.nan

return_conc = 'valid'
ma_weight = 0.8
scheme = 'quantiles'

# methods = ['cox', 'bigSurv', 'RSF', 'bigSurvMLP', 'CoxCC', 'DeepSurv', 'CoxTime', 'DeepHit', 'MTLR', 'PCHazard', 'PMF']

# A for loop for a single split of training/testing data
for i_rep in range(num_rep_hyper):
    seed_num = 10000 * i_rep
    np.random.seed(seed_num)

    # divide data into training and testing
    indexTrain = np.arange(0, x_all.shape[0])
    indexTest = np.random.choice(indexTrain, size=np.int(np.floor(0.2 * len(indexTrain))), replace=False)
    indexTrain = np.setdiff1d(indexTrain, indexTest)
    df_train = df_all.loc[indexTrain]
    df_test = df_all.loc[indexTest]
    x_train = x_all[indexTrain, :]
    x_test = x_all[indexTest, :]

    # Indices for k-fold CV
    indices_fold = np.random.choice(len(indexTrain),
                                    size=np.int(num_folds * np.floor(len(indexTrain) / num_folds)),
                                    replace=False, p=None)
    indices_fold_mat = indices_fold.reshape(np.int(np.floor(len(indexTrain) / num_folds)),
                                            num_folds)
    for i_cv in range(num_folds):
        df_valCV = df_train.iloc[indices_fold_mat[:, i_cv]]
        df_trainCV = df_train.iloc[np.setdiff1d(indices_fold, indices_fold_mat[:, i_cv])]
        x_valCV = x_train[indices_fold_mat[:, i_cv], :]
        x_trainCV = x_train[np.setdiff1d(indices_fold, indices_fold_mat[:, i_cv]), :]

        for i_tree in range(len(num_trees)):
            num_tree = num_trees[i_tree]
            for i_node in range(len(num_nodes_rsf)):
                num_node_rsf = num_nodes_rsf[i_node]
                for i_try in range(len(m_tries)):
                    num_try = m_tries[i_try]
                    params = [num_tree, num_node_rsf, num_try, return_conc,
                              df_trainCV, df_valCV, None]
                    conc_RSF_hyper[i_rep, i_cv, i_tree, i_node, i_try] = rsfConcPy(params)

        for i_bs in range(len(batch_sizes)):
            batch_size = batch_sizes[i_bs]
            for i_dr in range(len(dropouts)):
                dropout = dropouts[i_dr]
                for i_lr in range(len(LRs)):
                    LR = LRs[i_lr]

                    params_bigSurvMLP = [batch_size, dropout, LR, num_nodes,
                                         BATCH_NORM, ACTIVATION, epochs, strata_size,
                                         epoch_test, if_early_stop, Patience, return_conc,
                                         df_trainCV, x_trainCV, df_valCV, x_valCV,
                                         df_test, x_test]
                    params_Cont = [batch_size, dropout, LR, num_nodes,
                                   BATCH_NORM, ACTIVATION, epochs, Patience,
                                   return_conc, df_trainCV, x_trainCV,
                                   df_valCV, x_valCV, df_test, x_test]
                    params_Disc = [batch_size, dropout, LR, num_nodes,
                                   BATCH_NORM, ACTIVATION, epochs, Patience,
                                   num_durations, return_conc, df_trainCV,
                                   x_trainCV, df_valCV, x_valCV, df_test, x_test]

                    conc_bigSurvMLP_hyper[i_rep, i_cv, i_bs, i_dr, i_lr] = bigSurvSGDMLP(params_bigSurvMLP)
                    conc_CoxCC_hyper[i_rep, i_cv, i_bs, i_dr, i_lr] = CoxCC_met(params_Cont)
                    conc_CoxTime_hyper[i_rep, i_cv, i_bs, i_dr, i_lr] = CoxTime_met(params_Cont)
                    conc_DeepSurv_hyper[i_rep, i_cv, i_bs, i_dr, i_lr] = DeepSurv_met(params_Cont)
                    conc_DeepHit_hyper[i_rep, i_cv, i_bs, i_dr, i_lr] = DeepHit_met(params_Disc)
                    conc_MTLR_hyper[i_rep, i_cv, i_bs, i_dr, i_lr] = MTLR_met(params_Disc)
                    conc_PMF_hyper[i_rep, i_cv, i_bs, i_dr, i_lr] = PMF_met(params_Disc)
                    conc_PCHazard_hyper[i_rep, i_cv, i_bs, i_dr, i_lr] = PCHazard_met(params_Disc)



conc_RSF_hyper_mean = np.nanmean(np.nanmean(conc_RSF_hyper, axis=0), axis=0)
best_indices_RSF = np.where(np.nanmax(conc_RSF_hyper_mean)==conc_RSF_hyper_mean)
print(best_indices_RSF)
num_tree_opt = num_trees[int(best_indices_RSF[0])]
num_node_opt = num_nodes_rsf[int(best_indices_RSF[1])]
num_try_opt = m_tries[int(best_indices_RSF[2])]

conc_bigSurvMLP_hyper_mean = np.nanmean(np.nanmean(conc_bigSurvMLP_hyper, axis=0), axis=0)
best_indices_bigSurvMLP = np.where(np.nanmax(conc_bigSurvMLP_hyper_mean)==conc_bigSurvMLP_hyper_mean)
batch_size_bigSurvMLP_opt = batch_sizes[int(best_indices_bigSurvMLP[0])]
dropout_bigSurvMLP_opt = dropouts[int(best_indices_bigSurvMLP[1])]
LR_bigSurvMLP_opt = LRs[int(best_indices_bigSurvMLP[2])]

conc_CoxCC_hyper_mean = np.nanmean(np.nanmean(conc_CoxCC_hyper, axis=0), axis=0)
best_indices_CoxCC = np.where(np.nanmax(conc_CoxCC_hyper_mean)==conc_CoxCC_hyper_mean)
batch_size_CoxCC_opt = batch_sizes[int(best_indices_CoxCC[0])]
dropout_CoxCC_opt = dropouts[int(best_indices_CoxCC[1])]
LR_CoxCC_opt = LRs[int(best_indices_CoxCC[2])]

conc_CoxTime_hyper_mean = np.nanmean(np.nanmean(conc_CoxTime_hyper, axis=0), axis=0)
best_indices_CoxTime = np.where(np.nanmax(conc_CoxTime_hyper_mean)==conc_CoxTime_hyper_mean)
batch_size_CoxTime_opt = batch_sizes[int(best_indices_CoxTime[0])]
dropout_CoxTime_opt = dropouts[int(best_indices_CoxTime[1])]
LR_CoxTime_opt = LRs[int(best_indices_CoxTime[2])]

conc_DeepSurv_hyper_mean = np.nanmean(np.nanmean(conc_DeepSurv_hyper, axis=0), axis=0)
best_indices_DeepSurv = np.where(np.nanmax(conc_DeepSurv_hyper_mean)==conc_DeepSurv_hyper_mean)
batch_size_DeepSurv_opt = batch_sizes[int(best_indices_DeepSurv[0])]
dropout_DeepSurv_opt = dropouts[int(best_indices_DeepSurv[1])]
LR_DeepSurv_opt = LRs[int(best_indices_DeepSurv[2])]

conc_DeepHit_hyper_mean = np.nanmean(np.nanmean(conc_DeepHit_hyper, axis=0), axis=0)
best_indices_DeepHit = np.where(np.nanmax(conc_DeepHit_hyper_mean)==conc_DeepHit_hyper_mean)
batch_size_DeepHit_opt = batch_sizes[int(best_indices_DeepHit[0])]
dropout_DeepHit_opt = dropouts[int(best_indices_DeepHit[1])]
LR_DeepHit_opt = LRs[int(best_indices_DeepHit[2])]

conc_MTLR_hyper_mean = np.nanmean(np.nanmean(conc_MTLR_hyper, axis=0), axis=0)
best_indices_MTLR = np.where(np.nanmax(conc_MTLR_hyper_mean)==conc_MTLR_hyper_mean)
batch_size_MTLR_opt = batch_sizes[int(best_indices_MTLR[0])]
dropout_MTLR_opt = dropouts[int(best_indices_MTLR[1])]
LR_MTLR_opt = LRs[int(best_indices_MTLR[2])]

conc_PMF_hyper_mean = np.nanmean(np.nanmean(conc_PMF_hyper, axis=0), axis=0)
best_indices_PMF = np.where(np.nanmax(conc_PMF_hyper_mean)==conc_PMF_hyper_mean)
batch_size_PMF_opt = batch_sizes[int(best_indices_PMF[0])]
dropout_PMF_opt = dropouts[int(best_indices_PMF[1])]
LR_PMF_opt = LRs[int(best_indices_PMF[2])]

conc_PCHazard_hyper_mean = np.nanmean(np.nanmean(conc_PCHazard_hyper, axis=0), axis=0)
best_indices_PCHazard = np.where(np.nanmax(conc_PCHazard_hyper_mean)==conc_PCHazard_hyper_mean)
batch_size_PCHazard_opt = batch_sizes[int(best_indices_PCHazard[0])]
dropout_PCHazard_opt = dropouts[int(best_indices_PCHazard[1])]
LR_PCHazard_opt = LRs[int(best_indices_PCHazard[2])]


conc_coxph = np.empty((num_rep), dtype="float64")
conc_bigSurvSGDs2 = np.empty((num_rep), dtype="float64")
conc_bigSurvSGDs5 = np.empty((num_rep), dtype="float64")
conc_bigSurvSGDs10 = np.empty((num_rep), dtype="float64")
conc_bigSurvSGDs20 = np.empty((num_rep), dtype="float64")
conc_bigSurvSGDs50 = np.empty((num_rep), dtype="float64")

conc_RSF = np.empty((num_rep), dtype="float64")
conc_RSF[:] = np.nan
conc_bigSurvMLP = np.empty((num_rep_hyper, num_folds), dtype="float64")
conc_bigSurvMLP[:] = np.nan
conc_CoxCC = np.empty((num_rep_hyper, num_folds), dtype="float64")
conc_CoxCC[:] = np.nan
conc_CoxTime = np.empty((num_rep_hyper, num_folds), dtype="float64")
conc_CoxTime[:] = np.nan
conc_DeepSurv = np.empty((num_rep_hyper, num_folds), dtype="float64")
conc_DeepSurv[:] = np.nan
conc_DeepHit = np.empty((num_rep_hyper, num_folds), dtype="float64")
conc_DeepHit[:] = np.nan
conc_MTLR = np.empty((num_rep_hyper, num_folds), dtype="float64")
conc_MTLR[:] = np.nan
conc_PMF = np.empty((num_rep_hyper, num_folds), dtype="float64")
conc_PMF[:] = np.nan
conc_PCHazard = np.empty((num_rep_hyper, num_folds), dtype="float64")
conc_PCHazard[:] = np.nan

return_conc = 'test'
ma_weight = 0.8
scheme = 'quantiles'

# methods = ['cox', 'bigSurv', 'RSF', 'bigSurvMLP', 'CoxCC', 'DeepSurv', 'CoxTime', 'DeepHit', 'MTLR', 'PCHazard', 'PMF']

# A for loop for a single split of training/testing data
for i_rep in range(num_rep):
    ## Indices for k-fold CV
    np.random.seed(seed_num)

    # divide data into training and testing
    seed_num = 10000 * i_rep
    np.random.seed(seed_num)
    indexTrain = np.arange(0, x_all.shape[0])
    indexTest = np.random.choice(indexTrain, size=np.int(np.floor(0.2 * len(indexTrain))),
                                 replace=False)
    indexTrain = np.setdiff1d(indexTrain, indexTest)
    df_train = df_all.loc[indexTrain]
    df_test = df_all.loc[indexTest]
    x_train = x_all[indexTrain, :]
    x_test = x_all[indexTest, :]

    conc_coxph[i_rep] = coxPHConcPy([df_train, df_test])
    conc_bigSurvSGDs2[i_rep] = bigSurvConcPy([df_train, df_test, 2])
    conc_bigSurvSGDs5[i_rep] = bigSurvConcPy([df_train, df_test, 5])
    conc_bigSurvSGDs10[i_rep] = bigSurvConcPy([df_train, df_test, 10])
    conc_bigSurvSGDs20[i_rep] = bigSurvConcPy([df_train, df_test, 20])
    conc_bigSurvSGDs50[i_rep] = bigSurvConcPy([df_train, df_test, 50])

    params = [num_tree_opt, num_node_opt, num_try_opt, return_conc, df_trainCV, df_valCV,
              df_test]
    conc_RSF_hyper[i_rep] = rsfConcPy(params)

    indices_fold = np.random.choice(len(indexTrain),
                                    size=np.int(num_folds * np.floor(len(indexTrain) / num_folds)),
                                    replace=False, p=None)
    indices_fold_mat = indices_fold.reshape(np.int(np.floor(len(indexTrain) / num_folds)),
                                            num_folds)
    for i_cv in range(num_folds):
        df_valCV = df_train.iloc[indices_fold_mat[:, i_cv]]
        df_trainCV = df_train.iloc[np.setdiff1d(indices_fold, indices_fold_mat[:, i_cv])]
        x_valCV = x_train[indices_fold_mat[:, i_cv], :]
        x_trainCV = x_train[np.setdiff1d(indices_fold, indices_fold_mat[:, i_cv]), :]



        params_bigSurvMLP = [batch_size_bigSurvMLP_opt, dropout_bigSurvMLP_opt,
                             LR_bigSurvMLP_opt, num_nodes,
                             BATCH_NORM, ACTIVATION, epochs, strata_size,
                             epoch_test, if_early_stop, Patience, return_conc,
                             df_trainCV, x_trainCV, df_valCV, x_valCV,
                             df_test, x_test]
        params_CoxCC = [batch_size_CoxCC_opt, dropout_CoxCC_opt,
                        LR_CoxCC_opt, num_nodes,
                        BATCH_NORM, ACTIVATION, epochs, Patience,
                        return_conc, df_trainCV, x_trainCV, df_valCV,
                        x_valCV, df_test, x_test]
        params_CoxTime = [batch_size_CoxTime_opt, dropout_CoxTime_opt,
                          LR_CoxTime_opt, num_nodes,
                          BATCH_NORM, ACTIVATION, epochs, Patience,
                          return_conc, df_trainCV, x_trainCV, df_valCV,
                          x_valCV, df_test, x_test]
        params_DeepSurv = [batch_size_DeepSurv_opt, dropout_DeepSurv_opt,
                           LR_DeepSurv_opt, num_nodes,
                           BATCH_NORM, ACTIVATION, epochs, Patience,
                           return_conc, df_trainCV, x_trainCV, df_valCV,
                           x_valCV, df_test, x_test]
        params_DeepHit = [batch_size_DeepHit_opt, dropout_DeepHit_opt,
                          LR_DeepHit_opt, num_nodes,
                          BATCH_NORM, ACTIVATION, epochs, Patience,
                          num_durations, return_conc, df_trainCV, x_trainCV,
                          df_valCV, x_valCV, df_test, x_test]
        params_MTLR = [batch_size_MTLR_opt, dropout_MTLR_opt,
                       LR_MTLR_opt, num_nodes,
                       BATCH_NORM, ACTIVATION, epochs, Patience,
                       num_durations, return_conc, df_trainCV, x_trainCV,
                       df_valCV, x_valCV, df_test, x_test]
        params_PMF = [batch_size_PMF_opt, dropout_PMF_opt,
                      LR_PMF_opt, num_nodes,
                      BATCH_NORM, ACTIVATION, epochs, Patience,
                      num_durations, return_conc, df_trainCV, x_trainCV,
                      df_valCV, x_valCV, df_test, x_test]
        params_PCHazard = [batch_size_PCHazard_opt, dropout_PCHazard_opt,
                           LR_PCHazard_opt, num_nodes,
                           BATCH_NORM, ACTIVATION, epochs, Patience,
                           num_durations, return_conc, df_trainCV, x_trainCV,
                           df_valCV, x_valCV, df_test, x_test]

        conc_bigSurvMLP[i_rep, i_cv] = bigSurvSGDMLP(params_bigSurvMLP)
        conc_CoxCC[i_rep, i_cv] = CoxCC_met(params_CoxCC)
        conc_CoxTime[i_rep, i_cv] = CoxTime_met(params_CoxTime)
        conc_DeepSurv[i_rep, i_cv] = DeepSurv_met(params_DeepSurv)
        conc_DeepHit[i_rep, i_cv] = DeepHit_met(params_DeepHit)
        conc_MTLR[i_rep, i_cv] = MTLR_met(params_MTLR)
        conc_PMF[i_rep, i_cv] = PMF_met(params_PMF)
        conc_PCHazard[i_rep, i_cv] = PCHazard_met(params_PCHazard)

print('coxph(): ', np.mean(conc_coxph))
print('bigSurvSGDs2: ', np.mean(conc_bigSurvSGDs2))
print('bigSurvSGDs5: ', np.mean(conc_bigSurvSGDs5))
print('bigSurvSGDs10: ', np.mean(conc_bigSurvSGDs10))
print('bigSurvSGDs20: ', np.mean(conc_bigSurvSGDs20))
print('bigSurvSGDs50: ', np.mean(conc_bigSurvSGDs50))
print('RSF: ', np.mean(conc_RSF))
print('bigSurvMLP: ', np.mean(conc_bigSurvMLP))
print('CoxCC: ', np.mean(conc_CoxCC))
print('CoxTime: ', np.mean(conc_CoxTime))
print('DeepSurv: ', np.mean(conc_DeepSurv))
print('DeepHit: ', np.mean(conc_DeepHit))
print('MTLR: ', np.mean(conc_MTLR))
print('PMF: ', np.mean(conc_PMF))
print('PCHazard: ', np.mean(conc_PCHazard))