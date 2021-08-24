################################################################################
# Number of training/testing splits
startRep = 0
endRep = 99
num_rep = endRep - startRep + 1

# Number of folds for cross-validation
num_folds = 5


################################################################################


################################################################################
## data preprocessing and preparation ##
################################################################################
def initializer_params():
    ## hyperparameters for RSF
    num_trees = [100, 500]
    num_nodes_rsf = [5, 15]
    m_tries = [2, 3, 4, 5]
    hyperParams_RSF = []
    for i_tree in num_trees:
        for i_node in num_nodes_rsf:
            for i_try in m_tries:
                hyperParams_RSF.append([i_tree, i_node, i_try])

    # hyperparameters for BigSurvSGD
    hyperParams_BigSurv = [2, 5, 10, 20, 50]

    # Network architecture
    num_nodes = [32, 32]
    out_features = 1
    BATCH_NORM = False
    ACTIVATION = 'relu'
    dropouts = [0.0, 0.2, 0.4]  # 0.0, 0.1, 0.2, 0.3, 0.4, 0.5

    output_bias = False
    batch_sizes = [64, 256]

    strata_size = 2
    epochs = 200

    if_early_stop = True
    Patience = 5

    epoch_test = 5  # epoch step for calculating testing concordance

    num_durations = 100  # number of discrete intervals on time axis
    LRs = [0.0001, 0.001, 0.01]  # 0.1, 0.01, 0.001, 0.0001

    # hyperparameters for BigSurvMLP
    hyperParams_BigSurvMLP = []
    for i_bs in batch_sizes:
        for i_dr in dropouts:
            for i_lr in LRs:
                hyperParams_BigSurvMLP.append([i_bs, i_dr, i_lr, num_nodes, BATCH_NORM, ACTIVATION,
                                               epochs, strata_size, epoch_test, if_early_stop, Patience])

    # hyperparameters for other continuous methods
    hyperParams_Cont = []
    for i_bs in batch_sizes:
        for i_dr in dropouts:
            for i_lr in LRs:
                hyperParams_Cont.append([i_bs, i_dr, i_lr, num_nodes, BATCH_NORM,
                                         epochs, if_early_stop, Patience])

    # hyperparameters for other continuous methods
    hyperParams_Disc = []
    for i_bs in batch_sizes:
        for i_dr in dropouts:
            for i_lr in LRs:
                hyperParams_Disc.append([i_bs, i_dr, i_lr, num_nodes, BATCH_NORM,
                                         epochs, num_durations, if_early_stop, Patience])

    return hyperParams_RSF, hyperParams_BigSurv, hyperParams_BigSurvMLP, \
           hyperParams_Cont, hyperParams_Disc


################################################################################


################################################################################
## import modules ##
################################################################################
def initializer_modules():
    import tensorflow as tf
    import multiprocessing as mp
    import numpy as np
    import time
    import sklearn
    import random
    import os
    from sklearn.preprocessing import StandardScaler
    from sklearn_pandas import DataFrameMapper
    import torchtuples as tt
    import pandas as pd
    import pycox as Pycox
    return tf, mp, np, time, sklearn, random, os, StandardScaler, DataFrameMapper, tt, pd, Pycox


################################################################################


################################################################################
## data preprocessing and preparation ##
################################################################################
def initializer_Data(Data, Pycox, pd, StandardScaler, DataFrameMapper):
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
    import numpy as np
    import rpy2.robjects as robjects
    import rpy2.robjects.packages as rpackages
    from rpy2.robjects import pandas2ri
    pandas2ri.activate()
    # import R's "base" package
    base = rpackages.importr('base')
    # import R's utility package
    utils = rpackages.importr('utils')
    ## if 'survival' and 'randomForestSRC' are not installed, install them by de-commenting two below lines:
    # utils.install_packages('survival')
    # utils.install_packages('randomForestSRC')
    ## Load 'survival' and 'randomForestSRC' packages
    survival = rpackages.importr('survival')
    randomForestSRC = rpackages.importr('randomForestSRC')
    num_tree = x[0]
    num_node = x[1]
    m_try = x[2]
    df_train = x[3]
    df_valid = x[4]

    rsfConc = robjects.r('''
                        # A function to estimate concordance using random forest
                        concRandomForest <- function(df_train, df_valid, num_tree, num_node, m_try){
                            for (i in 1:ncol(df_valid)){
                                df_valid[,i] <- as.numeric(as.character(df_valid[,i]))
                                df_train[,i] <- as.numeric(as.character(df_train[,i]))
                            }
                            modRFSRC <- rfsrc(Surv(time, status) ~ ., data = df_train, ntree = num_tree, nodesize=num_node, mtry=m_try)
                            survival.results <- predict(modRFSRC, newdata = df_valid)$survival
                            median.survival <- apply(survival.results, 1, function(x) median(x))
                            as.numeric(median.survival)
                        }
            ''')
    try:
        f_beta = -rsfConc(df_train, df_valid, num_tree, num_node, m_try)
        times_valid = df_valid['time'].to_numpy(dtype="float64")
        events_valid = df_valid['status'].to_numpy(dtype="int64")
        orderedIndices = np.argsort(times_valid)
        Events = np.array(events_valid)[orderedIndices]
        f_beta = np.array(f_beta)[orderedIndices]
        Times = np.array(times_valid)[orderedIndices]
        return(calc_conc(f_beta, Times, Events))
    except:
        return(np.nan)


################################################################################


################################################################################
## Calling coxph package and claculate its concordance
def coxPHConcPy(x):
    import rpy2.robjects as robjects
    import rpy2.robjects.packages as rpackages
    from rpy2.robjects import pandas2ri
    pandas2ri.activate()
    ## import R's "base" package
    base = rpackages.importr('base')
    ## import R's utility package
    utils = rpackages.importr('utils')
    ## if 'survival' is not installed, install it by de-commenting below line:
    # utils.install_packages('survival')
    survival = rpackages.importr('survival')

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
    import numpy as np
    import rpy2.robjects as robjects
    import rpy2.robjects.packages as rpackages
    from rpy2.robjects import pandas2ri
    pandas2ri.activate()
    ## import R's "base" package
    base = rpackages.importr('base')
    ## import R's utility package
    utils = rpackages.importr('utils')
    ## if 'survival', 'parallel', 'doParallel', 'bigmemory', 'bigSurvSGD', and 'randomForestSRC' are not installed,
    ## install them by de-commenting below lines:
    # utils.install_packages('survival')
    # utils.install_packages('parallel')
    # utils.install_packages('doParallel')
    # utils.install_packages('bigmemory')
    # utils.install_packages('bigSurvSGD')
    bigSurvSGD = rpackages.importr('bigSurvSGD')
    survival = rpackages.importr('survival')
    bigmemory = rpackages.importr('bigmemory')
    doParallel = rpackages.importr('doParallel')
    parallel = rpackages.importr('parallel')

    df_train = x[1]
    df_test = x[2]
    strata_size = x[0]

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
    import tensorflow as tf
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
    import numpy as np
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
    import numpy as np
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
    import numpy as np
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
    import numpy as np
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
    import tensorflow as tf
    import numpy as np

    with tf.GradientTape() as tape:
        outs = mnist_model([image1, image2], training=True)
        loss_all = loss_object([outs[0], outs[1]])
        loss_value = tf.keras.backend.mean(loss_all, axis=0, keepdims=True)
        conc = np.exp(-loss_value)
        grads = tape.gradient(loss_value, mnist_model.trainable_variables)
        optimizer.apply_gradients(zip(grads, mnist_model.trainable_variables))
    return (conc)


################################################################################


################################################################################
## Customized smoothed concordence index
def loss_object(outs):
    import tensorflow as tf
    return (-tf.keras.backend.log(
        tf.keras.backend.exp(outs[0]) / (tf.keras.backend.exp(outs[0]) + tf.keras.backend.exp(outs[1]))))


################################################################################


################################################################################
## The main fucntion to read data and train network
def bigSurvSGDMLP(x):
    import numpy as np
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

    import tensorflow as tf

    optimizer = tf.keras.optimizers.Adam(learning_rate=LR, amsgrad=True)
    x_valid = np.expand_dims(x_valid, axis=2)
    x_train = np.expand_dims(x_train, axis=2)

    times_valid = df_valid['time'].to_numpy(dtype="float64")
    events_valid = df_valid['status'].to_numpy(dtype="int64")

    times_train = df_train['time'].to_numpy(dtype="float64")
    events_train = df_train['status'].to_numpy(dtype="int64")

    inputDim = x_train.shape[1]
    indices_train = np.arange(0, len(times_train))
    indices_valid = np.arange(0, len(times_valid))
    # Initialize model
    mnist_model = whole_model(inputDim, num_nodes, BATCH_NORM, ACTIVATION, DROPOUT)
    conc_valid = []
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
                results = train_step(x_train[Index1, :,:],
                                     x_train[Index2, :,:],
                                     mnist_model, optimizer)
            conc_train.append(results)
        ## consider the remaining strata that are not a complete batch
        if (indices.shape[0] % batch_size) > 0:
            Index1 = indices[(batch_size *
                              np.int(np.floor(indices.shape[0] / batch_size))):indices.shape[0], 0]
            Index2 = indices[(batch_size *
                              np.int(np.floor(indices.shape[0] / batch_size))):indices.shape[0], 1]
            results = train_step(x_train[Index1, :, :],
                                 x_train[Index2, :, :],
                                 mnist_model, optimizer)
            conc_train.append(results)
        conc_trainAll.append(np.nanmean(conc_train))
        #print("training conc: ", np.mean(conc_trainAll))
        if return_conc == 'valid':
            if (i_e + 1) % min(epoch_test, epochs) == 0:
                f_beta_valid = eval(x_valid, indices_valid, mnist_model)
                conc_valid.append(calc_conc(f_beta_valid, times_valid, events_valid))
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
        #print("validation conc: ", np.mean(conc_valid))
    if return_conc == 'valid':
        conc_return = np.nanmax(conc_valid)
        optEpoch = np.nanargmax(conc_valid)
        return [conc_return, (optEpoch+1)*epoch_test]
    else:
        f_beta_valid = eval(x_valid, indices_valid, mnist_model)
        return calc_conc(f_beta_valid, times_valid, events_valid)


################################################################################


################################################################################
## CoxCC method
def CoxCC_met(x):
    import pycox as Pycox
    import torchtuples as tt
    import numpy as np

    batch_size = x[0]
    DROPOUT = x[1]
    LR = x[2]
    num_nodes = x[3]
    BATCH_NORM = x[4]
    epochs = x[5]
    if_early_stop = x[6]
    Patience = x[7]
    return_conc = x[8]
    df_train = x[9]
    x_train = x[10]
    df_valid = x[11]
    x_valid = x[12]


    out_features = 1
    get_target = lambda df: (df['time'].values, df['status'].values)
    y_train = get_target(df_train)
    y_val = get_target(df_valid)
    durations_valid, events_valid = get_target(df_valid)
    val = tt.tuplefy(x_valid, y_val)
    in_features = x_train.shape[1]
    net = tt.practical.MLPVanilla(in_features, num_nodes, out_features,
                                  BATCH_NORM, DROPOUT, output_bias=False)
    model_CoxCC = Pycox.models.CoxCC(net, tt.optim.Adam)
    model_CoxCC.optimizer.set_lr(LR)

    if return_conc == 'valid':
        if if_early_stop:
            callbacks = [tt.callbacks.EarlyStopping(patience=Patience)]
            log = model_CoxCC.fit(x_train, y_train, batch_size, epochs=epochs,
                                  callbacks=callbacks, verbose=False,
                                  val_data=val.repeat(10).cat())
            try:
                val_loss_epochs = log.to_pandas()['val_loss']
                optEpochs = np.nanargmin(val_loss_epochs)
                optValLoss = np.nanmin(val_loss_epochs)
            except:
                optEpochs = np.nan
                optValLoss = np.nan
        else:
            log = model_CoxCC.fit(x_train, y_train, batch_size, epochs, verbose=False,
                              val_data=val.repeat(10).cat())
            try:
                val_loss_epochs = log.to_pandas()['val_loss']
                optEpochs = np.nanargmin(val_loss_epochs)
                optValLoss = np.nanmin(val_loss_epochs)
            except:
                optEpochs = np.nan
                optValLoss = np.nan
        return ([optValLoss, optEpochs+1])
    else:
        log = model_CoxCC.fit(x_train, y_train, batch_size, epochs, verbose=False)
        _ = model_CoxCC.compute_baseline_hazards()
        surv = model_CoxCC.predict_surv_df(x_test)
        return (concDisc(surv, np.array(durations_valid), np.array(events_valid)))


################################################################################


################################################################################
## DeepSurv method
def DeepSurv_met(x):
    import pycox as Pycox
    import torchtuples as tt
    import numpy as np

    batch_size = x[0]
    DROPOUT = x[1]
    LR = x[2]
    num_nodes = x[3]
    BATCH_NORM = x[4]
    epochs = x[5]
    if_early_stop = x[6]
    Patience = x[7]
    return_conc = x[8]
    df_train = x[9]
    x_train = x[10]
    df_valid = x[11]
    x_valid = x[12]

    out_features = 1
    get_target = lambda df: (df['time'].values, df['status'].values)
    y_train = get_target(df_train)
    y_val = get_target(df_valid)
    durations_valid, events_valid = get_target(df_valid)
    val = tt.tuplefy(x_valid, y_val)
    in_features = x_train.shape[1]
    net = tt.practical.MLPVanilla(in_features, num_nodes, out_features,
                                  BATCH_NORM, DROPOUT, output_bias=False)
    model_DeepSurv = Pycox.models.CoxPH(net, tt.optim.Adam)
    model_DeepSurv.optimizer.set_lr(LR)


    if return_conc == 'valid':
        if if_early_stop:
            callbacks = [tt.callbacks.EarlyStopping(patience=Patience)]
            log = model_DeepSurv.fit(x_train, y_train, batch_size, epochs=epochs,
                            callbacks=callbacks, verbose=False,
                            val_data=val.repeat(10).cat())
            try:
                val_loss_epochs = log.to_pandas()['val_loss']
                optEpochs = np.nanargmin(val_loss_epochs)
                optValLoss = np.nanmin(val_loss_epochs)
            except:
                optEpochs = np.nan
                optValLoss = np.nan
        else:
            log = model_DeepSurv.fit(x_train, y_train, batch_size, epochs, verbose=False,
                              val_data=val.repeat(10).cat())
            try:
                val_loss_epochs = log.to_pandas()['val_loss']
                optEpochs = np.nanargmin(val_loss_epochs)
                optValLoss = np.nanmin(val_loss_epochs)
            except:
                optEpochs = np.nan
                optValLoss = np.nan
        return ([optValLoss, optEpochs + 1])
    else:
        log = model_DeepSurv.fit(x_train, y_train, batch_size, epochs, verbose=False)
        _ = model_DeepSurv.compute_baseline_hazards()
        surv = model_DeepSurv.predict_surv_df(x_test)

        return (concDisc(surv, np.array(durations_valid), np.array(events_valid)))


################################################################################


################################################################################
## CoxTime method
def CoxTime_met(x):
    import pycox as Pycox
    import torchtuples as tt
    import numpy as np

    batch_size = x[0]
    DROPOUT = x[1]
    LR = x[2]
    num_nodes = x[3]
    BATCH_NORM = x[4]
    epochs = x[5]
    if_early_stop = x[6]
    Patience = x[7]
    return_conc = x[8]
    df_train = x[9]
    x_train = x[10]
    df_valid = x[11]
    x_valid = x[12]

    labtrans_CoxTime = Pycox.models.CoxTime.label_transform()
    get_target = lambda df: (df['time'].values, df['status'].values)
    y_train = labtrans_CoxTime.fit_transform(*get_target(df_train))
    y_val = labtrans_CoxTime.transform(*get_target(df_valid))
    durations_valid, events_valid = get_target(df_valid)
    val = tt.tuplefy(x_valid, y_val)
    in_features = x_train.shape[1]
    net = Pycox.models.cox_time.MLPVanillaCoxTime(in_features=in_features, num_nodes=num_nodes, batch_norm=BATCH_NORM,
                                                  dropout=DROPOUT)
    model_CoxTime = Pycox.models.CoxTime(net, tt.optim.Adam, labtrans=labtrans_CoxTime)
    model_CoxTime.optimizer.set_lr(LR)

    if return_conc == 'valid':
        if if_early_stop:
            callbacks = [tt.callbacks.EarlyStopping(patience=Patience)]
            log = model_CoxTime.fit(x_train, y_train, batch_size, epochs=epochs,
                                    callbacks=callbacks,
                                    verbose=False, val_data=val.repeat(10).cat())
            try:
                val_loss_epochs = log.to_pandas()['val_loss']
                optEpochs = np.nanargmin(val_loss_epochs)
                optValLoss = np.nanmin(val_loss_epochs)
            except:
                optEpochs = np.nan
                optValLoss = np.nan
        else:
            log = model_CoxTime.fit(x_train, y_train, batch_size, epochs,
                                    verbose=False, val_data=val.repeat(10).cat())
            try:
                val_loss_epochs = log.to_pandas()['val_loss']
                optEpochs = np.nanargmin(val_loss_epochs)
                optValLoss = np.nanmin(val_loss_epochs)
            except:
                optEpochs = np.nan
                optValLoss = np.nan
        return ([optValLoss, optEpochs + 1])
    else:
        log = model_CoxTime.fit(x_train, y_train, batch_size, epochs, verbose=False)
        _ = model_CoxTime.compute_baseline_hazards()
        surv = model_CoxTime.predict_surv_df(x_test)
        return (concDisc(surv, np.array(durations_valid), np.array(events_valid)))


################################################################################
## DeepHit method
def DeepHit_met(x):
    import pycox as Pycox
    import torchtuples as tt
    import numpy as np

    batch_size = x[0]
    DROPOUT = x[1]
    LR = x[2]
    num_nodes = x[3]
    BATCH_NORM = x[4]
    epochs = x[5]
    num_durations = x[6]
    if_early_stop = x[7]
    Patience = x[8]
    return_conc = x[9]
    df_train = x[10]
    x_train = x[11]
    df_valid = x[12]
    x_valid = x[13]


    labtrans_DeepHit = Pycox.models.DeepHitSingle.label_transform(num_durations)
    get_target = lambda df: (df['time'].values, df['status'].values)
    y_train = labtrans_DeepHit.fit_transform(*get_target(df_train))
    y_val = labtrans_DeepHit.transform(*get_target(df_valid))
    val = (x_valid, y_val)
    # We don't need to transform the test labels
    durations_valid, events_valid = get_target(df_valid)
    in_features = x_train.shape[1]
    out_features = labtrans_DeepHit.out_features

    net = tt.practical.MLPVanilla(in_features=in_features, num_nodes=num_nodes,
                                  out_features=out_features, batch_norm=BATCH_NORM,
                                  dropout=DROPOUT)
    model_DeepHit = Pycox.models.DeepHitSingle(net, tt.optim.Adam, alpha=0.2, sigma=0.1,
                                               duration_index=labtrans_DeepHit.cuts)
    model_DeepHit.optimizer.set_lr(LR)

    if return_conc == 'valid':
        if if_early_stop:
            callbacks = [tt.callbacks.EarlyStopping(patience=Patience)]
            log = model_DeepHit.fit(x_train, y_train, batch_size,epochs=epochs,
                                    callbacks=callbacks, val_data=val, verbose=False)
            try:
                val_loss_epochs = log.to_pandas()['val_loss']
                optEpochs = np.nanargmin(val_loss_epochs)
                optValLoss = np.nanmin(val_loss_epochs)
            except:
                optEpochs = np.nan
                optValLoss = np.nan
        else:
            log = model_DeepHit.fit(x_train, y_train, batch_size, epochs,
                                    val_data=val, verbose=False)
            try:
                val_loss_epochs = log.to_pandas()['val_loss']
                optEpochs = np.nanargmin(val_loss_epochs)
                optValLoss = np.nanmin(val_loss_epochs)
            except:
                optEpochs = np.nan
                optValLoss = np.nan
        return ([optValLoss, optEpochs + 1])
    else:
        log = model_DeepHit.fit(x_train, y_train, batch_size, epochs, verbose=False)
        surv = model_DeepHit.interpolate(10).predict_surv_df(x_test)
        return (concDisc(surv, np.array(durations_valid), np.array(events_valid)))


################################################################################


################################################################################
## MTLR method
def MTLR_met(x):
    import pycox as Pycox
    import torchtuples as tt
    import numpy as np

    batch_size = x[0]
    DROPOUT = x[1]
    LR = x[2]
    num_nodes = x[3]
    BATCH_NORM = x[4]
    epochs = x[5]
    num_durations = x[6]
    if_early_stop = x[7]
    Patience = x[8]
    return_conc = x[9]
    df_train = x[10]
    x_train = x[11]
    df_valid = x[12]
    x_valid = x[13]

    labtrans_MTLR = Pycox.models.MTLR.label_transform(num_durations)
    get_target = lambda df: (df['time'].values, df['status'].values)
    y_train = labtrans_MTLR.fit_transform(*get_target(df_train))
    y_val = labtrans_MTLR.transform(*get_target(df_valid))
    val = (x_valid, y_val)
    # We don't need to transform the test labels
    durations_valid, events_valid = get_target(df_valid)
    in_features = x_train.shape[1]
    out_features = labtrans_MTLR.out_features
    net = tt.practical.MLPVanilla(in_features=in_features, num_nodes=num_nodes, out_features=out_features,
                                  batch_norm=BATCH_NORM, dropout=DROPOUT)
    model_MTLR = Pycox.models.MTLR(net, tt.optim.Adam, duration_index=labtrans_MTLR.cuts)
    model_MTLR.optimizer.set_lr(LR)

    if return_conc == 'valid':
        if if_early_stop:
            callbacks = [tt.callbacks.EarlyStopping(patience=Patience)]
            log = model_MTLR.fit(x_train, y_train, batch_size, epochs=epochs,
                                 callbacks=callbacks, val_data=val, verbose=False)
            try:
                val_loss_epochs = log.to_pandas()['val_loss']
                optEpochs = np.nanargmin(val_loss_epochs)
                optValLoss = np.nanmin(val_loss_epochs)
            except:
                optEpochs = np.nan
                optValLoss = np.nan
        else:
            log = model_MTLR.fit(x_train, y_train, batch_size, epochs,
                                 val_data=val, verbose=False)
            try:
                val_loss_epochs = log.to_pandas()['val_loss']
                optEpochs = np.nanargmin(val_loss_epochs)
                optValLoss = np.nanmin(val_loss_epochs)
            except:
                optEpochs = np.nan
                optValLoss = np.nan
        return ([optValLoss, optEpochs+1])
    else:
        log = model_MTLR.fit(x_train, y_train, batch_size, epochs, verbose=False)
        surv = model_MTLR.predict_surv_df(x_test)
        return (concDisc(surv, np.array(durations_valid), np.array(events_valid)))


################################################################################


################################################################################
## PCHazard method
def PCHazard_met(x):
    import pycox as Pycox
    import torchtuples as tt
    import numpy as np

    batch_size = x[0]
    DROPOUT = x[1]
    LR = x[2]
    num_nodes = x[3]
    BATCH_NORM = x[4]
    epochs = x[5]
    num_durations = x[6]
    if_early_stop = x[7]
    Patience = x[8]
    return_conc = x[9]
    df_train = x[10]
    x_train = x[11]
    df_valid = x[12]
    x_valid = x[13]

    labtrans_PCHazard = Pycox.models.PCHazard.label_transform(num_durations)
    get_target = lambda df: (df['time'].values, df['status'].values)
    y_train = labtrans_PCHazard.fit_transform(*get_target(df_train))
    y_val = labtrans_PCHazard.transform(*get_target(df_valid))
    val = (x_valid, y_val)
    durations_valid, events_valid = get_target(df_valid)
    in_features = x_train.shape[1]
    out_features = labtrans_PCHazard.out_features
    net = tt.practical.MLPVanilla(in_features=in_features, num_nodes=num_nodes, out_features=out_features,
                                  batch_norm=BATCH_NORM, dropout=DROPOUT)
    model_PCHazard = Pycox.models.PCHazard(net, tt.optim.Adam, duration_index=labtrans_PCHazard.cuts)
    model_PCHazard.optimizer.set_lr(LR)

    if return_conc == 'valid':
        if if_early_stop:
            callbacks = [tt.callbacks.EarlyStopping(patience=Patience)]
            log = model_PCHazard.fit(x_train, y_train, batch_size, epochs=epochs,
                                     callbacks=callbacks, val_data=val, verbose=False)
            try:
                val_loss_epochs = log.to_pandas()['val_loss']
                optEpochs = np.nanargmin(val_loss_epochs)
                optValLoss = np.nanmin(val_loss_epochs)
            except:
                optEpochs = np.nan
                optValLoss = np.nan
        else:
            log = model_PCHazard.fit(x_train, y_train, batch_size, epochs,
                                     val_data=val, verbose=False)
            try:
                val_loss_epochs = log.to_pandas()['val_loss']
                optEpochs = np.nanargmin(val_loss_epochs)
                optValLoss = np.nanmin(val_loss_epochs)
            except:
                optEpochs = np.nan
                optValLoss = np.nan
        return ([optValLoss, optEpochs+1])
    else:
        log = model_PCHazard.fit(x_train, y_train, batch_size, epochs, verbose=False)
        model_PCHazard.sub = 10
        surv = model_PCHazard.predict_surv_df(x_test)
        return (concDisc(surv, np.array(durations_valid), np.array(events_valid)))


################################################################################


################################################################################
## PMF method
def PMF_met(x):
    import pycox as Pycox
    import torchtuples as tt
    import numpy as np

    batch_size = x[0]
    DROPOUT = x[1]
    LR = x[2]
    num_nodes = x[3]
    BATCH_NORM = x[4]
    epochs = x[5]
    num_durations = x[6]
    if_early_stop = x[7]
    Patience = x[8]
    return_conc = x[9]
    df_train = x[10]
    x_train = x[11]
    df_valid = x[12]
    x_valid = x[13]

    labtrans_PMF = Pycox.models.PMF.label_transform(num_durations)
    get_target = lambda df: (df['time'].values, df['status'].values)
    y_train = labtrans_PMF.fit_transform(*get_target(df_train))
    y_val = labtrans_PMF.transform(*get_target(df_valid))
    val = (x_valid, y_val)
    durations_valid, events_valid = get_target(df_valid)
    in_features = x_train.shape[1]
    out_features = labtrans_PMF.out_features
    net = tt.practical.MLPVanilla(in_features=in_features, num_nodes=num_nodes, out_features=out_features,
                                  batch_norm=BATCH_NORM, dropout=DROPOUT)
    model_PMF = Pycox.models.PMF(net, tt.optim.Adam, duration_index=labtrans_PMF.cuts)
    model_PMF.optimizer.set_lr(LR)


    if return_conc == 'valid':
        if if_early_stop:
            callbacks = [tt.callbacks.EarlyStopping(patience=Patience)]
            log = model_PMF.fit(x_train, y_train, batch_size, epochs=epochs,
                                callbacks=callbacks, val_data=val, verbose=False)
            try:
                val_loss_epochs = log.to_pandas()['val_loss']
                optEpochs = np.nanargmin(val_loss_epochs)
                optValLoss = np.nanmin(val_loss_epochs)
            except:
                optEpochs = np.nan
                optValLoss = np.nan
        else:
            log = model_PMF.fit(x_train, y_train, batch_size, epochs,
                                val_data=val, verbose=False)
            try:
                val_loss_epochs = log.to_pandas()['val_loss']
                optEpochs = np.nanargmin(val_loss_epochs)
                optValLoss = np.nanmin(val_loss_epochs)
            except:
                optEpochs = np.nan
                optValLoss = np.nan
        return ([optValLoss, optEpochs+1])
    else:
        log = model_PMF.fit(x_train, y_train, batch_size, epochs, verbose=False)
        surv = model_PMF.predict_surv_df(x_test)
        return (concDisc(surv, np.array(durations_valid), np.array(events_valid)))


################################################################################


################################################################################
## Multi-processing through multiprocessing packsge to speedup the analysis
if __name__ == '__main__':
    # Load required packages
    tf, mp, np, time, sklearn, random, os, StandardScaler, DataFrameMapper, tt, pd, Pycox = initializer_modules()
    datas = ['flchain', 'metabric', 'gbsg', 'nwtco', 'support']
    num_data = len(datas)
    # Initialize a numpy array for saving results
    allConc_coxph = np.empty((num_data, num_rep), dtype="float64")
    allConc_coxph[:] = np.nan
    allTime_coxph = np.empty((num_data, num_rep), dtype="float64")
    allTime_coxph[:] = np.nan

    allConc_bigSurvSGDs2 = np.empty((num_data, num_rep), dtype="float64")
    allConc_bigSurvSGDs2[:] = np.nan
    allTime_bigSurvSGDs2 = np.empty((num_data, num_rep), dtype="float64")
    allTime_bigSurvSGDs2[:] = np.nan

    allConc_bigSurvSGDs5 = np.empty((num_data, num_rep), dtype="float64")
    allConc_bigSurvSGDs5[:] = np.nan
    allTime_bigSurvSGDs5 = np.empty((num_data, num_rep), dtype="float64")
    allTime_bigSurvSGDs5[:] = np.nan

    allConc_bigSurvSGDs10 = np.empty((num_data, num_rep), dtype="float64")
    allConc_bigSurvSGDs10[:] = np.nan
    allTime_bigSurvSGDs10 = np.empty((num_data, num_rep), dtype="float64")
    allTime_bigSurvSGDs10[:] = np.nan

    allConc_bigSurvSGDs20 = np.empty((num_data, num_rep), dtype="float64")
    allConc_bigSurvSGDs20[:] = np.nan
    allTime_bigSurvSGDs20 = np.empty((num_data, num_rep), dtype="float64")
    allTime_bigSurvSGDs20[:] = np.nan

    allConc_bigSurvSGDs50 = np.empty((num_data, num_rep), dtype="float64")
    allConc_bigSurvSGDs50[:] = np.nan
    allTime_bigSurvSGDs50 = np.empty((num_data, num_rep), dtype="float64")
    allTime_bigSurvSGDs50[:] = np.nan

    allConc_bigSurvMLP = np.empty((num_data, num_rep), dtype="float64")
    allConc_bigSurvMLP[:] = np.nan
    allTime_bigSurvMLP = np.empty((num_data, num_rep), dtype="float64")
    allTime_bigSurvMLP[:] = 0.0

    allConc_RSF = np.empty((num_data, num_rep), dtype="float64")
    allConc_RSF[:] = np.nan
    allTime_RSF = np.empty((num_data, num_rep), dtype="float64")
    allTime_RSF[:] = 0.0

    allConc_CoxCC = np.empty((num_data, num_rep), dtype="float64")
    allConc_CoxCC[:] = np.nan
    allTime_CoxCC = np.empty((num_data, num_rep), dtype="float64")
    allTime_CoxCC[:] = 0.0

    allConc_CoxTime = np.empty((num_data, num_rep), dtype="float64")
    allConc_CoxTime[:] = np.nan
    allTime_CoxTime = np.empty((num_data, num_rep), dtype="float64")
    allTime_CoxTime[:] = 0.0

    allConc_DeepSurv = np.empty((num_data, num_rep), dtype="float64")
    allConc_DeepSurv[:] = np.nan
    allTime_DeepSurv = np.empty((num_data, num_rep), dtype="float64")
    allTime_DeepSurv[:] = 0.0

    allConc_DeepHit = np.empty((num_data, num_rep), dtype="float64")
    allConc_DeepHit[:] = np.nan
    allTime_DeepHit = np.empty((num_data, num_rep), dtype="float64")
    allTime_DeepHit[:] = 0.0

    allConc_MTLR = np.empty((num_data, num_rep), dtype="float64")
    allConc_MTLR[:] = np.nan
    allTime_MTLR = np.empty((num_data, num_rep), dtype="float64")
    allTime_MTLR[:] = 0.0

    allConc_PMF = np.empty((num_data, num_rep), dtype="float64")
    allConc_PMF[:] = np.nan
    allTime_PMF = np.empty((num_data, num_rep), dtype="float64")
    allTime_PMF[:] = 0.0

    allConc_PCHazard = np.empty((num_data, num_rep), dtype="float64")
    allConc_PCHazard[:] = np.nan
    allTime_PCHazard = np.empty((num_data, num_rep), dtype="float64")
    allTime_PCHazard[:] = 0.0

    # A for loop for a single split of training/testing data
    for i_rep in range(num_rep):
        print("======================================")
        print(i_rep+1)
        print("======================================")
        # import modules
        hyperParams_RSF, hyperParams_BigSurv, hyperParams_BigSurvMLP, \
        hyperParams_Cont, hyperParams_Disc = initializer_params()

        for i_data in range(len(datas)):
            Data = datas[i_data]
            print("======================================")
            print('Rep: ', i_rep + 1)
            print('Data: ', Data)
            print("======================================")
            # load data
            df_all, x_all = initializer_Data(Data, Pycox, pd, StandardScaler, DataFrameMapper)
            # divide data into training and testing
            seed_num = 10000 * (i_rep+startRep)
            np.random.seed(seed_num)
            indexTrain = np.arange(0, x_all.shape[0])
            indexTest = np.random.choice(indexTrain, size=np.int(np.floor(0.2 * len(indexTrain))), replace=False)
            indexTrain = np.setdiff1d(indexTrain, indexTest)
            df_train = df_all.loc[indexTrain]
            df_test = df_all.loc[indexTest]
            x_train = x_all[indexTrain, :]
            x_test = x_all[indexTest, :]

            allConc_RSF_hyper = np.empty((num_folds, len(hyperParams_RSF)), dtype="float64")
            allConc_RSF_hyper[:] = np.nan

            allConc_bigSurvMLP_hyper = np.empty((num_folds, len(hyperParams_BigSurvMLP)), dtype="float64")
            allConc_bigSurvMLP_hyper[:] = np.nan
            allEpoch_bigSurvMLP_hyper = np.empty((num_folds, len(hyperParams_BigSurvMLP)), dtype="float64")
            allEpoch_bigSurvMLP_hyper[:] = np.nan

            allConc_CoxCC_hyper = np.empty((num_folds, len(hyperParams_Cont)), dtype="float64")
            allConc_CoxCC_hyper[:] = np.nan
            allEpoch_CoxCC_hyper = np.empty((num_folds, len(hyperParams_Cont)), dtype="float64")
            allEpoch_CoxCC_hyper[:] = np.nan

            allConc_CoxTime_hyper = np.empty((num_folds, len(hyperParams_Cont)), dtype="float64")
            allConc_CoxTime_hyper[:] = np.nan
            allEpoch_CoxTime_hyper = np.empty((num_folds, len(hyperParams_Cont)), dtype="float64")
            allEpoch_CoxTime_hyper[:] = np.nan

            allConc_DeepSurv_hyper = np.empty((num_folds, len(hyperParams_Cont)), dtype="float64")
            allConc_DeepSurv_hyper[:] = np.nan
            allEpoch_DeepSurv_hyper = np.empty((num_folds, len(hyperParams_Cont)), dtype="float64")
            allEpoch_DeepSurv_hyper[:] = np.nan

            allConc_DeepHit_hyper = np.empty((num_folds, len(hyperParams_Disc)), dtype="float64")
            allConc_DeepHit_hyper[:] = np.nan
            allEpoch_DeepHit_hyper = np.empty((num_folds, len(hyperParams_Disc)), dtype="float64")
            allEpoch_DeepHit_hyper[:] = np.nan

            allConc_MTLR_hyper = np.empty((num_folds, len(hyperParams_Disc)), dtype="float64")
            allConc_MTLR_hyper[:] = np.nan
            allEpoch_MTLR_hyper = np.empty((num_folds, len(hyperParams_Disc)), dtype="float64")
            allEpoch_MTLR_hyper[:] = np.nan

            allConc_PMF_hyper = np.empty((num_folds, len(hyperParams_Disc)), dtype="float64")
            allConc_PMF_hyper[:] = np.nan
            allEpoch_PMF_hyper = np.empty((num_folds, len(hyperParams_Disc)), dtype="float64")
            allEpoch_PMF_hyper[:] = np.nan

            allConc_PCHazard_hyper = np.empty((num_folds, len(hyperParams_Disc)), dtype="float64")
            allConc_PCHazard_hyper[:] = np.nan
            allEpoch_PCHazard_hyper = np.empty((num_folds, len(hyperParams_Disc)), dtype="float64")
            allEpoch_PCHazard_hyper[:] = np.nan

            # coxph
            start_time = time.time()
            allConc_coxph[i_data, i_rep] = coxPHConcPy([df_train, df_test])
            allTime_coxph[i_data, i_rep] = time.time() - start_time
            print('conc for coxph and data ' + Data + ' : ', allConc_coxph[i_data, i_rep])
            print('time for coxph and data ' + Data + ' : ', allTime_coxph[i_data, i_rep])

            # bigSurvSGD with different starta_size
            # s=2
            start_time = time.time()
            allConc_bigSurvSGDs2[i_data, i_rep] = bigSurvConcPy([2, df_train, df_test])
            allTime_bigSurvSGDs2[i_data, i_rep] = time.time() - start_time
            print('conc for bigSurvSGDs2 and data ' + Data + ' : ', allConc_bigSurvSGDs2[i_data, i_rep])
            print('time for bigSurvSGDs2 and data ' + Data + ' : ', allTime_bigSurvSGDs2[i_data, i_rep])

            # s=5
            start_time = time.time()
            allConc_bigSurvSGDs5[i_data, i_rep] = bigSurvConcPy([5, df_train, df_test])
            allTime_bigSurvSGDs5[i_data, i_rep] = time.time() - start_time
            print('conc for bigSurvSGDs5 and data ' + Data + ' : ', allConc_bigSurvSGDs5[i_data, i_rep])
            print('time for bigSurvSGDs5 and data ' + Data + ' : ', allTime_bigSurvSGDs5[i_data, i_rep])

            # s=10
            start_time = time.time()
            allConc_bigSurvSGDs10[i_data, i_rep] = bigSurvConcPy([10, df_train, df_test])
            allTime_bigSurvSGDs10[i_data, i_rep] = time.time() - start_time
            print('conc for bigSurvSGDs10 and data ' + Data + ' : ', allConc_bigSurvSGDs10[i_data, i_rep])
            print('time for bigSurvSGDs10 and data ' + Data + ' : ', allTime_bigSurvSGDs10[i_data, i_rep])

            # s=20
            start_time = time.time()
            allConc_bigSurvSGDs20[i_data, i_rep] = bigSurvConcPy([20, df_train, df_test])
            allTime_bigSurvSGDs20[i_data, i_rep] = time.time() - start_time
            print('conc for bigSurvSGDs20 and data ' + Data + ' : ', allConc_bigSurvSGDs20[i_data, i_rep])
            print('time for bigSurvSGDs20 and data ' + Data + ' : ', allTime_bigSurvSGDs20[i_data, i_rep])

            # s=50
            start_time = time.time()
            allConc_bigSurvSGDs50[i_data, i_rep] = bigSurvConcPy([50, df_train, df_test])
            allTime_bigSurvSGDs50[i_data, i_rep] = time.time() - start_time
            print('conc for bigSurvSGDs50 and data ' + Data + ' : ', allConc_bigSurvSGDs50[i_data, i_rep])
            print('time for bigSurvSGDs50 and data ' + Data + ' : ', allTime_bigSurvSGDs50[i_data, i_rep])
            ## Indices for k-fold CV
            indices_fold = np.random.choice(len(indexTrain),
                                            size=np.int(num_folds * np.floor(len(indexTrain) / num_folds)),
                                            replace=False, p=None)
            indices_fold_mat = indices_fold.reshape(np.int(np.floor(len(indexTrain) / num_folds)),
                                                    num_folds)

            # Prepare data and hyperparameters for all folds
            for i_cv in range(num_folds):
                df_valCV = df_train.iloc[indices_fold_mat[:, i_cv]]
                df_trainCV = df_train.iloc[np.setdiff1d(indices_fold, indices_fold_mat[:, i_cv])]
                x_valCV = x_train[indices_fold_mat[:, i_cv], :]
                x_trainCV = x_train[np.setdiff1d(indices_fold, indices_fold_mat[:, i_cv]), :]

                hyperParams_RSF_list = []
                hyperParams_BigSurvMLP_list = []
                hyperParams_Cont_list = []
                hyperParams_Disc_list = []

                # hyperparams for RSF
                for i in range(len(hyperParams_RSF)):
                    hyperParams_RSF_single = hyperParams_RSF[i].copy()
                    hyperParams_RSF_single.append(df_trainCV)
                    hyperParams_RSF_single.append(df_valCV)
                    hyperParams_RSF_list.append(hyperParams_RSF_single)

                # hyperparams for bigSurvMLP
                for i in range(len(hyperParams_BigSurvMLP)):
                    hyperParams_BigSurvMLP_single = hyperParams_BigSurvMLP[i].copy()
                    hyperParams_BigSurvMLP_single.append('valid')
                    hyperParams_BigSurvMLP_single.append(df_trainCV)
                    hyperParams_BigSurvMLP_single.append(x_trainCV)
                    hyperParams_BigSurvMLP_single.append(df_valCV)
                    hyperParams_BigSurvMLP_single.append(x_valCV)
                    hyperParams_BigSurvMLP_list.append(hyperParams_BigSurvMLP_single)

                # hyperparams for continuous methods
                for i in range(len(hyperParams_Cont)):
                    hyperParams_Cont_single = hyperParams_Cont[i].copy()
                    hyperParams_Cont_single.append('valid')
                    hyperParams_Cont_single.append(df_trainCV)
                    hyperParams_Cont_single.append(x_trainCV)
                    hyperParams_Cont_single.append(df_valCV)
                    hyperParams_Cont_single.append(x_valCV)
                    hyperParams_Cont_list.append(hyperParams_Cont_single)

                # hyperparams for discrete methods
                for i in range(len(hyperParams_Disc)):
                    hyperParams_Disc_single = hyperParams_Disc[i].copy()
                    hyperParams_Disc_single.append('valid')
                    hyperParams_Disc_single.append(df_trainCV)
                    hyperParams_Disc_single.append(x_trainCV)
                    hyperParams_Disc_single.append(df_valCV)
                    hyperParams_Disc_single.append(x_valCV)
                    hyperParams_Disc_list.append(hyperParams_Disc_single)

                ################################################################################
                # RSF: concordance over folds and hyperparams

                start_time = time.time()
                p_RSF = mp.Pool()
                results_RSF_mp = p_RSF.map(rsfConcPy, hyperParams_RSF_list)
                p_RSF.close()
                allTime_RSF[i_data, i_rep] += time.time()-start_time
                allConc_RSF_hyper[i_cv, :] = np.squeeze(np.array(results_RSF_mp))

                ################################################################################
                # bigSurvMLP: concordance over folds and hyperparams
                start_time = time.time()
                p = mp.Pool()
                results_mp = p.map(bigSurvSGDMLP, hyperParams_BigSurvMLP_list)
                p.close()
                allTime_bigSurvMLP[i_data, i_rep] += time.time() - start_time

                allConc_bigSurvMLP_hyper[i_cv, :] = np.array([item[0] for item in results_mp])
                allEpoch_bigSurvMLP_hyper[i_cv, :] = np.array([item[1] for item in results_mp])

                ################################################################################
                # CoxCC: concordance over folds and hyperparams
                start_time = time.time()
                p = mp.Pool()
                results_mp = p.map(CoxCC_met, hyperParams_Cont_list)
                p.close()
                allTime_CoxCC[i_data, i_rep] += time.time() - start_time

                allConc_CoxCC_hyper[i_cv, :] = np.array([item[0] for item in results_mp])
                allEpoch_CoxCC_hyper[i_cv, :] = np.array([item[1] for item in results_mp])

                ################################################################################
                # DeepSurv: concordance over folds and hyperparams
                start_time = time.time()
                p = mp.Pool()
                results_mp = p.map(DeepSurv_met, hyperParams_Cont_list)
                p.close()
                allTime_DeepSurv[i_data, i_rep] += time.time() - start_time

                allConc_DeepSurv_hyper[i_cv, :] = np.array([item[0] for item in results_mp])
                allEpoch_DeepSurv_hyper[i_cv, :] = np.array([item[1] for item in results_mp])

                ################################################################################
                # CoxTime: concordance over folds and hyperparams
                start_time = time.time()
                p = mp.Pool()
                results_mp = p.map(CoxTime_met, hyperParams_Cont_list)
                p.close()
                allTime_CoxTime[i_data, i_rep] += time.time() - start_time

                allConc_CoxTime_hyper[i_cv, :] = np.array([item[0] for item in results_mp])
                allEpoch_CoxTime_hyper[i_cv, :] = np.array([item[1] for item in results_mp])

                ################################################################################
                # DeepHit: concordance over folds and hyperparams
                start_time = time.time()
                p = mp.Pool()
                results_mp = p.map(DeepHit_met, hyperParams_Disc_list)
                p.close()
                allTime_DeepHit[i_data, i_rep] += time.time() - start_time

                allConc_DeepHit_hyper[i_cv, :] = np.array([item[0] for item in results_mp])
                allEpoch_DeepHit_hyper[i_cv, :] = np.array([item[1] for item in results_mp])
                ################################################################################
                # MTLR: concordance over folds and hyperparams
                start_time = time.time()
                p = mp.Pool()
                results_mp = p.map(MTLR_met, hyperParams_Disc_list)
                p.close()
                allTime_MTLR[i_data, i_rep] += time.time() - start_time

                allConc_MTLR_hyper[i_cv, :] = np.array([item[0] for item in results_mp])
                allEpoch_MTLR_hyper[i_cv, :] = np.array([item[1] for item in results_mp])

                ################################################################################
                # PCHazard: concordance over folds and hyperparams
                start_time = time.time()
                p = mp.Pool()
                results_mp = p.map(PCHazard_met, hyperParams_Disc_list)
                p.close()
                allTime_PCHazard[i_data, i_rep] += time.time() - start_time

                allConc_PCHazard_hyper[i_cv, :] = np.array([item[0] for item in results_mp])
                allEpoch_PCHazard_hyper[i_cv, :] = np.array([item[1] for item in results_mp])

                ################################################################################
                # PMF: concordance over folds and hyperparams
                start_time = time.time()
                p = mp.Pool()
                results_mp = p.map(PMF_met, hyperParams_Disc_list)
                p.close()
                allTime_PMF[i_data, i_rep] += time.time() - start_time

                allConc_PMF_hyper[i_cv, :] = np.array([item[0] for item in results_mp])
                allEpoch_PMF_hyper[i_cv, :] = np.array([item[1] for item in results_mp])

            ################################################################################
            ################################################################################

            # Determine the best set of hyperparameters for for testing RSF
            best_hyper_index_RSF = np.nanargmax(np.nanmean(allConc_RSF_hyper, axis=0))
            best_hyper_RSF = hyperParams_RSF_list.copy()[best_hyper_index_RSF]
            best_hyper_RSF[3] = df_train
            best_hyper_RSF[4] = df_test

            start_time = time.time()
            allConc_RSF[i_data, i_rep] = rsfConcPy(best_hyper_RSF)
            allTime_RSF[i_data, i_rep] += time.time() - start_time
            print('conc for RSF and data '+ Data + ' : ', allConc_RSF[i_data, i_rep])
            print('time for RSF and data '+ Data + ' : ', allTime_RSF[i_data, i_rep])

            ################################################################################
            # Determine the best set of hyperparameters for testing bifSurvMLP
            best_hyper_index_bigSurvMLP = np.nanargmax(np.nanmean(allConc_bigSurvMLP_hyper, axis=0))
            best_hyper_bigSurvMLP = hyperParams_BigSurvMLP_list.copy()[best_hyper_index_bigSurvMLP]
            best_epoch_bigSurvMLP = np.nanmean(allEpoch_bigSurvMLP_hyper, axis=0)[best_hyper_index_bigSurvMLP]

            best_hyper_bigSurvMLP[6] = int(best_epoch_bigSurvMLP)
            best_hyper_bigSurvMLP[11] = 'test'
            best_hyper_bigSurvMLP[12] = df_train
            best_hyper_bigSurvMLP[13] = x_train
            best_hyper_bigSurvMLP[14] = df_test
            best_hyper_bigSurvMLP[15] = x_test

            start_time = time.time()
            allConc_bigSurvMLP[i_data, i_rep] = bigSurvSGDMLP(best_hyper_bigSurvMLP)
            allTime_bigSurvMLP[i_data, i_rep] += time.time() - start_time
            print('conc for bigSurvMLP and data '+ Data + ' : ', allConc_bigSurvMLP[i_data, i_rep])
            print('time for bigSurvMLP and data '+ Data + ' : ', allTime_bigSurvMLP[i_data, i_rep])

            ################################################################################
            # Determine the best set of hyperparameters for testin CoxCC
            best_hyper_index_CoxCC = np.nanargmin(np.nanmean(allConc_CoxCC_hyper, axis=0))
            best_hyper_CoxCC = hyperParams_Cont_list.copy()[best_hyper_index_CoxCC]
            best_epoch_CoxCC = np.nanmean(allEpoch_CoxCC_hyper, axis=0)[best_hyper_index_CoxCC]

            best_hyper_CoxCC[5] = int(best_epoch_CoxCC)
            best_hyper_CoxCC[8] = 'test'
            best_hyper_CoxCC[9] = df_train
            best_hyper_CoxCC[10] = x_train
            best_hyper_CoxCC[11] = df_test
            best_hyper_CoxCC[12] = x_test

            start_time = time.time()
            allConc_CoxCC[i_data, i_rep] = CoxCC_met(best_hyper_CoxCC)
            allTime_CoxCC[i_data, i_rep] += time.time() - start_time
            print('conc for CoxCC and data '+ Data + ' : ', allConc_CoxCC[i_data, i_rep])
            print('time for CoxCC and data '+ Data + ' : ', allTime_CoxCC[i_data, i_rep])

            ################################################################################
            # Determine the best set of hyperparameters for testing DeepSurv
            best_hyper_index_DeepSurv = np.nanargmin(np.nanmean(allConc_DeepSurv_hyper, axis=0))
            best_hyper_DeepSurv = hyperParams_Cont_list.copy()[best_hyper_index_DeepSurv]
            best_epoch_DeepSurv = np.nanmean(allEpoch_DeepSurv_hyper, axis=0)[best_hyper_index_DeepSurv]

            best_hyper_DeepSurv[5] = int(best_epoch_DeepSurv)
            best_hyper_DeepSurv[8] = 'test'
            best_hyper_DeepSurv[9] = df_train
            best_hyper_DeepSurv[10] = x_train
            best_hyper_DeepSurv[11] = df_test
            best_hyper_DeepSurv[12] = x_test

            start_time = time.time()
            allConc_DeepSurv[i_data, i_rep] = DeepSurv_met(best_hyper_DeepSurv)
            allTime_DeepSurv[i_data, i_rep] += time.time() - start_time
            print('conc for DeepSurv and data '+ Data + ' : ', allConc_DeepSurv[i_data, i_rep])
            print('time for DeepSurv and data '+ Data + ' : ', allTime_DeepSurv[i_data, i_rep])


            ################################################################################
            # Determine the best set of hyperparameters for testing CoxTime
            best_hyper_index_CoxTime = np.nanargmin(np.nanmean(allConc_CoxTime_hyper, axis=0))
            best_hyper_CoxTime = hyperParams_Cont_list.copy()[best_hyper_index_CoxTime]
            best_epoch_CoxTime = np.nanmean(allEpoch_CoxTime_hyper, axis=0)[best_hyper_index_CoxTime]

            best_hyper_CoxTime[5] = int(best_epoch_CoxTime)
            best_hyper_CoxTime[8] = 'test'
            best_hyper_CoxTime[9] = df_train
            best_hyper_CoxTime[10] = x_train
            best_hyper_CoxTime[11] = df_test
            best_hyper_CoxTime[12] = x_test

            start_time = time.time()
            allConc_CoxTime[i_data, i_rep] = CoxTime_met(best_hyper_CoxTime)
            allTime_CoxTime[i_data, i_rep] += time.time() - start_time
            print('conc for CoxTime and data '+ Data + ' : ', allConc_CoxTime[i_data, i_rep])
            print('time for CoxTime and data '+ Data + ' : ', allTime_CoxTime[i_data, i_rep])

            ################################################################################
            # Determine the best set of hyperparameters for testing DeepHit
            best_hyper_index_DeepHit = np.nanargmin(np.nanmean(allConc_DeepHit_hyper, axis=0))
            best_hyper_DeepHit = hyperParams_Disc_list.copy()[best_hyper_index_DeepHit]
            best_epoch_DeepHit = np.nanmean(allEpoch_DeepHit_hyper, axis=0)[best_hyper_index_DeepHit]

            best_hyper_DeepHit[5] = int(best_epoch_DeepHit)
            best_hyper_DeepHit[9] = 'test'
            best_hyper_DeepHit[10] = df_train
            best_hyper_DeepHit[11] = x_train
            best_hyper_DeepHit[12] = df_test
            best_hyper_DeepHit[13] = x_test

            start_time = time.time()
            allConc_DeepHit[i_data, i_rep] = DeepHit_met(best_hyper_DeepHit)
            allTime_DeepHit[i_data, i_rep] += time.time() - start_time
            print('conc for DeepHit and data '+ Data + ' : ', allConc_DeepHit[i_data, i_rep])
            print('time for DeepHit and data '+ Data + ' : ', allTime_DeepHit[i_data, i_rep])

            ################################################################################
            # Determine the best set of hyperparameters for testing MTLR
            best_hyper_index_MTLR = np.nanargmin(np.nanmean(allConc_MTLR_hyper, axis=0))
            best_hyper_MTLR = hyperParams_Disc_list.copy()[best_hyper_index_MTLR]
            best_epoch_MTLR = np.nanmean(allEpoch_MTLR_hyper, axis=0)[best_hyper_index_MTLR]

            best_hyper_MTLR[5] = int(best_epoch_MTLR)
            best_hyper_MTLR[9] = 'test'
            best_hyper_MTLR[10] = df_train
            best_hyper_MTLR[11] = x_train
            best_hyper_MTLR[12] = df_test
            best_hyper_MTLR[13] = x_test

            start_time = time.time()
            allConc_MTLR[i_data, i_rep] = MTLR_met(best_hyper_MTLR)
            allTime_MTLR[i_data, i_rep] += time.time() - start_time
            print('conc for MTLR and data '+ Data + ' : ', allConc_MTLR[i_data, i_rep])
            print('time for MTLR and data '+ Data + ' : ', allTime_MTLR[i_data, i_rep])

            ################################################################################
            # Determine the best set of hyperparameters for testing PCHazard
            best_hyper_index_PCHazard = np.nanargmin(np.nanmean(allConc_PCHazard_hyper, axis=0))
            best_hyper_PCHazard = hyperParams_Disc_list.copy()[best_hyper_index_PCHazard]
            best_epoch_PCHazard = np.nanmean(allEpoch_PCHazard_hyper, axis=0)[best_hyper_index_PCHazard]

            best_hyper_PCHazard[5] = int(best_epoch_PCHazard)
            best_hyper_PCHazard[9] = 'test'
            best_hyper_PCHazard[10] = df_train
            best_hyper_PCHazard[11] = x_train
            best_hyper_PCHazard[12] = df_test
            best_hyper_PCHazard[13] = x_test

            start_time = time.time()
            allConc_PCHazard[i_data, i_rep] = PCHazard_met(best_hyper_PCHazard)
            allTime_PCHazard[i_data, i_rep] += time.time() - start_time
            print('conc for PCHazard and data '+ Data + ' : ', allConc_PCHazard[i_data, i_rep])
            print('time for PCHazard and data '+ Data + ' : ', allTime_PCHazard[i_data, i_rep])

            ################################################################################
            # Determine the best set of hyperparameters for testing PMF
            best_hyper_index_PMF = np.nanargmin(np.nanmean(allConc_PMF_hyper, axis=0))
            best_hyper_PMF = hyperParams_Disc_list.copy()[best_hyper_index_PMF]
            best_epoch_PMF = np.nanmean(allEpoch_PMF_hyper, axis=0)[best_hyper_index_PMF]

            best_hyper_PMF[5] = int(best_epoch_PMF)
            best_hyper_PMF[9] = 'test'
            best_hyper_PMF[10] = df_train
            best_hyper_PMF[11] = x_train
            best_hyper_PMF[12] = df_test
            best_hyper_PMF[13] = x_test

            start_time = time.time()
            allConc_PMF[i_data, i_rep] = PMF_met(best_hyper_PMF)
            allTime_PMF[i_data, i_rep] += time.time() - start_time
            print('conc for PMF and data '+ Data + ' : ', allConc_PMF[i_data, i_rep])
            print('time for PMF and data '+ Data + ' : ', allTime_PMF[i_data, i_rep])

    print('==================================================================================')
    print('Average of testing concordance index for different methods and tabular datasets: ')
    print('Mean concordance index for CoxPH: ', np.nanmean(allConc_coxph, axis=1))
    print('Mean concordance index for bigSurvSGDs2 : ', np.nanmean(allConc_bigSurvSGDs2, axis=1))
    print('Mean concordance index for bigSurvSGDs5 : ', np.nanmean(allConc_bigSurvSGDs5, axis=1))
    print('Mean concordance index for bigSurvSGDs10 : ', np.nanmean(allConc_bigSurvSGDs10, axis=1))
    print('Mean concordance index for bigSurvSGDs20 : ', np.nanmean(allConc_bigSurvSGDs20, axis=1))
    print('Mean concordance index for bigSurvSGDs50 : ', np.nanmean(allConc_bigSurvSGDs50, axis=1))
    print('Mean concordance index for RSF : ', np.nanmean(allConc_RSF, axis=1))
    print('Mean concordance index for CoxCC : ', np.nanmean(allConc_CoxCC, axis=1))
    print('Mean concordance index for CoxTime : ', np.nanmean(allConc_CoxTime, axis=1))
    print('Mean concordance index for DeepSurv : ', np.nanmean(allConc_DeepSurv, axis=1))
    print('Mean concordance index for DeepHit : ', np.nanmean(allConc_DeepHit, axis=1))
    print('Mean concordance index for MTLR : ', np.nanmean(allConc_MTLR, axis=1))
    print('Mean concordance index for PMF : ', np.nanmean(allConc_PMF, axis=1))
    print('Mean concordance index for PCHazard : ', np.nanmean(allConc_PCHazard, axis=1))
    print('==================================================================================')