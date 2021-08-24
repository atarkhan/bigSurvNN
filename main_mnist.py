#############################################
## import all required packages
import numpy as np
import tensorflow as tf
import torchtuples as tt
from torchvision import datasets, transforms
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pycox.models import MTLR, PMF
from torch.utils.data.sampler import SubsetRandomSampler
#############################################



#############################################
## parameters
epochs = 1000  # maximum number of epochs
epoch_test = 5  # After any epoch_test epochs, we calculate validation accuracy (to save coputing time)
Patience = 5  # number of epochs continuing after reaching maximum validation concordance index
p_c = 0.2  # probability of censoring
num_channels = 1  # MNIST has only one grey channel
if_early_stop = True  # if we want to use early stopping
strata_size = 2  # number of patient per strata (s) for our framework- different from batch size
epoch_per_test = 5  # epoch step for which we calculate validation/test error
N_samples = [100, 1000, 10000]  # training sample size
batch_sizes = [64, 256]  # mini-batch sizes
dropouts = [0.0, 0.2, 0.4]  # dropout rates
LRs = [0.001]  # learning rates
L2s = [0, 1e-2, 1e-1]  # coefficient of L2 regularization penalty
num_filters = [32, 64, 64]
num_intervals = [10, 100]  # number of bins for MTLR and PMF
N_FC = 128  # number of nodes in last fully-connected layers for bigSurvCNN
ma_weight = 0.8  # moving average weight for stopping criterion of bigSurvCNN
etas = [0.1, 0.3, 0.5, 0.7, 0.9, 5]  # proportional constant for risk score
digit_ratio = np.repeat(0.1, 10)  # percentage of digits to use for validation and training data

#############################################



#############################################
## simulated time-to-event outcomes
def sim_event_times(mnist, eta, p_c=0.2):
    digits = mnist.targets.numpy()  # digit values
    censored = np.random.binomial(1, p_c, len(digits))  # censoring indices
    etaX = eta * digits  # risk score
    time_E = np.multiply(np.divide(-np.log(np.random.uniform(low=0, high=1, size=len(digits))),
                                   np.exp(etaX)), (1 - censored))
    time_C = np.multiply(np.divide(-np.log(np.random.uniform(low=0, high=1, size=len(digits))),
                                   np.exp(etaX)), censored)
    event_times = time_E + time_C
    censored = censored > 0
    return tt.tuplefy(event_times, ~censored)
#############################################


#############################################
## calculation of concordance index
def calc_conc(f_beta, Timess, Eventss):
    orderedIndices = np.argsort(Timess)  # order based on times
    Eventss = np.array(Eventss)[orderedIndices]
    f_beta = np.array(f_beta)[orderedIndices]
    Timess = np.array(Timess)[orderedIndices]
    conc_Bin = 0
    k = 0  # counts number of comparable pairs
    for i in range(len(Eventss) - 1):
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


#############################################


#############################################
## calculation of risk score for bigSurvCNN
def eval(x, y, mnist_model):
    B = 1
    S = 2
    f_beta = np.zeros((len(y)), dtype="float16")
    for i in range(np.int(len(y) / 2)):
        outs = mnist_model([np.expand_dims(x[y[i * B * S], :, :, :], axis=0),
                            np.expand_dims(x[y[(i * B * S + 1)], :, :, :], axis=0)],
                           training=False)
        f_beta[i * B * S] = np.squeeze(outs[0])
        f_beta[(i * B * S + 1)] = np.squeeze(outs[1])

    if len(y) % 2 == 1:
        Index1 = y[(len(y) - 1)]
        outs = mnist_model(np.expand_dims([x[Index1, :, :, :]], axis=0),
                           np.expand_dims([x[Index1, :, :, :]], axis=0), training=False)
        f_beta[(len(y) - 1)] = np.squeeze(outs[0])
    return (f_beta)


#############################################


#############################################
## steps for training: train the model using one pair of patients
def train_step(image1, image2, mnist_model, optimizer):
    with tf.GradientTape() as tape:
        outs = mnist_model([image1, image2], training=True)
        loss_all = loss_object(outs[1] - outs[0])
        conc1 = np.nanmean(np.exp(-loss_all))
        grads = tape.gradient(loss_all, mnist_model.trainable_variables)
        optimizer.apply_gradients(zip(grads, mnist_model.trainable_variables))
    return (conc1)


#############################################


#############################################
## Loss function
def loss_object(outDiff):
    return (tf.keras.backend.log(1 + tf.keras.backend.exp(outDiff)))


#############################################


#############################################
## bigSurvCNN network architecture
# network architecture for a single line
def single_model(num_channels, dropout, l2_p):
    inputs = tf.keras.Input(shape=(28, 28, num_channels))
    initializer = tf.random_normal_initializer()

    x = tf.keras.layers.Conv2D(32, (5, 5), padding="valid",
                               kernel_regularizer=tf.keras.regularizers.l1_l2(l2=L2, l1=0.0),
                               bias_initializer=initializer,
                               kernel_initializer=initializer)(inputs)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Conv2D(64, (5, 5), padding="valid",
                               kernel_regularizer=tf.keras.regularizers.l1_l2(l2=L2, l1=0.0),
                               bias_initializer=initializer,
                               kernel_initializer=initializer)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Conv2D(64, (5, 5), padding="same",
                               kernel_regularizer=tf.keras.regularizers.l1_l2(l2=L2, l1=0.0),
                               bias_initializer=initializer,
                               kernel_initializer=initializer)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(N_FC, kernel_regularizer=tf.keras.regularizers.l1_l2(l2=L2, l1=0.0),
                              bias_initializer=initializer,
                              kernel_initializer=initializer)(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l1_l2(l2=L2, l1=0.0),
                                    bias_initializer=initializer,
                                    kernel_initializer=initializer)(x)
    return tf.keras.Model(inputs, outputs)


# Our model uses two parallel networks (strata size of 2) that are exactly the same
def whole_model(num_channels, dropout, L2):
    inputs1 = tf.keras.Input(shape=(28, 28, num_channels))
    inputs2 = tf.keras.Input(shape=(28, 28, num_channels))
    singlemodel = single_model(num_channels, dropout, L2)
    outputs1 = singlemodel(inputs1)
    outputs2 = singlemodel(inputs2)
    return tf.keras.Model([inputs1, inputs2], [outputs1, outputs2])


#############################################


#############################################
## Simulatied data from MNIST. Read a single entry at a time.
class MnistSimDatasetSingle(Dataset):
    def __init__(self, mnist_dataset, time, event):
        self.mnist_dataset = mnist_dataset
        self.time, self.event = tt.tuplefy(time, event).to_tensor()

    def __len__(self):
        return len(self.mnist_dataset)

    def __getitem__(self, index):
        if type(index) is not int:
            raise ValueError(f"Need `index` to be `int`. Got {type(index)}.")
        img = self.mnist_dataset[index][0]
        return img, (self.time[index], self.event[index])


# Aggregate data for one mini-batch
def collate_fn(batch):
    return tt.tuplefy(batch).stack()


class MnistSimInput(Dataset):
    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset

    def __len__(self):
        return len(self.mnist_dataset)

    def __getitem__(self, index):
        img = self.mnist_dataset[index][0]
        return img
    #############################################


#############################################
## Network architechture used for MTLR and PMF
class Net(nn.Module):
    def __init__(self, out_features, dropout):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, 1, padding=1)
        self.max_pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout2d(dropout)
        self.conv2 = nn.Conv2d(32, 64, 5, 1, padding=1)
        self.max_pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout2d(dropout)
        self.conv3 = nn.Conv2d(64, 64, 5, 1, padding=1)
        self.max_pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout2d(dropout)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64, out_features)
        # self.fc2 = nn.Linear(256, out_features)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        # x = F.batch_norm(x, running_mean=0, running_var=1)
        x = self.max_pool(x)
        x = self.dropout(x)

        x = F.relu(self.conv2(x))
        # x = F.batch_norm(x, running_mean=0, running_var=1)
        x = self.max_pool(x)
        x = self.dropout(x)

        x = F.relu(self.conv3(x))
        # x = F.batch_norm(x, running_mean=0, running_var=1)
        x = self.max_pool(x)
        x = self.dropout(x)

        # x = self.glob_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        # x = self.fc2(x)
        return x


#############################################


#############################################
## Data preparation for discrete models (MTLR and PMF)
## Data preparation for discrete models
def dataPrepDisc(numInterval, MTLR, scheme, sim_train, mnist_train,
                 index_train, index_valid, batch_size):
    labtrans = MTLR.label_transform(numInterval, scheme)
    # Getting target values
    target_train = labtrans.fit_transform(*sim_train)
    # Training and testing data
    dataset_train = MnistSimDatasetSingle(mnist_train, *target_train)
    # Indices to choose a subset of training data
    indicesTrain = list([int(index) for index in index_train])
    if index_valid is not None:
        indicesValid = list([int(index) for index in index_valid])
        valid_sampler = SubsetRandomSampler(indicesValid)
        dl_valid = DataLoader(dataset_train, batch_size=1, collate_fn=collate_fn,
                              sampler=valid_sampler, shuffle=False)
        dataset_valid_x = MnistSimInput(mnist_train)
        dl_valid_x = DataLoader(dataset_valid_x, batch_size, sampler=valid_sampler, shuffle=False)

    else:
        dl_valid = None
        dl_valid_x = None
    train_sampler = SubsetRandomSampler(indicesTrain)

    # prepare mini-batch of trainingdata
    dl_train = DataLoader(dataset_train, batch_size, collate_fn=collate_fn,
                          sampler=train_sampler)
    return dl_train, dl_valid, dl_valid_x, labtrans


#############################################


#############################################
def concDisc(surv, Times, Events):
    times = list(np.array(surv.head(n=len(surv)).index, dtype="float64"))
    columns = list(surv.columns)
    median_surv_time = np.array([np.interp(0.5, xp=np.flip(np.array(surv[columns[i]], dtype="float64")),
                                           fp=np.sort(times)[::-1]) for i in range(len(columns))], dtype="float64")
    orderedIndices = np.argsort(Times)
    Eventss = np.array(Events)[orderedIndices]
    median_surv_time = median_surv_time[orderedIndices]
    Timess = np.array(Times)[orderedIndices]
    return (calc_conc(-median_surv_time, Timess, Eventss))
#############################################


#############################################
## The main fucntion to read data and train network
def bigSurvSGDNN(x):
    epochs = int(x[0])
    epoch_test = int(x[1])
    num_channels = int(x[2])
    batch_size = int(x[3])
    dropout = x[4]
    LR = x[5]
    L2 = x[6]
    strata_size = int(x[7])
    if_early_stop = x[8]
    Patience = int(x[9])
    return_conc = x[10]
    x_train = x[11]/256
    x_valid = x[12]/256
    times_train = x[14]
    times_valid = x[15]
    times_test = x[16]
    events_train = x[17]
    events_valid = x[18]
    events_test = x[19]
    ma_weight = x[20]

    x_train = x_train / 256
    x_valid = x_valid / 256
    if return_conc == 'test':
        x_test = x[13]/256

    startTime = time.time()
    optimizer = tf.keras.optimizers.Adam(learning_rate=LR, amsgrad=True)
    if return_conc == 'test':
        indices_test = np.arange(0, len(times_test))
    indices_train = np.arange(0, len(times_train))
    indices_valid = np.arange(0, len(times_valid))
    # Initialize model
    mnist_model = whole_model(num_channels, dropout, L2)
    conc_valid = []
    conc_test = []
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
        ## consider complete batches of starta
        if indices.shape[0] >= batch_size:
            for b in range(np.int(np.floor(indices.shape[0] / batch_size))):
                Index1 = indices[(b * batch_size):((b + 1) * batch_size), 0]
                Index2 = indices[(b * batch_size):((b + 1) * batch_size), 1]
                results = train_step(x_train[Index1, :, :, :],
                                     x_train[Index2, :, :, :],
                                     mnist_model, optimizer)
                conc_train.append(results)
        ## consider the remaining strata that are not a complete batch
        if (indices.shape[0] % batch_size) > 0:
            Index1 = indices[(batch_size * np.int(np.floor(indices.shape[0] / batch_size))):indices.shape[0], 0]
            Index2 = indices[(batch_size * np.int(np.floor(indices.shape[0] / batch_size))):indices.shape[0], 1]
            results = train_step(x_train[Index1, :, :, :],
                                 x_train[Index2, :, :, :],
                                 mnist_model, optimizer)
            conc_train.append(results)
        conc_trainAll.append(np.nanmean(conc_train))

        if return_conc == 'test':
            if i_e == (epochs-1):
                f_beta_test = eval(x_test, indices_test, mnist_model)
                conc_test = calc_conc(f_beta_test, times_test, events_test)
        else: # if return_conc == 'valid':
            if (i_e + 1) % min(epochs, epoch_test) == 0:
                f_beta_valid = eval(x_valid, indices_valid, mnist_model)
                conc_valid.append(calc_conc(f_beta_valid, times_valid, events_valid))

                if (if_early_stop & (i_e > (Patience * epoch_test))):
                    conc_validMA = np.empty((len(conc_valid)))
                    conc_validMA[:] = np.nan
                    conc_validMA[0] = conc_valid[0]
                    for k in range(1, len(conc_valid)):
                        weights = np.power(ma_weight, k - np.arange(k + 1))
                        conc_validMA[k] = np.ma.average(conc_valid[0:(k + 1)], weights=weights)
                    if np.prod(conc_validMA[len(conc_validMA) - 1] < conc_validMA[(len(conc_validMA) - Patience - 1):(
                            len(conc_validMA) - 1)]):
                        break
    if return_conc == 'test':
        conc_return = conc_test
        f_beta = f_beta_test
        optEpoch = epochs
    else:
        if if_early_stop == True:
            optEpoch = np.int((np.nanargmax(conc_valid) + 1) * epoch_test)
            conc_return = np.nanmax(conc_valid)
        else:
            optEpoch = epochs
            conc_return = conc_valid[len(conc_valid) - 1]
        f_beta = None

    endTime = time.time() - startTime
    return ([conc_return, optEpoch, endTime, conc_trainAll, f_beta])
#############################################


#############################################
## The main fucntion to read data and train network
def MTLR_CNN(x):
    dropout = x[0]
    LR = x[1]
    l2_p = x[2]
    epochs = x[3]
    if_early_stop = x[4]
    Patience = x[5]
    labtrans = x[6]
    return_conc = x[7]
    dl_train = x[8]
    dl_valid = x[9]
    dl_test_x = x[10]
    sim_test = x[11]

    out_features = labtrans.out_features
    net = Net(out_features, dropout)
    model = MTLR(net, tt.optim.Adam(LR, weight_decay=l2_p), duration_index=labtrans.cuts)

    if return_conc == 'valid':
        if if_early_stop:
            callbacks = [tt.callbacks.EarlyStopping(patience=Patience)]
            log = model.fit_dataloader(dl_train, epochs=epochs, callbacks=callbacks,
                                            verbose=False, val_dataloader=dl_valid)
            val_loss_epochs = log.to_pandas()['val_loss']
            optEpochs = np.nanargmin(val_loss_epochs)
            optValLoss = np.nanmin(val_loss_epochs)
        else:
            log = model.fit_dataloader(dl_train, epochs, verbose=False, val_dataloader=dl_valid)
            val_loss_epochs = log.to_pandas()['val_loss']
            optEpochs = np.nanargmin(val_loss_epochs)
            optValLoss = np.nanmin(val_loss_epochs)
        return ([optValLoss, optEpochs + 1])
    else:
        log = model.fit_dataloader(dl_train, epochs, verbose=False)
        surv = model.predict_surv_df(dl_test_x)
        return (concDisc(surv, np.array(sim_test[0]), np.array(sim_test[1])))


#############################################
## The main fucntion to read data and train network for MTLR
def PMF_CNN(x):
    dropout = x[0]
    LR = x[1]
    l2_p = x[2]
    epochs = x[3]
    if_early_stop = x[4]
    Patience = x[5]
    labtrans = x[6]
    return_conc = x[7]
    dl_train = x[8]
    dl_valid = x[9]
    dl_test_x = x[10]
    sim_test = x[11]

    out_features = labtrans.out_features
    net = Net(out_features, dropout)
    model = PMF(net, tt.optim.Adam(LR, weight_decay=l2_p), duration_index=labtrans.cuts)

    if return_conc == 'valid':
        if if_early_stop:
            callbacks = [tt.callbacks.EarlyStopping(patience=Patience)]
            log = model.fit_dataloader(dl_train, epochs=epochs, callbacks=callbacks,
                                       verbose=False, val_dataloader=dl_valid)
            val_loss_epochs = log.to_pandas()['val_loss']
            optEpochs = np.nanargmin(val_loss_epochs)
            optValLoss = np.nanmin(val_loss_epochs)
        else:
            log = model.fit_dataloader(dl_train, epochs, verbose=False, val_dataloader=dl_valid)
            val_loss_epochs = log.to_pandas()['val_loss']
            optEpochs = np.nanargmin(val_loss_epochs)
            optValLoss = np.nanmin(val_loss_epochs)
        return ([optValLoss, optEpochs + 1])
    else:
        log = model.fit_dataloader(dl_train, epochs, verbose=False)
        surv = model.predict_surv_df(dl_test_x)
        return (concDisc(surv, np.array(sim_test[0]), np.array(sim_test[1])))
#############################################


#############################################
# Data transformation for MNIST dataset
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.1307,), (0.3081,))])
# Read and transform MNIST training images
mnist_train = datasets.MNIST('.', train=True, download=True, transform=transform)
# Read and transform MNIST testing images
mnist_test = datasets.MNIST('.', train=False, download=True, transform=transform)
# Prepare testing datasets
dataset_test_x = MnistSimInput(mnist_test)
dl_test_x = DataLoader(dataset_test_x, 1, shuffle=False)
#############################################


#############################################
## 10 random splits of training/validation to tune hyper parameters
num_reps = 1
conc_PMF_hyper = np.empty((num_reps, len(N_samples), len(etas), len(batch_sizes), len(dropouts),
                              len(LRs), len(L2s), len(num_intervals)), dtype="float64")
conc_PMF_hyper[:] = np.nan

epoch_PMF_hyper = np.empty((num_reps, len(N_samples), len(etas), len(batch_sizes), len(dropouts),
                              len(LRs), len(L2s), len(num_intervals)), dtype="float64")
epoch_PMF_hyper[:] = np.nan

conc_MTLR_hyper = np.empty((num_reps, len(N_samples), len(etas), len(batch_sizes), len(dropouts),
                               len(LRs), len(L2s), len(num_intervals)), dtype="float64")
conc_MTLR_hyper[:] = np.nan

epoch_MTLR_hyper = np.empty((num_reps, len(N_samples), len(etas), len(batch_sizes), len(dropouts),
                               len(LRs), len(L2s), len(num_intervals)), dtype="float64")
epoch_MTLR_hyper[:] = np.nan

conc_bigSurv_hyper = np.empty((num_reps, len(N_samples), len(etas), len(batch_sizes), len(dropouts),
                                  len(LRs), len(L2s)), dtype="float64")
conc_bigSurv_hyper[:] = np.nan

epoch_bigSurv_hyper = np.empty((num_reps, len(N_samples), len(etas), len(batch_sizes), len(dropouts),
                                  len(LRs), len(L2s)), dtype="float64")
epoch_bigSurv_hyper[:] = np.nan

return_conc = 'valid'
ma_weight = 0.8
scheme = 'quantiles'

for i_rep in range(num_reps):
    seed_num = i_rep * 1000
    np.random.seed(seed_num)
    tf.random.set_seed(seed_num)
    torch.random.manual_seed(seed_num)

    # All training digit values
    allDigits = mnist_train.targets.numpy()
    allIndices_train = []
    allIndices_valid = []

    for i_n in range(len(N_samples)):
        N_sample = N_samples[i_n]
        N_train = int(0.8*N_sample)
        N_valid = N_sample - N_train
        for i in range(10):  # 10: number of digits
            N_sample_digit = int(digit_ratio[0] * N_sample)
            N_tr = int(digit_ratio[0] * N_train)
            indices_sample = list(np.random.choice([x for x in range(len(allDigits)) if allDigits[x] == i],
                                            size=N_sample_digit, replace=False))
            allIndices_train = np.append(allIndices_train, indices_sample[0:N_tr])
            allIndices_valid = np.append(allIndices_valid, indices_sample[N_tr:len(indices_sample)])

        index_train = np.array(allIndices_train, dtype=int)
        index_valid = np.array(allIndices_valid, dtype=int)

        x_train_all = np.expand_dims(mnist_train.data.numpy(), axis=3)
        x_train = x_train_all[index_train, :, :, :]
        x_valid = x_train_all[index_valid, :, :, :]

        for i_eta in range(len(etas)):
            eta = etas[i_eta]
            # Simulate time-to-event and event outcomes
            sim_test = sim_event_times(mnist_test, eta)
            sim_train = sim_event_times(mnist_train, eta)

            times_events_test = np.array(sim_test)
            times_test = times_events_test[0]
            events_test = times_events_test[1]
            x_test = np.expand_dims(mnist_test.data.numpy(), axis=3)

            times_events_train = np.array(sim_train)
            times_train_all = times_events_train[0]
            events_train_all = times_events_train[1]


            times_train = times_train_all[index_train]
            times_valid = times_train_all[index_valid]
            events_train = events_train_all[index_train]
            events_valid = events_train_all[index_valid]

            for i_bs in range(len(batch_sizes)):
                batch_size = batch_sizes[i_bs]
                for i_dr in range(len(dropouts)):
                    dropout = dropouts[i_dr]
                    for i_lr in range(len(LRs)):
                        LR = LRs[i_lr]
                        for i_l2 in range(len(L2s)):
                            L2 = L2s[i_l2]
                            params = [epochs, epoch_test, num_channels, batch_size, dropout, LR,
                                      L2, strata_size, if_early_stop, Patience, return_conc,
                                      x_train, x_valid, x_test, times_train, times_valid,
                                      times_test, events_train, events_valid, events_test, ma_weight]
                            results_bigSurv = bigSurvSGDNN(params)
                            conc_bigSurv_hyper[i_rep, i_n, i_eta, i_bs, i_dr, i_lr, i_l2] = results_bigSurv[0]
                            epoch_bigSurv_hyper[i_rep, i_n, i_eta, i_bs, i_dr, i_lr, i_l2] = results_bigSurv[1]

                            for i_num in range(len(num_intervals)):
                                num_interval = num_intervals[i_num]
                                dl_train, dl_valid, dl_valid_x, labtrans = dataPrepDisc(num_interval,MTLR,scheme,
                                                                                        sim_train,mnist_train,index_train,
                                                                                        index_valid,batch_size)
                                params = [dropout, LR, L2, epochs, if_early_stop, Patience, labtrans,
                                          return_conc, dl_train, dl_valid, dl_test_x, sim_test]
                                results_MTLR = MTLR_CNN(params)
                                conc_MTLR_hyper[i_rep, i_n, i_eta, i_bs, i_dr, i_lr, i_l2, i_num] = results_MTLR[0]
                                epoch_MTLR_hyper[i_rep, i_n, i_eta, i_bs, i_dr, i_lr, i_l2, i_num] = results_MTLR[1]

                                results_PMF = PMF_CNN(params)
                                conc_PMF_hyper[i_rep, i_n, i_eta, i_bs, i_dr, i_lr, i_l2, i_num] = results_PMF[0]
                                epoch_PMF_hyper[i_rep, i_n, i_eta, i_bs, i_dr, i_lr, i_l2, i_num] = results_PMF[1]
#############################################


#############################################
## Find the best hyperparameters over 10 random splits
# For MTLR
batch_size_MTLR = np.empty((num_reps, len(N_samples), len(etas)), dtype="float64")
batch_size_MTLR[:] = np.nan
dropout_MTLR = np.empty((num_reps, len(N_samples), len(etas)), dtype="float64")
dropout_MTLR[:] = np.nan
LR_MTLR = np.empty((num_reps, len(N_samples), len(etas)), dtype="float64")
LR_MTLR[:] = np.nan
L2_MTLR = np.empty((num_reps, len(N_samples), len(etas)), dtype="float64")
L2_MTLR[:] = np.nan
num_interval_MTLR = np.empty((num_reps, len(N_samples), len(etas)), dtype="float64")
num_interval_MTLR[:] = np.nan
epoch_MTLR = np.empty((num_reps, len(N_samples), len(etas)), dtype="int")
epoch_MTLR[:] = np.nan

# For MTLR
batch_size_PMF = np.empty((num_reps, len(N_samples), len(etas)), dtype="float64")
batch_size_PMF[:] = np.nan
dropout_PMF = np.empty((num_reps, len(N_samples), len(etas)), dtype="float64")
dropout_PMF[:] = np.nan
LR_PMF = np.empty((num_reps, len(N_samples), len(etas)), dtype="float64")
LR_PMF[:] = np.nan
L2_PMF = np.empty((num_reps, len(N_samples), len(etas)), dtype="float64")
L2_PMF[:] = np.nan
num_interval_PMF = np.empty((num_reps, len(N_samples), len(etas)), dtype="float64")
num_interval_PMF[:] = np.nan
epoch_PMF = np.empty((num_reps, len(N_samples), len(etas)), dtype="int")
epoch_PMF[:] = np.nan

# For bigSurv
batch_size_bigSurv = np.empty((num_reps, len(N_samples), len(etas)), dtype="float64")
batch_size_bigSurv[:] = np.nan
dropout_bigSurv = np.empty((num_reps, len(N_samples), len(etas)), dtype="float64")
dropout_bigSurv[:] = np.nan
LR_bigSurv = np.empty((num_reps, len(N_samples), len(etas)), dtype="float64")
LR_bigSurv[:] = np.nan
L2_bigSurv = np.empty((num_reps, len(N_samples), len(etas)), dtype="float64")
L2_bigSurv[:] = np.nan
epoch_bigSurv = np.empty((num_reps, len(N_samples), len(etas)), dtype="int")
epoch_bigSurv[:] = np.nan

for i_r in range(num_reps):
    for i_n in range(len(N_samples)):
        for i_eta in range(len(etas)):
            best_hypers_MTLR = np.where(np.nanmin(conc_MTLR_hyper[i_r, i_n, i_eta,:,:,:,:,:]) == conc_MTLR_hyper[i_r, i_n, i_eta,:,:,:,:,:])
            batch_size_MTLR[i_r, i_n,i_eta]=batch_sizes[int(best_hypers_MTLR[0])]
            dropout_MTLR[i_r, i_n,i_eta]=dropouts[int(best_hypers_MTLR[1])]
            LR_MTLR[i_r, i_n,i_eta]=LRs[int(best_hypers_MTLR[2])]
            L2_MTLR[i_r, i_n,i_eta]=L2s[int(best_hypers_MTLR[3])]
            num_interval_MTLR[i_r, i_n,i_eta]=num_intervals[int(best_hypers_MTLR[4])]
            epoch_MTLR[i_r, i_n,i_eta] = epoch_MTLR_hyper[i_r, i_n, i_eta,
                                         int(best_hypers_MTLR[0]),
                                         int(best_hypers_MTLR[1]),
                                         int(best_hypers_MTLR[2]),
                                         int(best_hypers_MTLR[3]),
                                         int(best_hypers_MTLR[4])]

            best_hypers_PMF = np.where(np.nanmin(conc_PMF_hyper[i_r, i_n, i_eta, :, :, :, :,:]) == conc_PMF_hyper[i_r, i_n, i_eta, :, :, :, :,:])
            batch_size_PMF[i_r, i_n,i_eta]=batch_sizes[int(best_hypers_PMF[0])]
            dropout_PMF[i_r, i_n,i_eta]=dropouts[int(best_hypers_PMF[1])]
            LR_PMF[i_r, i_n,i_eta]=LRs[int(best_hypers_PMF[2])]
            L2_PMF[i_r, i_n,i_eta]=L2s[int(best_hypers_PMF[3])]
            num_interval_PMF[i_r, i_n,i_eta] = num_intervals[int(best_hypers_PMF[4])]
            epoch_PMF[i_r, i_n, i_eta] = epoch_PMF_hyper[i_r, i_n, i_eta,
                                                           int(best_hypers_PMF[0]),
                                                           int(best_hypers_PMF[1]),
                                                           int(best_hypers_PMF[2]),
                                                           int(best_hypers_PMF[3]),
                                                           int(best_hypers_PMF[4])]

            best_hypers_bigSurv = np.where(np.nanmax(conc_bigSurv_hyper[i_r, i_n, i_eta, :, :, :,:]) == conc_bigSurv_hyper[i_r, i_n, i_eta, :, :, :,:])
            best_hypers_bigSurv = np.asarray(best_hypers_bigSurv)
            batch_size_bigSurv[i_r, i_n,i_eta]=batch_sizes[best_hypers_bigSurv[0][0].astype('int')]
            dropout_bigSurv[i_r, i_n,i_eta]=dropouts[best_hypers_bigSurv[1][0].astype('int')]
            LR_bigSurv[i_r, i_n,i_eta]=LRs[best_hypers_bigSurv[2][0].astype('int')]
            L2_bigSurv[i_r, i_n,i_eta]=L2s[best_hypers_bigSurv[3][0].astype('int')]
            epoch_bigSurv[i_r, i_n, i_eta] = epoch_PMF_hyper[i_r, i_n, i_eta,
                                                         int(best_hypers_bigSurv[0]),
                                                         int(best_hypers_bigSurv[1]),
                                                         int(best_hypers_bigSurv[2]),
                                                         int(best_hypers_bigSurv[3])]
#############################################


#############################################
## 10 random splits of training/validation to tune hyper parameters
num_reps = 1
conc_PMF_test = np.empty((num_reps, len(N_samples), len(etas)), dtype="float64")
conc_PMF_test[:] = np.nan
conc_MTLR_test = np.empty((num_reps, len(N_samples), len(etas)), dtype="float64")
conc_MTLR_test[:] = np.nan
conc_bigSurv_test = np.empty((num_reps, len(N_samples), len(etas)), dtype="float64")
conc_bigSurv_test[:] = np.nan
return_conc = 'test'
ma_weight = 0.8
scheme = 'quantiles'
f_betas_bigSurv = np.empty((num_reps, len(N_samples), len(etas), 10000), dtype="float64")
f_betas_bigSurv[:] = np.nan
digits = mnist_test.targets.numpy()  # digit values

conc_oracle = np.empty((num_reps, len(N_samples), len(etas)), dtype="float64")

for i_rep in range(num_reps):
    seed_num = i_rep * 1000
    np.random.seed(seed_num)
    tf.random.set_seed(seed_num)
    torch.random.manual_seed(seed_num)

    # All training digit values
    allDigits = mnist_train.targets.numpy()
    allIndices_train = []
    allIndices_valid = []

    for i_n in range(len(N_samples)):
        N_sample = N_samples[i_n]
        N_train = int(0.8*N_sample)
        N_valid = N_samples[i_n]-N_train

        for i in range(10):  # 10: number of digits
            N_sample_digit = int(digit_ratio[0] * N_sample)
            N_tr = int(digit_ratio[0] * N_train)
            indices = list(np.random.choice([x for x in range(len(allDigits)) if allDigits[x] == i],
                                        size=N_sample_digit, replace=False))
            allIndices_train = np.append(allIndices_train, indices[0:N_tr])
            allIndices_valid = np.append(allIndices_valid, indices[N_tr:len(indices)])

        index_train = np.array(allIndices_train, dtype=int)
        index_valid = np.array(allIndices_valid, dtype=int)

        for i_eta in range(len(etas)):
            eta = etas[i_eta]
            # Simulate time-to-event and event outcomes
            sim_test = sim_event_times(mnist_test, eta)
            sim_train = sim_event_times(mnist_train, eta)

            times_events_test = np.array(sim_test)
            times_test = times_events_test[0]
            events_test = times_events_test[1]
            x_test = np.expand_dims(mnist_test.data.numpy(), axis=3)

            conc_oracle[i_rep, i_n, i_eta] = calc_conc(digits, times_test, events_test)

            times_events_train = np.array(sim_train)
            times_train_all = times_events_train[0]
            events_train_all = times_events_train[1]

            x_train_all = np.expand_dims(mnist_train.data.numpy(), axis=3)
            x_train = x_train_all[index_train, :, :, :]
            x_valid = x_train_all[index_valid, :, :, :]
            times_train = times_train_all[index_train]
            times_valid = times_train_all[index_valid]
            events_train = events_train_all[index_train]
            events_valid = events_train_all[index_valid]
            print(epoch_bigSurv[i_rep, i_n, i_eta])
            params = [epoch_bigSurv[i_rep, i_n, i_eta], epoch_test, num_channels, int(batch_size_bigSurv[i_rep, i_n, i_eta]),
                      dropout_bigSurv[i_rep, i_n, i_eta], LR_bigSurv[i_rep, i_n, i_eta],  L2_bigSurv[i_rep, i_n, i_eta],
                      strata_size, if_early_stop, Patience, return_conc, x_train, x_valid, x_test, times_train,
                      times_valid, times_test, events_train, events_valid, events_test, ma_weight]
            results_bigSurv_test = bigSurvSGDNN(params)
            conc_bigSurv_test[i_rep, i_n, i_eta] = results_bigSurv_test[0]
            f_betas_bigSurv[i_rep, i_n, i_eta, :] = results_bigSurv_test[4]

            dl_train_MTLR, dl_valid_MTLR, dl_valid_x_MTLR, labtrans_MTLR = dataPrepDisc(int(num_interval_MTLR[i_rep, i_n, i_eta]), MTLR,
                                                                                                                         scheme, sim_train,
                                                                                                                        mnist_train, index_train,
                                                                                                                         index_valid,
                                                                                        int(batch_size_MTLR[i_rep, i_n, i_eta]))

            params = [dropout_MTLR[i_rep, i_n, i_eta], LR_MTLR[i_rep, i_n, i_eta], L2_MTLR[i_rep, i_n, i_eta],
                      epoch_MTLR[i_rep, i_n, i_eta], if_early_stop, Patience, labtrans_MTLR,return_conc,
                      dl_train_MTLR, dl_valid_MTLR, dl_test_x, sim_test]
            conc_MTLR_test[i_rep, i_n, i_eta] = MTLR_CNN(params)

            dl_train_PMF, dl_valid_PMF, dl_valid_x_PMF, labtrans_PMF = dataPrepDisc(int(num_interval_PMF[i_rep, i_n, i_eta]), MTLR,
                                                                                                                   scheme, sim_train, mnist_train,
                                                                                                                   index_train, index_valid,
                                                                                                                   int(batch_size_PMF[i_rep, i_n, i_eta]))
            params = [dropout_PMF[i_rep, i_n, i_eta], LR_PMF[i_rep, i_n, i_eta], L2_PMF[i_rep, i_n, i_eta],
                      epoch_PMF[i_rep, i_n, i_eta], if_early_stop, Patience, labtrans_PMF, return_conc,
                      dl_train_PMF, dl_valid_PMF, dl_test_x, sim_test]
            conc_PMF_test[i_rep, i_n, i_eta] = PMF_CNN(params)
#############################################



#############################################
## Find the best hyperparameters over 10 random splits
mean_rep_test_MTLR = np.nanmean(conc_MTLR_test, axis=0)
mean_rep_test_PMF = np.nanmean(conc_PMF_test, axis=0)
mean_rep_test_bigSurv = np.nanmean(conc_bigSurv_test, axis=0)
mean_rep_test_oracle = np.nanmean(np.nanmean(conc_oracle, axis=0), axis=0)
#############################################



#############################################
# Plot figure
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects import pandas2ri
pandas2ri.activate()
# import R's "base" package
base = rpackages.importr('base')
# import R's utility package
utils = rpackages.importr('utils')
plotFig5Main = robjects.r('''
                        # Plot and save Figure 5 in the main manuscript and Figure 3 in the Supplementary materials
                        plotFig5Main <- function(etas, mean_rep_test_bigSurv, mean_rep_test_MTLR, mean_rep_test_PMF, mean_rep_test_oracle){
                            # plots figure 5 in the main manuscript
                            pdf(file = "plot_balanced_mnist.pdf")
                            plot(etas, as.numeric(as.character(mean_rep_test_bigSurv[1,])), type = "l", pch=20, log="x", ylab = "conrodance", xlab=expression(eta), ylim = c(0.5,0.95), lwd=2, col="green")
                            lines(etas, as.numeric(as.character(mean_rep_test_bigSurv[2,])), lty=2, pch=0, lwd=2, col="green")
                            lines(etas, as.numeric(as.character(mean_rep_test_bigSurv[3,])), lty=4, pch=0, lwd=2, col="green")
                            lines(etas, as.numeric(as.character(mean_rep_test_MTLR[1,])), lty=1, pch=1, col='blue', lwd=2)
                            lines(etas, as.numeric(as.character(mean_rep_test_MTLR[2,])), lty=2, pch=1, col='blue', lwd=2)
                            lines(etas, as.numeric(as.character(mean_rep_test_MTLR[3,])), lty=4, pch=1, col='blue', lwd=2)
                            lines(etas, as.numeric(as.character(mean_rep_test_PMF[1,])), lty=1, pch=1, col='red', lwd=2)
                            lines(etas, as.numeric(as.character(mean_rep_test_PMF[2,])), lty=2, pch=1, col='red', lwd=2)
                            lines(etas, as.numeric(as.character(mean_rep_test_PMF[3,])), lty=4, pch=1, col='red', lwd=2)
                            lines(etas, mean_rep_test_oracle, lty=1, pch=1, col='black', lwd=2)
                            legend("bottomright", lty = c(rep(c(1,2,4),3), 1), 
                            col=c("green", "green", "green", "red","red","red", "blue","blue","blue", "black"),
                               c(expression(paste("bigSurvCNN, ",plain(n)[train] == 100)), 
                                        expression(paste("bigSurvCNN, ",plain(n)[train] == 1000)), 
                                                 expression(paste("bigSurvCNN, ",plain(n)[train] == 10000)),
                                                          expression(paste("PMF, ",plain(n)[train] == 100)), 
                                                                   expression(paste("PMF, ",plain(n)[train] == 1000)), 
                                                                            expression(paste("PMF, ",plain(n)[train] == 10000)),
                                                                                     expression(paste("MTLR, ",plain(n)[train] == 100)), 
                                                                                              expression(paste("MTLR, ",plain(n)[train] == 1000)), 
                                                                                                       expression(paste("MTLR, ",plain(n)[train] == 10000)),
                                                                                                                "Oracle"), lwd = rep(2,7),bty = "n", cex = 1)
                            dev.off()                            
                        }
''')
plotFig5Main(etas, mean_rep_test_bigSurv, mean_rep_test_MTLR, mean_rep_test_PMF, mean_rep_test_oracle)

# Predicted risk score for bigSurv
f_betas_N100_eta01 = f_betas_bigSurv[0,0,0,:]
f_betas_N100_eta50 = f_betas_bigSurv[0,0,len(etas),:]
f_betas_N10000_eta01 = f_betas_bigSurv[0,len(N_samples),0,:]
f_betas_N10000_eta50 = f_betas_bigSurv[0,len(N_samples),len(etas),:]

plotFig3Supp = robjects.r('''
                        # plots figure 3 in the supplementary materials
                        plotFig3Supp <- function(f_betas_N100_eta01, f_betas_N100_eta50, f_betas_N10000_eta01, f_betas_N10000_eta50, digits){
                            pdf(file="trueVSpredScore_jitter_LR.pdf")
                            par(mfrow=c(2,2))
                            f_betas_N100_eta01 = f_betas_N100_eta01 - mean(f_betas_N100_eta01)
                            linModel <- summary(lm(f_betas_N100_eta01~digits))$coef[,1]
                            xLM <- 0:9
                            yLM <- linModel[1] + linModel[2]*xLM
                            plot(f_betas_N100_eta01~jitter(digits,1), pch='.', cex=0.5,
                            main= expression(paste(plain(n)[train] == 100, "  and  ", eta == 0.1)),
                            xlab="digit value",
                            ylab="centered predicted score",
                            col="grey",
                            cex.main=1,
                            ylim = c(-15, 15))
                            lines(xLM, yLM, lty=1, col='blue')
                            legend(-1,15, paste0("slope = ", round(linModel[2],3)), cex = 1, bty = "n", 
                                bg = "white", text.col = 'blue')

                            f_betas_N100_eta50 = f_betas_N100_eta50 - mean(f_betas_N100_eta50)
                            linModel <- summary(lm(f_betas_N100_eta50~digits))$coef[,1]
                            yLM <- linModel[1] + linModel[2]*xLM
                            plot(f_betas_N100_eta50~jitter(digits,1), pch='.', cex=0.5,
                            main= expression(paste(plain(n)[train] == 100, "  and  ", eta == 5)),
                            xlab="digit value",
                            ylab="centered predicted score",
                            col="grey",
                            cex.main=1,
                            ylim = c(-15, 15))
                            lines(xLM, yLM, lty=1, col='blue')
                            legend(-1,15, paste0("slope = ", round(linModel[2],3)), cex = 1, bty = "n", 
                            bg = "white", text.col = 'blue')

                            f_betas_N10000_eta01 = f_betas_N10000_eta01 - mean(f_betas_N10000_eta01)
                            linModel <- summary(lm(f_betas_N10000_eta01~digits))$coef[,1]
                            yLM <- linModel[1] + linModel[2]*xLM
                            plot(f_betas_N10000_eta01~jitter(digits,1), pch='.', cex=0.5,
                                main= expression(paste(plain(n)[train] == 10000, "  and  ", eta == 0.1)),
                                xlab="digit value",
                                ylab="centered predicted score",
                                col="grey",
                                cex.main=1,
                                ylim = c(-15, 15))
                            lines(xLM, yLM, lty=1, col='blue')
                            legend(-1,15, paste0("slope = ", round(linModel[2],3)), cex = 1, bty = "n", 
                                bg = "white", text.col = 'blue')


                            f_betas_N10000_eta50 = f_betas_N10000_eta50 - mean(f_betas_N10000_eta50)
                            linModel <- summary(lm(f_betas_N10000_eta50~digits))$coef[,1]
                            yLM <- linModel[1] + linModel[2]*xLM
                            plot(f_betas_N10000_eta50~jitter(digits,1), pch='.', cex=0.5,
                                main= expression(paste(plain(n)[train] == 10000, "  and  ", eta == 5)),
                                xlab="digit value",
                                ylab="centered predicted score",
                                col="grey",
                                cex.main=1,
                                ylim = c(-15, 15))
                            lines(xLM, yLM, lty=1, col='blue')
                            legend(-1,15, paste0("slope = ", round(linModel[2],3)), cex = 1, bty = "n", 
                                        bg = "white", text.col = 'blue')

                            dev.off()                            
                        }
''')
plotFig3Supp(f_betas_N100_eta01, f_betas_N100_eta50, f_betas_N10000_eta01, f_betas_N10000_eta50, digits)