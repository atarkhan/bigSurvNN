################################################################################
## data preprocessing and preparation ##
################################################################################



################################################################################
## import required packages
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
import pycox as Pycox



## METABRIC
df_all = Pycox.datasets.metabric.read_df()
df_all.columns = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6',
                          'x7', 'x8', 'time', 'status']
print('Censoring for METABRIC dataset is: ', 1 - np.sum(df_all['status'] / len(df_all['status'])))

cols_standardize = ['x0', 'x1', 'x2', 'x3', 'x8']
cols_leave = ['x4', 'x5', 'x6', 'x7']
standardize = [([col], StandardScaler()) for col in cols_standardize]
leave = [(col, None) for col in cols_leave]
x_mapper = DataFrameMapper(standardize + leave)
x_all = x_mapper.fit_transform(df_all).astype('float32')

print(type(df_all))

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
print('Censoring for FLCHAIN dataset is: ', 1 - np.sum(df_all['status'] / len(df_all['status'])))

for i in range(9, len(df_all.columns)):
    df_all[df_all.columns[i]] = df_all[df_all.columns[i]].astype('int64')

cols_standardize = ['age', 'kappa', 'sample.yr', 'lambda', 'creatinine', 'mgus']
cols_leave = ['status', 'flc.grp_1', 'flc.grp_2', 'flc.grp_3', 'flc.grp_4',
                      'flc.grp_5', 'flc.grp_6', 'flc.grp_7', 'flc.grp_8', 'flc.grp_9']
standardize = [([col], StandardScaler()) for col in cols_standardize]
leave = [(col, None) for col in cols_leave]
x_mapper = DataFrameMapper(standardize + leave)
x_all = x_mapper.fit_transform(df_all).astype('float32')

## GBSG
df_all = Pycox.datasets.gbsg.read_df()
df_all.columns = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'time', 'status']
print('Cesnsoring for GBSG is: ', 1 - np.sum(df_all['status'] / len(df_all['status'])))
cols_standardize = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6']
cols_leave = []
standardize = [([col], StandardScaler()) for col in cols_standardize]
leave = [(col, None) for col in cols_leave]
x_mapper = DataFrameMapper(standardize + leave)
x_all = x_mapper.fit_transform(df_all).astype('float32')

## NWTCO
df_all = Pycox.datasets.nwtco.read_df()
df_all.columns = ['stage', 'age', 'in.subcohort', 'instit_2', 'histol_2',
                          'study_4', 'time', 'status']
print('Cesnsoring for NWTCO dataset is: ', 1 - np.sum(df_all['status'] / len(df_all['status'])))

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

## SUPPORT
df_all = Pycox.datasets.support.read_df()
df_all.columns = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6',
                          'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13',
                          'time', 'status']
print('Cesnsoring for SUPPORT dataset is: ', 1 - np.sum(df_all['status'] / len(df_all['status'])))
cols_standardize = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6',
                            'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13']
cols_leave = []
standardize = [([col], StandardScaler()) for col in cols_standardize]
leave = [(col, None) for col in cols_leave]
x_mapper = DataFrameMapper(standardize + leave)
x_all = x_mapper.fit_transform(df_all).astype('float32')

## simulated
num_features = 15
num_samples = 50
X = np.random.uniform(low=0, high=1, size=(num_samples, num_features))
beta = np.repeat(1, num_features).reshape((num_features, 1))
betaX = np.squeeze(np.matmul(X, beta))
p_c = 0.2
events = np.random.binomial(1, 1 - p_c, X.shape[0])
time_E = np.multiply(np.divide(-np.log(np.random.uniform(low=0, high=1, size=num_samples)),
                                       np.exp(betaX)), events)
time_C = np.multiply(np.divide(-np.log(np.random.uniform(low=0, high=1, size=num_samples)),
                                       np.exp(betaX)), 1 - events)

event_times = time_E + time_C
event_times = event_times.astype('float32')

events = events.astype('int32')

data = {'x0': X[:, 0], 'x1': X[:, 1], 'x2': X[:, 2], 'x3': X[:, 3], 'x4': X[:, 4],
                'x5': X[:, 5], 'x6': X[:, 6], 'x7': X[:, 7], 'x8': X[:, 8], 'x9': X[:, 9],
                'x10': X[:, 10], 'x11': X[:, 11], 'x12': X[:, 12], 'x13': X[:, 13], 'x14': X[:, 14],
                'time': event_times, 'status': events}
df_all = pd.DataFrame(data, columns=['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6',
                                             'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13',
                                             'x14', 'time', 'status'])
print('Cesnsoring for SIMULATED is: ', 1 - np.sum(df_all['status'] / len(df_all['status'])))

cols_standardize = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7',
                            'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14']
cols_leave = []
standardize = [([col], StandardScaler()) for col in cols_standardize]
leave = [(col, None) for col in cols_leave]
x_mapper = DataFrameMapper(standardize + leave)
x_all = x_mapper.fit_transform(df_all).astype('float32')