import itertools
import random
from operator import concat
import sys
import os
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import joblib
from math import sqrt
from scipy.stats import gaussian_kde
from sklearn.metrics import mean_squared_error, confusion_matrix, r2_score, classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import xgboost as xgb
from keras.layers import Dense
from keras import backend as K
from keras.models import Model
from keras.models import Sequential, load_model
from sklearn.inspection import permutation_importance
from pandas.testing import assert_frame_equal
from keras.layers import Dense, LSTM,Bidirectional, Dropout
from datetime import datetime
from sklearn.utils import shuffle
from scipy import stats

os.chdir('C:/Work/ORM/Data')


pd.set_option('display.max_rows',500)
pd.set_option('display.max_columns',500)


# data = pd.read_csv('processed_data/DEBW004_data.csv')
data = pd.read_csv('C:/Work/ORM/Data/clustered_data/cluster2_data.csv')
# print(data.describe())
# num_rows_to_train = int(len(data) * 4/5)
# data = data.iloc[:num_rows_to_train]

# # # Calculate the number of rows to select (4/5ths of the total rows)
# num_rows_to_select = int(len(data) * 4/5)

# data_train = data.iloc[:num_rows_to_select]
# data_train.to_csv("C:/Work/ORM/Data/clustered_data/cluster2_train_suffled.csv",index=False)

# data_test = data.iloc[num_rows_to_select:]
# data_test.to_csv("C:/Work/ORM/Data/clustered_data/cluster2_test_suffled.csv",index=False)

# data = data[(data['tg']>20)]
# data = data[(data['qq']>200)]
# data = data[(data['hu']<60)]
# data = data[(data['ozone']>120)]
# data = data[(data['Month']>6)&(data['Month']<10)]

# data['Time'] = pd.to_datetime(data['Time'])
# data = data.set_index('Time')
# data = data['2018']
# print(data.index)
print(data.describe())

# extreme_data = data[(data['ozone']>120)]
# print(extreme_data.describe())


values = data.values[:,:-1]

def custom_loss(preds, dtrain):
    y_true = dtrain.get_label()
    additional_weight = 1  # Adjust as needed
    weights = np.where(y_true > 120, additional_weight, 1)

    grad = (preds - y_true) * weights
    hess = weights

    return grad, hess

def custom_loss(preds, dtrain,ozone=values[:, 0],weight = 1):
    # Invert the normalization for predictions
    min_value = np.min(ozone)  # min value used in normalization
    max_value = np.max(ozone)  # max value used in normalization
    # unnormalized_preds = preds * (max_value - min_value) + min_value

    # Get the original, unnormalized labels
    y_true = dtrain.get_label()
    y_label = y_true*(max_value - min_value) + min_value

    # Additional weight for samples where original ozone concentration > 120
    additional_weight = weight
    weights = np.where(y_label > 120, additional_weight, 1)

    # grad = (unnormalized_preds - y_true) * weights
    grad = (preds - y_true) * weights
    hess = weights

    return grad, hess


def e_weight_mse(y_true, y_pred):
    return K.mean(K.exp(0  * y_true) * K.square(y_pred - y_true), axis=-1)


# scaled data
scaler = MinMaxScaler(feature_range=(0, 1))
values_scaled = scaler.fit_transform(values)

# Reshape-input data

def re_input(values,n_time,n_feature):
    v = values.shape[0] - n_time+1
    X = np.zeros([v,n_time*n_feature+1])
    for i in range(v):
        X[i,1:] = values[i:i+n_time,1:].reshape([1,-1])
        X[i,0] = values[i+n_time-1, 0]
    return X

def re_input_seq(values,n_time,n_feature):
    v = values.shape[0] - n_time+1
    X = np.zeros([v,n_time*(n_feature+1)])
    for i in range(v):
        X[i,n_time:] = values[i:i+n_time,1:].reshape([1,-1])
        X[i,0:n_time] = values[i:i+n_time, 0]
    return X


# Real-time split
def split_data_real(X,test_size,n_time,seq):
    train = X[:-test_size, :]
    test = X[-test_size:,:]
    # split into input and outputs
    if seq==0:
            train_X, train_y = train[:, 1:], train[:, 0]
            test_X, test_y = test[:, 1:], test[:, 0]
    else:
        train_X, train_y = train[:, n_time:], train[:, :n_time]
        test_X, test_y = test[:, n_time:], test[:, :n_time]
    # # reshape input to be 3D [samples, timesteps, features]
    # train_X = train_X.reshape((-1, n_time, n_feature))
    # test_X = test_X.reshape((-1, n_time, n_feature))
    return train_X, train_y, test_X, test_y

def station_based_split(X,site_num,site_order):
    sample_per = int(X.shape[0]/site_num)
    i = site_order
    X1 = X[:i*sample_per,:]
    X2 = X[(i+1)*sample_per:,:]
    y = X[i*sample_per:(i+1)*sample_per,:]
    train = np.concatenate([X1,X2], axis=0)
    test = y
    train_X, train_y = train[:, 1:], train[:, 0]
    test_X, test_y = test[:, 1:], test[:, 0]
    return train_X, train_y, test_X, test_y


def ReSampling(data,rp_time):
    sample = []
    for i in range(rp_time):
        sample.append(data[random.randint(0, len(data) - 1)])
    return np.array(sample)

# transform regression into classification function
def  rg_2_cl (x):
    y = copy.deepcopy(x)
    for i in range(y.shape[0]):
        if (y[i] < 120):
            y[i] = 0
        # elif (y[i] < 100):
        #     y[i] = 1
        else:
            y[i] = 1
    return y

def fit_model(train_X, train_y, test_X, test_y):
    # design network
    model = Sequential()
    # train_X = train_X.reshape((-1, 5, 7))
    # test_X = test_X.reshape((-1, 5, 7))
    # model.add(Bidirectional(LSTM(100, return_sequences=True,activation='relu'), input_shape=(train_X.shape[1], train_X.shape[2])),)
    # model.add(Bidirectional(LSTM(100, return_sequences=True,activation='relu'), ),)
    # model.add(Bidirectional(LSTM(100, activation='relu'), ),)
    # model.add(LSTM(100, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True, activation='relu'),)
    # model.add(Dropout(0.2))
    # model.add(LSTM(100,  return_sequences=True, activation='relu',), )
    # model.add(Dropout(0.2))
    # model.add(LSTM(100,  return_sequences=True, activation='relu'), )
    # model.add(Dropout(0.2))
    # model.add(LSTM(100,  activation='relu'), )
    # model.add(Dropout(0.2))
    model.add(Dense(100,activation = 'relu',input_shape=(train_X.shape[1],)))
    # model.add(Dense(30,activation = 'relu'))
    # model.add(Dense(200,activation = 'relu'))
    model.add(Dense(100,activation = 'relu'))
    # model.add(Dense(30,activation = 'relu'))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    # fit network
    history = model.fit(train_X, train_y, epochs=100, batch_size=256,
                        validation_data=(test_X, test_y), verbose=1, shuffle=False)
    # make a prediction
    # pred_y_train = model.predict(train_X)
    pred_y = model.predict(test_X)
    # print(test_X.shape)
    # print(pred_y.shape)
    return history, pred_y, model

def fit_tl_model(reset_model, train_X, train_y, test_X, test_y):
    model = Sequential([
        reset_model,
        Dense(100,activation = 'relu'),
        Dense(100,activation = 'relu'),
        Dense(100,activation = 'relu'),
        Dense(100,activation = 'relu'),
        Dense(1),
    ])
    # adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='mse', optimizer='adam')
    history = model.fit(train_X, train_y, epochs=40, batch_size=256,
                        validation_data=(test_X, test_y), verbose=2, shuffle=False)
    pred_y = model.predict(test_X)
    pred_y_train = model.predict(train_X)
    return history, pred_y_train, pred_y, model


def load_reset_model(filename, layername):
    base_model = load_model(filename)
    base_model.summary()
    reset_model = Model(inputs=base_model.input, outputs=base_model.get_layer(layername).output)
    reset_model.summary()
    reset_model.trainable = False
    return reset_model

def plot_curve(data, true, predicted, title):
    data_time = data['time']
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1, 1, 1)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.xaxis.set_major_locator(mdates.DayLocator(), )
    plt.title(title)
    plt.plot(data_time, predicted, 'r-o',linewidth=1, label='Prediction')
    plt.plot(data_time, true, 'b-o',linewidth=1, label='true')
    for tick in ax1.get_xticklabels():  # Rotation coordinates
        tick.set_rotation(10)
    for ind, label in enumerate(ax1.xaxis.get_ticklabels()):  # Partial display coordinates
        if ind % 7 == 0:  # adjust this part 'iterval == start point'
            label.set_visible(True)
        else:
            label.set_visible(False)
    plt.legend()
    # plt.show()


def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title='Confusion matrix: DEBE051, 2014-2018',):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, cmap=plt.cm.Greens)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes,) # rotation=45 if necessary
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    # for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #     plt.text(j, i, format(cm[i, j], fmt),horizontalalignment="center",)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.legend()
    # plt.show()

def data_inv(x, pred_y):
    x_max = np.max(x)
    x_min = np.min(x)
    x_m = x_max - x_min
    inv_yhat = pred_y * x_m + x_min
    return inv_yhat

# Prediction
def result_evalu_show(true, pred):
    rmse = sqrt(mean_squared_error(true, pred))
    Corr = np.corrcoef(true, pred)
    xy = np.vstack([true,pred])  #  将两个维度的数据叠加
    # z = gaussian_kde(xy)(xy) 
    # mae = np.mean(abs(true - pred))
    # df = pd.DataFrame()
    # df['a'] = true
    # df['b'] = pred
    # print('Mae: ' + str(mae))
    print('RMSE: ' + str(rmse))
    print('R2:')
    print(r2_score(true,pred, multioutput= 'uniform_average'))
    # print('Corr:')
    # print(Corr)
    # print()
    # Create figure and axis objects
    # Filter out data points where the observed value is zero
    mask = test_yy != 0
    filtered_pred = pred[mask]
    filtered_test_yy = test_yy[mask]

#     obsvspd = pd.DataFrame({
#     'Observed_MDA8_Concentration': filtered_test_yy,
#     'Predicted_Ozone': filtered_pred
# })
#     obsvspd.to_csv('filtered_ozone_data_eeia_rf.csv', index=False)
    # Create figure and axis objects
    fig, ax = plt.subplots(figsize=(8, 6))  # Adjust the figure size as needed

    # Create a scatter plot
    scatter = ax.scatter(filtered_test_yy, filtered_pred, alpha=0.5, edgecolors='w', label='Data Points')

    # Fit a line to the filtered data
    slope, intercept, r_value, p_value, std_err = stats.linregress(filtered_test_yy, filtered_pred)
    line = slope * filtered_test_yy + intercept

    # Plot the fitted line
    ax.plot(filtered_test_yy, line, color='red', label=f'Fit: y={slope:.2f}x+{intercept:.2f}')

    # Labeling the plot
    ax.set_xlabel('Observed MDA8 Concentration (ug/m³)', fontsize=12)
    ax.set_ylabel('Predicted Ozone (ug/m³)', fontsize=12)
    ax.set_title('EEIA-RF Model', fontsize=14)
    ax.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9)

    # Improve layout
    plt.tight_layout()

    # Optional: Set grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7)

    # # Save the figure
    # plt.savefig('ozone_scatter_plot.png', dpi=300)  # High resolution for publication

    # # Show the plot
    # plt.show()





# RNN Reshape
time_step = 1
feature_num = 7
re_values = re_input(values, n_time=time_step, n_feature=feature_num)
re_values_scaled = re_input(values_scaled, n_time=time_step, n_feature=feature_num)


# # Real-time split
# # test_size = 1
# test_size = int(re_values.shape[0]/5)
# train_X, train_y, test_X, test_y = split_data_real(re_values_scaled, test_size,
#                                                 n_time=time_step,seq=0)
# train_XX, train_yy, test_XX, test_yy = split_data_real(re_values, test_size,
#                                                 n_time=time_step,seq=0)

# print('test_y:')
# print(test_y)

# Random split
r_state = 10
ts = 0.2

train_X, test_X, train_y, test_y = train_test_split(re_values_scaled[:,1:],re_values_scaled[:,0], test_size=ts, random_state = r_state)

train_XX, test_XX, train_yy, test_yy = train_test_split(re_values[:,1:],re_values[:,0], test_size=ts,random_state = r_state)



# # # Model Training
print('Job Start')

# # Train the model with Keras
# history, pred_y, model = fit_model(train_X, train_y, test_X, test_y)

# # Train the model with sklearn

forest = RandomForestRegressor(n_estimators=100,criterion='squared_error',bootstrap=True)
# forest = MLPRegressor()
# forest =  LinearRegression()
model = forest.fit(train_X,train_y,)
# model = forest.fit(enhanced_X, enhanced_y)
# model = forest.fit(train_X_resampled , train_y_resampled )
# model = forest.fit(train_X,train_y,sample_weight=sample_weights)
print(model)
pred_y = model.predict(test_X)
pred_y_train = model.predict(train_X)


# # load LE DATA
# le_result = pd.read_csv('Berlin_le_2018.csv')
# pred = le_result.values.reshape([-1,])

# # Load the model 
# print('Model load ...')
# model = joblib.load("forest_model_rg_munchen_eeia.joblib")
# pred_y = model.predict(test_X)
# pred_y_train = model.predict(train_X)

# # Inverse data
pred = data_inv(values[:, 0], pred_y).reshape([-1,])
pred_train = data_inv(values[:, 0], pred_y_train)

# pred = pred + 1
# # print('Train Size:'+ str(enhanced_X.shape))
# print('Train Size:'+ str(train_X_resampled.shape))
# # print('Train Size:'+ str(train_XX.shape))
# print('Test Size: '+ str(pred.shape))
# print('Training Error:')
# result_evalu_show(train_yy, pred_train)
# print()
print('Test Error:')
result_evalu_show(test_yy, pred)

y_true = rg_2_cl(test_yy)
y_pred = rg_2_cl(pred)
class_names = np.array(['<120','>120'])
plot_confusion_matrix(y_true,y_pred,class_names,title='Confusion matrix', normalize=False)
plot_confusion_matrix(y_true,y_pred,class_names,title='Confusion matrix',normalize=True)



# # Determine the number of samples to add and remove
# num_samples_to_add = 6000 # Adjust as needed
# num_samples_to_remove = num_samples_to_add

# # Generate random indices
# random_indices = np.random.choice(train_X.shape[0], num_samples_to_add, replace=True)

# # Randomly duplicate exceedance samples
# gen_exceedance_X = train_X[random_indices]
# gen_exceedance_XX = train_XX[random_indices] # data without normalization
# gen_exceedance_y = model.predict(gen_exceedance_X)
# gen_exceedance_y_inv = data_inv(values[:, 0], gen_exceedance_y).reshape([-1,])
# print(gen_exceedance_X.shape)
# print(gen_exceedance_y.shape)

# # Merge and shuffle
# new_sample = np.concatenate([gen_exceedance_y_inv.reshape([-1,1]), gen_exceedance_XX], axis=1)

# new_df = pd.DataFrame(new_sample)
# print(new_df.describe())
# new_df.to_csv("C:/Work/ORM/Data/clustered_data/gen_new_sample.csv",index=False)


# print('Start Exceedance data Generation')
# # gen_new_sample_part_1 = np.concatenate([pred_train.reshape([-1,1]), train_XX], axis=1)
# # gen_new_sample_part_2 = np.concatenate([pred.reshape([-1,1]), test_XX], axis=1)
# # new_sample =  np.concatenate([gen_new_sample_part_1,gen_new_sample_part_2],axis=0)

# new_sample = np.concatenate([pred_train.reshape([-1,1]), train_XX], axis=1)
# new_df = pd.DataFrame(new_sample)
# print(new_df.describe())
# new_df.to_csv("C:/Work/ORM/Data/clustered_data/new_sample.csv",index=False)
# print('Finihsh EEIA Generation, Save in csv files')

# joblib.dump(model, "forest_model_rg_cluster2_eeia.joblib")
# joblib.dump(model, "forest_model_rg_munchen_eeia.joblib")

# result_saved = np.concatenate([pred.reshape([-1,1]), test_yy.reshape([-1,1])], axis=1)
# result_df = pd.DataFrame(result_saved)
# print(result_df.describe())
# result_df.to_csv("result_rf.csv",)

# result = permutation_importance(model, test_X, test_y, scoring = 'neg_mean_squared_error',n_repeats=10,random_state=0)
# # feature_name = ['Rain','RH','SWR', 'T','WD', 'WS']
# feature_name = ['QQ','HU','TG', 'TX','RR', 'PP','DAY','MONTH','YEAR','LAT','LON','HEIGHT','NOx','NMVOC','CH4']
# forest_importances = pd.Series(result.importances_mean, index=feature_name)
# print(forest_importances)
# print()
# fig, ax = plt.subplots()
# forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
# ax.set_title("Feature importances of NN at station DEBE010 (Whole Year model)")
# ax.set_ylabel("MSE decrease")
# fig.tight_layout()
# plt.show()

# fi_save = np.zeros([15,15])
# for i in range(15):
#     result = permutation_importance(forest, train_X[365*i:365*(i+1),:], train_y[365*i:365*(i+1)], scoring = 'neg_mean_squared_error',n_repeats=10,random_state=0)
#     # feature_name = ['QQ','HU','TG', 'TX','RR', 'PP']
#     feature_name = ['QQ','HU','TG', 'TX','RR', 'PP','DAY','MONTH','YEAR','LAT','LON','HEIGHT','NOx','NMVOC','CH4']
#     importances = result.importances_mean
#     fi_save[i,:] = importances
#     print('Finish '+str(i)+' round.')
#     print(importances)
# # print(fi_save)
# # np.savetxt('fi_train.txt',fi_save)
# print(importances)
# forest_importances = pd.Series(result.importances_mean, index=feature_name)
# print(forest_importances)
# print()
# fig, ax = plt.subplots()
# forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
# ax.set_title("Feature importances using permutation on test sets")
# ax.set_ylabel("MSE decrease")
# fig.tight_layout()
# plt.show()


# # Feature importance
# importances = forest.feature_importances_
# # std = np.std([tree.feature_importances_ for tree in forest.estimators_],axis=0) # comment if using GB
# indices = np.argsort(importances)[::-1]
# # feature_name = ['CO', 'NO2', 'NO', 'NOX', 'PM10', 'PM2.5', 'SO2','TX', 'TN', 'TG', 'RR', 'PP','QQ']
# # feature_name = ['QQ','HU','TG', 'TX','RR', 'PP','FG','NOX','NMVOC','CH4']
# # feature_name = ['QQ','HU','TG', 'TX','RR', 'PP','FG','DAY','MONTH','NOx','NMVOC','CH4']
# feature_name = ['QQ','HU','TG', 'TX','RR', 'PP','FG']
# # feature_name = ['O(t-2)','NOx(t-2)','O(t-1)','NOx(t-1)','O(t)','NOx(t)',
# # 'T(t-1)','SWR(t-1)','R(t-1)','RH(t-1)','WS(t-1)','WD(t-1)',
# # 'T(t)','SWR(t)','R(t)','RH(t)','WS(t)','WD(t)',
# # # 'T(t+1)','SWR(t+1)','R(t+1)','RH(t+1)','WS(t+1)','WD(t+1)',]
# # sorted_name = np.array(feature_name)[indices]
# # print(feature_name)
# # print(importances)
# # # print(indices)
# # # print(sorted_name)

# # np.savetxt('feature_ip.txt', importances,fmt='%.04f')

# feature_num = 7
# # Print the feature ranking
# # print("Feature ranking:")

# for f in range(feature_num):
#    print(str(f+1),') ',feature_name[indices[f]],': ', importances[indices[f]])


# forest_importances = pd.Series(importances, index=feature_name)

# fig, ax = plt.subplots()
# forest_importances.plot.bar(ax=ax)
# ax.set_title("Feature importances using MDI (Cluster 3)")
# ax.set_ylabel("Mean decrease in impurity")
# fig.tight_layout()
# plt.show()

# # save the model
# # model.save('month_model.h5')
