# Stock Indexes: Statistical measure of the value of a portion of overall stock market, price of an index is
# typically weighted average of prices of selected stock EG. DJIA, it consists of 30 of the most significant stocks
# in the U.S. such as Microsoft, Apple, Walt Disney etc. Each stock has an Open(Starting price for a given day),
# Close(Final price on that day), High(Max price on that day), Low, Volume(number of shares traded for the day)
# Features available close prices, historical as well as current open prices as well as historical performance(high,
# low and volume) Note *we don't use high, low or volume for each day it isn't realistic to foresee them* These
# aren't enough so we have to create our own features such as averages for a given week(5 trading days),
# month and year, ratio between average values, volume traded over a period of time. Stock volatility is also
# important.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor  # SGD based algorithms are sensitive to data with different scale
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

dataset = pd.read_csv('DJI.csv', index_col='Date')


def add_original_feature(df, df_new):
    df_new['open'] = df['Open']
    df_new['open_1'] = df['Open'].shift(1)
    df_new['close_1'] = df['Close'].shift(1)
    df_new['high_1'] = df['High'].shift(1)
    df_new['low_1'] = df['Low'].shift(1)
    df_new['volume_1'] = df['Volume'].shift(1)


def add_avg_price(df, df_new):
    df_new['avg_price_7'] = df['Close'].rolling(5).mean().shift(1)
    df_new['avg_price_30'] = df['Close'].rolling(21).mean().shift(1)
    df_new['avg_price_365'] = df['Close'].rolling(252).mean().shift(1)
    df_new['ratio_avg_price_7_30'] = df_new['avg_price_7'] / df_new['avg_price_30']
    df_new['ratio_avg_price_7_365'] = df_new['avg_price_7'] / df_new['avg_price_365']
    df_new['ratio_avg_price_30_365'] = df_new['avg_price_30'] / df_new['avg_price_365']


def add_avg_volume(df, df_new):
    df_new['avg_volume_7'] = df['Volume'].rolling(5).mean().shift(1)
    df_new['avg_volume_30'] = df['Volume'].rolling(21).mean().shift(1)
    df_new['avg_volume_365'] = df['Volume'].rolling(252).mean().shift(1)
    df_new['ratio_avg_volume_7_30'] = df_new['avg_volume_7'] / df_new['avg_volume_30']
    df_new['ratio_avg_volume_7_365'] = df_new['avg_volume_7'] / df_new['avg_volume_365']
    df_new['ratio_avg_volume_30_365'] = df_new['avg_volume_30'] / df_new['avg_volume_365']


def add_std_price(df, df_new):
    df_new['std_price_7'] = df['Close'].rolling(5).std().shift(1)
    df_new['std_price_30'] = df['Close'].rolling(21).std().shift(1)
    df_new['std_price_365'] = df['Close'].rolling(252).std().shift(1)
    df_new['ratio_std_price_7_30'] = df_new['std_price_7'] / df_new['std_price_30']
    df_new['ratio_std_price_7_365'] = df_new['std_price_7'] / df_new['std_price_365']
    df_new['ratio_std_price_30_365'] = df_new['std_price_30'] / df_new['std_price_365']


def add_std_volume(df, df_new):
    df_new['std_volume_7'] = df['Volume'].rolling(5).std().shift(1)
    df_new['std_volume_30'] = df['Volume'].rolling(21).std().shift(1)
    df_new['std_volume_365'] = df['Volume'].rolling(252).std().shift(1)
    df_new['ratio_std_volume_7_30'] = df_new['std_volume_7'] / df_new['std_volume_30']
    df_new['ratio_std_volume_7_365'] = df_new['std_volume_7'] / df_new['std_volume_365']
    df_new['ratio_std_volume_30_365'] = df_new['std_volume_30'] / df_new['std_volume_365']


def add_return_feature(df, df_new):
    df_new['return_1'] = ((df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1)).shift(1)
    df_new['return_7'] = ((df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5)).shift(1)
    df_new['return_30'] = ((df['Close'] - df['Close'].shift(21)) / df['Close'].shift(21)).shift(1)
    df_new['return_365'] = ((df['Close'] - df['Close'].shift(252)) / df['Close'].shift(252)).shift(1)
    df_new['moving_avg_7'] = df_new['return_1'].rolling(5).mean().shift(1)
    df_new['moving_avg_30'] = df_new['return_1'].rolling(21).mean().shift(1)
    df_new['moving_avg_365'] = df_new['return_1'].rolling(252).mean().shift(1)


def generate_features(df):
    """
    Generate features for a stock/index based on historical values
    :param df: dataframe with columns "Open", "Close", "High", "Volume", "Low", "Adjusted Close"
    :return: df_new
    """
    df_new = pd.DataFrame()
    add_original_feature(df, df_new)  # 6 original features
    add_avg_price(df, df_new)  # 31 new
    add_avg_volume(df, df_new)
    add_std_price(df, df_new)
    add_std_volume(df, df_new)
    add_return_feature(df, df_new)
    df_new['close'] = df['Close']  # target value
    df_new.dropna(axis=0, inplace=True)
    return df_new


data = generate_features(dataset)
train_end = '2019-09-01'
X_train = data.loc[:train_end, :].drop('close', axis=1).values
X_test = data.loc[train_end:, :].drop('close', axis=1).values
Y_train = data.loc[:train_end, :]['close'].values
Y_test = data.loc[train_end:, :]['close'].values

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# plt.scatter(y=data.close, x=data.index)
# plt.show()

# ******* SGDRegressor ********
param_grid = {
    "alpha": [1e-5, 3e-5, 1e-4],
    "eta0": [0.01, 0.03, 0.1]
}
regressor = SGDRegressor(penalty='l2', max_iter=1000)
grid_search = GridSearchCV(regressor, param_grid, cv=5, scoring='r2')
grid_search.fit(X_train_scaled, Y_train)
print(grid_search.best_params_)
regressor_best = grid_search.best_estimator_
pred = regressor_best.predict(X_test_scaled)

print("MSE: {0:.3f}".format(mean_squared_error(Y_test, pred)))  # 32287.737
print("MAE: {0:.3f}".format(mean_absolute_error(Y_test, pred)))  # 135.631
print("R2 Score: {0:.3f}".format(r2_score(Y_test, pred)))  # .938

plt.scatter(data.loc[train_end:, :].index, pred)
plt.scatter(data.loc[train_end:, :].index, Y_test, c='red')
plt.show()

# ******* RandomForest ********
param_grid = {
    "max_depth": [40, 50, 70],
    "min_samples_split": [5, 10, 50],
    "max_features": ['auto', 'sqrt'],
    "min_samples_leaf": [3, 5],
    "n_estimators": [500, 1000]
}

rf = RandomForestRegressor(n_jobs=-1)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, Y_train)
print(grid_search.best_params_)  # {'max_depth': 70, 'max_features': 'auto', 'min_samples_leaf': 5,
# 'min_samples_split': 5, 'n_estimators': 1000}
rf_best = grid_search.best_estimator_
pred = rf_best.predict(X_test)

print("MSE: {0:.3f}".format(mean_squared_error(Y_test, pred)))
print("MAE: {0:.3f}".format(mean_absolute_error(Y_test, pred)))
print("R2 Score: {0:.3f}".format(r2_score(Y_test, pred)))
# MSE: 492925.719
# MAE: 516.525
# R2 Score: 0.046
# Maybe should've used scaled training dataset?

plt.scatter(data.loc[train_end:, :].index, pred, c='green')
plt.show()

# ******* SVMRegeression ********
param_grid = [
    {
        'kernel': ['linear'],
        'C': [100, 300, 500],
        'epsilon': [0.00003, 0.0001]
    },
    {
        'kernel': ['rbf'],
        'gamma': [1e-3, 1e-4],
        'C': [10, 100, 1000],
        'epsilon': [0.00003, 0.0001]
    }
]
svr = SVR()
grid_search = GridSearchCV(svr, param_grid, cv=5, scoring='r2')
grid_search.fit(X_train_scaled, Y_train)
print(grid_search.best_params_)  # {'C': 500, 'epsilon': 3e-05, 'kernel': 'linear'}
svr_best = grid_search.best_estimator_
pred = svr_best.predict(X_test_scaled)

print("MSE: {0:.3f}".format(mean_squared_error(Y_test, pred)))
print("MAE: {0:.3f}".format(mean_absolute_error(Y_test, pred)))
print("R2 Score: {0:.3f}".format(r2_score(Y_test, pred)))
# MSE: 21860.809
# MAE: 110.410
# R2 Score: 0.958
plt.scatter(data.loc[train_end:, :].index, pred, c='green')
plt.scatter(data.loc[train_end:, :].index, Y_test, c='red')
plt.show()

# ******* Neural Network ********
param_grid = {
    'hidden_layer_sizes': [(50, 10), (30, 30)],
    'activation': ['logistic', 'tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'learning_rate_init': [0.0001, 0.0003, 0.001, 0.01],
    'alpha': [0.00003, 0.0001, 0.0003],
    'batch_size': [30, 50]
}
nn = MLPRegressor(random_state=42, max_iter=2000)
grid_search = GridSearchCV(nn, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train_scaled, Y_train)
print(grid_search.best_params_)
