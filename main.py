
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import pandas_datareader as web

from Shampoo import Shampoo
from expSmoth import ExpSmoothing
from myRLS import myRLS


def seriesTest():
    start = datetime.date(2007, 7, 11)
    end = datetime.date(2021, 3, 7)
    f = web.DataReader('SPY', 'iex', start, end, api_key='pk_f50c1a4af6cc468a9fd0d853f0a5478c')
    test_size = len(f)
    y = f['close'].values
    lam = 0.95
    num_vars = 5
    LS = myRLS(num_vars, lam)
    pred_x = []
    pred_y = []
    pred_error = []

    X = np.matrix(np.zeros((1, num_vars)))
    for t in range(test_size - num_vars):

        for j in range(num_vars):
            X[0, j] = y[t + j]
        y1 = y[t + num_vars]  # predict

        pred_x.append(t)
        pred_y.append(float(X * LS.w))

        print("index, prediction, actual", t+num_vars +1, float(X * LS.w), float(y1))
        pred_error.append(LS.get_error())
        LS.add_obs(X.T, y1)
    ax = plt.plot(pred_x[0:], pred_y[0:], label='predicted')
    _ = plt.plot(pred_x[0:], y[num_vars:], label='actual')
    _ = plt.plot(pred_x[0:], pred_error[0:], label='Error/residual')
    plt.text(2, 250, 'lag =5, forget=0.95')

    plt.title("SPY stock indicator closing price, 11/7/2014 - 3/7/2021")
    plt.legend()
    plt.show()


def iceCream():
    df_ice_cream = pd.read_csv('ice_cream.csv')
    df_ice_cream.rename(columns={'DATE': 'date', 'IPN31152N': 'production'}, inplace=True)
    df_ice_cream['date'] = pd.to_datetime(df_ice_cream.date)
    df_ice_cream.set_index('date', inplace=True)
    #start_date = pd.to_datetime('2010-01-01')
    start_date = pd.to_datetime('1980-01-01')
    df_ice_cream = df_ice_cream[start_date:]
    y = pd.DataFrame(df_ice_cream)
    df = y['production'].values
    test_size = len(df)
    lam = 0.95
    num_vars = 3
    LS = myRLS(num_vars, lam)
    pred_x = []
    pred_y = []
    X = np.matrix(np.zeros((1, num_vars)))
    pred_error = []
    for t in range(test_size - num_vars):
        for j in range(num_vars):
            X[0, j] = df[t+j]
        y1 = df[t + num_vars]  # predict
        pred_x.append(t)
        pred_y.append(float(X * LS.w))

        print("index, prediction, output", t+num_vars+1,  float(X * LS.w), float(y1))
        pred_error.append(LS.get_error())
        LS.add_obs(X.T, y1)
        # sum += abs(df[i] - pred_y[i])
        # sumq += (df[i] - pred_y[i]) ** 2
    ax = plt.plot(pred_x[0:], pred_y[0:], label='predicted')
    _ = plt.plot(pred_x[0:], df[num_vars:], label='actual')
    _ = plt.plot(pred_x[0:], pred_error[0:], label='Error/residual')

    plt.title("ice cream production prediction")
    plt.text(60, 50, 'lag = 3, forget=0.95')

    #########################################################

    plt.legend()
    plt.show()







# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #seriesTest()
    #iceCream()
    #exp = ExpSmoothing(0.5)
    #exp.expSmoothing()
    sh = Shampoo()
    sh.arma()


