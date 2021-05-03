import numpy as np
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot, pyplot as plt

from pandas import datetime
from pandas import read_csv
from pandas import DataFrame
# from statsmodels.tsa.arima.model import ARIMA
from matplotlib import pyplot

from myRLS import myRLS


class Shampoo:





    def readAndPlot(self):
        series = read_csv('shampoo.csv', header=0, parse_dates=[0], index_col=0, squeeze=True )
        print(series.head())
        series.plot()
        pyplot.show()

    def arma(self):

        def parser(x):
            return datetime.strptime('190' + x, '%Y-%m')


        series = read_csv('shampoo.csv', header=0, index_col=0, parse_dates=True, squeeze=True,
                          date_parser=parser)
        series.index = series.index.to_period('M')
        y= series.values

        print(y)

        test_size = len(y)
        lam = 0.9
        num_vars = 4
        LS = myRLS(num_vars, lam)
        pred_x = []
        pred_y = []
        X = np.matrix(np.zeros((1, num_vars)))
        pred_error = []
        for t in range(test_size - num_vars):
            for j in range(num_vars):
                X[0, j] = y[t + j]
            y1 = y[t + num_vars]  # predict
            pred_x.append(t)
            pred_y.append(float(X * LS.w))

            print("index, prediction, output", t + num_vars + 1, float(X * LS.w), float(y1))
            pred_error.append(LS.get_error())
            LS.add_obs(X.T, y1)
            # sum += abs(df[i] - pred_y[i])
            # sumq += (df[i] - pred_y[i]) ** 2
        ax = plt.plot(pred_x[0:], pred_y[0:], label='predicted')
        _ = plt.plot(pred_x[0:], y[num_vars:], label='actual')
        #_ = plt.plot(pred_x[0:], pred_error[0:], label='Error/residual')

        plt.title("shampoo sales prediction")
        plt.text(5, 400, 'lag = 4, forget=0.9')
        plt.legend()
        plt.show()
